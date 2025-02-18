import json
import os
import io
from petrel_client.client import Client
import random
import sys
import time
from typing import List

import yaml
from torch.utils.data.dataloader import DataLoader
from utils.augment_eeg import RandomShapeMasker, shift_data
import librosa
import numpy as np
import soundfile
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,random_split
from tqdm import tqdm
from utils.binary import DatasetReader
from utils.utils import preprocess_eeg_data, lowpass_filter, add_gaussian_noise, read_yaml
from data_augmentation_utils.full_aug import FullAug, EEGAug, EEGTextAug
import jsonlines
from utils.process_utils import torch_random_choices, read_jsonlines_from_petrel, write_jsonlines_with_petrel
import re
import copy
from transformers import AutoTokenizer
import textgrid as tg

client = Client()

def filter_ascii_str(text):
    return re.sub(r'[^a-zA-Z ]', '', text)


def filter_ascii_data_dict(data_dict):
    data_dict['sentence'] = filter_ascii_str(data_dict['sentence'])
    for i, sentence in enumerate(data_dict['sentences']):
        sentence['text'] = filter_ascii_str(sentence['text'])
        if "words" in sentence.keys():
            for j, w in enumerate(sentence['words']):
                w['word'] = filter_ascii_str(w['word'])
    return data_dict

def create_mfaphones():
    phones = "AA0 AA1 AA2 AE0 AE1 AE2 AH0 AH1 AH2 AO0 AO1 AO2 AW0 AW1 AW2 AY0 AY1 AY2 B CH D DH EH0 EH1 EH2 ER0 ER1 ER2 EY0 EY1 EY2 F G HH IH0 IH1 IH2 IY0 IY1 IY2 JH K L M N NG OW0 OW1 OW2 OY0 OY1 OY2 P R S SH T TH UH0 UH1 UH2 UW0 UW1 UW2 V W Y Z ZH"
    mfa_phones = phones.split()
    simp_phones = []
    for phone in mfa_phones:
        simp = re.sub(r'\d+$', '', phone)
        if simp not in simp_phones:
            simp_phones.append(simp)
    vowels = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
    consonants = ['B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']

    return simp_phones, vowels, consonants



class BetterDataset(Dataset):
    def __init__(self,
                 data_list_path,
                 processor,
                 data_list_dir='/home/yyang/dataset/multi_media/',
                 augment_config_path=None,
                 modal=('eeg', 'mel', 'subject_index'),
                 modal_ch=None,
                 language=None,
                 timestamps=False,
                 sample_rate=100,
                 min_duration=0.5,
                 max_duration=30,
                 level='sentence',
                 ):
        """
        The current shortcomings are mainly
         1. Some useless functions, such as filtering data sets, need to be placed elsewhere.
         2. There is no unified interface for data enhancement, and the execution order of multiple data enhancements cannot be specified in the parameters.
         3. The interface for loading data is not encapsulated with a dictionary, making it difficult to add new data. A list of loading fields should be added and all fields in this list should be loaded into the dictionary.
        """
        super().__init__()
        assert min_duration >= 0.5, f"min_duration cannot be less than 0.5, currently it is: {min_duration}"
        assert max_duration <= 30, f"max_duration cannot be greater than 30, currently it is: {max_duration}"
        self.data_list_path = data_list_path
        self.processor = processor
        self.processor_dict = {}
        self.signal_sample_rate = sample_rate
        self.language = language
        self.timestamps = timestamps
        self.level = level
        self.modal = modal
        self.modal_ch = modal_ch
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.vocab = self.processor.tokenizer.get_vocab()
        self.timestamp_begin = self.vocab['<|notimestamps|>'] + 1
        self.startoftranscript = self.vocab['<|startoftranscript|>']
        self.endoftext = self.vocab['<|endoftext|>']
        self.nocaptions = self.vocab['<|nocaptions|>']
        self.data_list: List[dict] = []
        self.phoneme_dict, _, _ = create_mfaphones()
        # Load data list
        self._load_data_list()
        # Data augmentation configuration parameters
        self.noises_path = None
        self.speed_rates = None
        self.keep_keys = [
            'labels', 'ctc_labels', 'frame_labels', 'subj', 'speech_mel_input_features', 'eeg', 'speech', 'sentence','speech_mel_useful_length',
            'eeg_raw', 'language']
        self.augmentor = self._get_aug(augment_config_path)
        self.ctc_processor = AutoTokenizer.from_pretrained('/mnt/petrelfs/zhangchi1/.cache/huggingface/hub/models--facebook--wav2vec2-base-960h/snapshots/22aad52d435eb6dbaf354bdad9b0da84ce7d6156')

    def _get_aug(self, augment_config_path):
        if augment_config_path is not None:
            with open(augment_config_path, 'r', encoding='utf-8') as f:
                augment_configs = json.load(f)
            return FullAug(configs=augment_configs['config'], augmentations=augment_configs['augmentations'])
        else:
            return FullAug()

    # Load data list
    def _load_data_list(self):
        data_list = read_jsonlines_from_petrel(self.data_list_path)
        self.data_list = data_list
        print(f'num of data:{len(self.data_list)}')

    # Get audio data, sample rate and text from the data list
    def _get_list_data(self, idx):
        unit = copy.deepcopy(self.data_list[idx])
        eeg_file = unit['eeg']["path"]
        speech_file = unit['speech']["path"]
        # load eeg, speech
        with io.BytesIO(client.get(eeg_file)) as f:
            eeg = np.load(f)
        if 'mel' not in unit.keys():
            with io.BytesIO(client.get(audio_file_name)) as f:
                speech, speech_sr = soundfile.read(f, dtype='float32', always_2d=True)
            speech = speech[:, 0]  # mono channel

            mel = self.processor(audio=speech, sampling_rate=16000,
                                 return_tensors="pt", return_attention_mask=True)
            unit['speech_mel_input_features'] = mel.input_features
            unit['speech_mel_useful_length'] = torch.sum(mel.attention_mask).item()
        else:
            with io.BytesIO(client.get(unit['mel']['path'])) as f:
                speech_mel_input_features = np.load(f)
            speech_mel_useful_length = unit['mel']['speech_mel_useful_length']
            unit['speech_mel_input_features'] = speech_mel_input_features
            unit['speech_mel_useful_length'] = speech_mel_useful_length

        unit['eeg_raw'] = eeg


        phoneme_length = round(eeg.shape[1]/2)
        aligned_file = "s3://MAD/Gwilliams2023/preprocess_10/audio_aligned/" + os.path.basename(speech_file).replace("wav", "TextGrid")
        phoneme_frame = np.zeros((1, phoneme_length))
        
        grid = tg.TextGrid.fromFile(aligned_file)
        intervals = grid.tiers[1].intervals
        for interval in intervals:
            start_time = interval.minTime
            end_time = interval.maxTime
            label = interval.mark
            if re.sub(r'\d+$', '', label) in self.phoneme_dict:
                start_point = max(round(start_time * self.signal_sample_rate/2), 0)
                end_point = min(round(end_time * self.signal_sample_rate) - 1, phoneme_length)
                for t in range(start_point, end_point+1):
                    phoneme_frame[0, t] = phoneme_dict.index(re.sub(r'\d+$', '', label)) + 1
        unit["frame_labels"] = phoneme_frame

        return unit

    def _aug(self, unit):
        unit = self.augmentor(unit)
        return unit

    def _load_timestamps_transcript(self, transcript: List[dict], processor):
        level = self.level
        if level == 'words':
            return self._load_timestamps_transcript_words(transcript, processor)
        elif level == 'sentences':
            return self._load_timestamps_transcript_sentences(transcript, processor)
        else:
            raise NotImplementedError

    def _load_timestamps_transcript_sentences(self, transcript: List[dict], processor):
        assert isinstance(transcript, list), f"transcript应该为list，当前为：{type(transcript)}"
        data = dict()
        labels = processor.tokenizer.prefix_tokens[:3]
        # print(f'transcript :{len(transcript),transcript}')
        for t in transcript:
            # Encode target text into tag ID
            start = t['start'] if round(t['start'] * 100) % 2 == 0 else t['start'] + 0.01
            start = self.timestamp_begin + round(start * 100) // 2
            end = t['end'] if round(t['end'] * 100) % 2 == 0 else t['end'] - 0.01
            end = self.timestamp_begin + round(end * 100) // 2
            label = processor(text=t['text']).input_ids[4:-1]
            # print(f'len label:{len(label)} label:{label} transcript:{transcript}')
            if max(label) > 51865:
                print(f'OOV text {t["text"]} label {label}\n')
                raise ValueError
            if start > 51865:
                print(f'OOV start {t["start"]} label {start}\n')
                raise ValueError
            if end > 51865:
                print(f'OOV start {t["end"]} label {end}\n')
                raise ValueError
            labels.extend([start])
            labels.extend(label)
            labels.extend([end])
        data['labels'] = labels + [self.endoftext]
        return data

    def _load_timestamps_transcript_words(self, transcript: List[dict], processor):
        assert isinstance(transcript, list), f"Transcript should be a list, currently it is:：{type(transcript)}"
        data = dict()
        labels = processor.tokenizer.prefix_tokens[:3]
        for t in transcript:
            # 将目标文本编码为标签ID
            words = t['words']
            for w in words:
                start = w['start'] if round(w['start'] * 100) % 2 == 0 else w['start'] + 0.01
                start = self.timestamp_begin + round(start * 100) // 2
                end = w['end'] if round(w['end'] * 100) % 2 == 0 else w['end'] - 0.01
                end = self.timestamp_begin + round(end * 100) // 2
                label = processor(text=w['word']).input_ids[4:-1]
                labels.extend([start])
                labels.extend(label)
                labels.extend([end])
        data['labels'] = labels + [self.endoftext]
        return data

    def _load_labels_for_unit(self, unit):
        t1=time.time()
        language = unit['language'] if unit['language'] is not None else self.language
        # if language is not None:
        #     if language not in self.processor_dict.keys():
        #         processor = copy.deepcopy(self.processor)
        #     else:
        #         processor = self.processor_dict[language]
        # else:
        #     processor=self.processor
        t2=time.time()
        self.processor.tokenizer.set_prefix_tokens(
            language=language)
        t3=time.time()

        transcript = unit["sentences"] if self.timestamps else unit["sentence"]
        if len(transcript) > 0:
            if self.timestamps:
                labels = self._load_timestamps_transcript(transcript, self.processor)['labels']
            else:
                labels = self.processor(text=transcript)['input_ids']
                inputs = self.ctc_processor(transcript.upper(),
                                return_tensors="pt",
                                add_special_tokens=True,
                                padding='max_length', 
                                max_length=1000)
                
                inputs = inputs["input_ids"].tolist()
                ctc_labels = torch.tensor(inputs[0])
        else:
            labels = [self.startoftranscript, self.nocaptions, self.endoftext]
            ctc_labels = torch.zeros(1000, dtype=torch.int64)
        t4=time.time()

        unit['labels'] = labels
        unit['ctc_labels'] = ctc_labels
        # print('load_labels',round(t2-t1,1),round(t3-t2,1),round(t4-t3,1))
        return unit

    def filter_unit(self, unit: dict, keys):
        return {key: unit[key] for key in unit.keys() if key in keys}

    def __getitem__(self, idx):
        t1=time.time()
        unit = self._get_list_data(idx)
        unit = self.filter_unit(unit, self.keep_keys)
        t2=time.time()
        unit = self.augmentor(unit)
        t3=time.time()
        unit = self._load_labels_for_unit(unit)
        t4=time.time()
        unit = self._pad_unit(unit)
        t5=time.time()
        unit = self.filter_unit(unit, ['labels', 'ctc_labels', 'subj', 'speech_mel_input_features',
                                       'speech_mel_useful_length','eeg_raw', 'frame_labels'])
        # print(unit.keys(),unit['speech_mel_useful_length'])
        t6=time.time()
        # print(round(t2-t1,1),round(t3-t2,1),round(t4-t3,1),round(t5-t4,1),round(t6-t1,1))
        import pdb;pdb.set_trace()
        return unit

    def __len__(self):
        return len(self.data_list)

    def _pad_unit(self, unit):
        unit['eeg_raw'] = self._padding_sample(unit['eeg_raw'])
        return unit

    def _padding_sample(self, sample):

        max_length = int(self.max_duration * self.signal_sample_rate)
        new_sample = np.zeros([self.modal_ch, max_length])
        min_ch = min(sample.shape[0], self.modal_ch)
        min_len = min(sample.shape[1], max_length)
        new_sample[:min_ch, :min_len] = sample[:min_ch, :min_len]
        return [new_sample]

