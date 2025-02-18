import json

import mne
import numpy as np
import mne.io
import soundfile as sf
import pandas as pd
from sklearn.preprocessing import RobustScaler
import jsonlines
import os
import io
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer)
from petrel_client.client import Client
import torch
import librosa
from IPython.display import Audio, display
import tqdm
import tempfile

# Main function description
# This is mainly to create speech of the same length and interception in different situations, which can increase generalization.

# The main control parameters are
# Voice interception length

# Since each task corresponds to a meg and some segmented speech, it can be cut according to the divided speech.

# After division, you can split again, it doesn’t matter here.
# The main workflow is
# 1, find all tsv files
# 2, find several audio files played and their corresponding transcription files in each tsv file
# 3. Cut the audio file corresponding to the EEG, slide it for 1s±0.5s to cut, cut each 4s segment, and use this time to find the corresponding words for annotation.
# 4. Save the cut fragments first, and then identify them after saving.
# 5. Take each piece of speech separately for speech recognition. Form the final jsonl file

# In the upgrade plan, the data will not be cut, but it will be done in the dataloader.

from torch.nn import functional as F
import numpy as np

def extract_words_in_range(words, start_time, end_time):
    """
    从给定的时间范围内提取单词。

    参数:
    words (list): 单词列表，每个单词是一个字典，包含'word', 'start', 'end'键。
    start_time (float): 时间范围的开始时间。
    end_time (float): 时间范围的结束时间。

    返回:
    list: 在时间范围内的单词列表。
    """
    # 筛选出在指定时间范围内的单词
    # 这个有点坑，如果是数字之类的，他不给start 和 end 的键
    start_index = None
    end_index = None

    for i, word in enumerate(words):
        if 'start' in word and word['start'] >= start_time:
            if start_index is None:
                start_index = i

        if 'end' in word and word['end'] >= end_time:
            if end_index is None:
                end_index = i

    if start_index is not None and end_index is not None:
        return words[start_index:end_index + 1]
    elif start_index is not None:
        return words[start_index:]
    elif end_index is not None:
        return words[:end_index + 1]
    else:
        return []


def get_sequences(tsv_path):
    file_content = client.get(tsv_path)
    # import pdb;pdb.set_trace()
    text = pd.read_csv(io.StringIO(file_content.decode('utf-8')), sep='\t')
    slice_dict_list = []
    for i in range(len(text)):
        tti = eval(text['trial_type'][i])
        if 'sequence_id' not in tti.keys():
            tti['sound'] = os.path.basename(tti['sound']).split('.')[0].lower()
            tti['story'] = tti['story'].lower()

            tti1 = eval(text['trial_type'][i + 1])
            tti['speech_rate'] = tti1['speech_rate']
            tti['voice'] = tti1['voice']

            slice_dict = {'sample': text.iloc[i]['sample'],
                          'onset': text.iloc[i]['onset'],
                          **tti}
            slice_dict_list.append(slice_dict)

    sentences = []

    # Merge the transcribed text of several slices
    # First find all the transcribed texts
    # Filter out the transcript of the story versus drink
    # Arrange text in order
    # Merge text
    # Give information about each sentence
    wav_dir = os.path.join(audio_folder_path, 'wav')
    transcription_dir = os.path.join(audio_folder_path, 'transcription')
    # transcription_files=os.listdir(transcription_dir)
    # story=slice_dict_list[0]['story'].lower()
    # transcription_files=[file for file in transcription_files if file.startswith(story)]
    # transcription_files=sorted(transcription_files, key=lambda x: int(x[len(story):].split('_')[0]))

    # Read sequentially played audio files from slice_dict
    # Find the corresponding text file
    # Extract and save information at sentence level
    for slice_dict in slice_dict_list:
        transcription_file_name = slice_dict['sound'] + '_16kHz.json'
        transcription_file_name = os.path.join(transcription_dir, transcription_file_name)
        audio_file_name = slice_dict['sound'] + '_16kHz.wav'
        audio_file_name = os.path.join(wav_dir, audio_file_name)
        transcription = read_json(transcription_file_name)
        language = transcription['language']
        onset = slice_dict['onset']  # the indices of eeg
        # print(f'transcription:{len(segments)}')
        # Just rewrite this piece of code and slide to get words here.
        # Words are generated based on the word timestamps of the transcribed text here. Therefore, there is no need to loop according to the segments here.
        # We need to first extract all the words in the transcript and convert them into a list.   You may need to re-use whisper-x for annotation. It is said that the timestamp of the word will be more accurate.

        words = [word for seg in transcription['segments'] for word in seg['words']]
        # Completion Dictionary
        # for i,word in enumerate(words):
        #     if 'start' not in word:
        #         word['start']=words[i-1]['end']
        #         word['end']=words[i+1]['start']
        assert len(words)>0

        audio_file = io.BytesIO(client.get(audio_file_name))
        wav, sr = sf.read(audio_file, always_2d=True)

        audio_secs = wav.shape[0] / sr

        for start_sec in range(0, int(np.ceil(audio_secs - seg_sec)), slide_sec):
            sent = {}
            start_sec += np.random.uniform(high=1) - 0.5
            start_sec = np.clip(start_sec, 0, audio_secs - seg_sec)
            end_sec = start_sec + seg_sec
            # words Need to be selected based on time
            # print(words)
            try:
                selected_words = extract_words_in_range(words, start_time=start_sec, end_time=end_sec)
            except Exception as e:

                print(f'{transcription_file_name}出错了')
                print(e)
                print(words[0].keys())
                # selected_words=[word for word in words if word['start'] >= start_sec and word['end'] <= end_sec]
                # print(selected_words)
                for i,word in enumerate(words):
                    print(i,word)
                    print('start',word['start'])
                    # if word['start'] >= start_sec and word['end'] <= end_sec:
                    #     print(word)
                print(slice_dict)
                print(sent)
                raise Exception

            sent['words'] = selected_words
            sent['text'] = ' '.join([word['word'] for word in selected_words])
            sent['language'] = language
            sent['audio_start'] = start_sec
            sent['audio_end'] = end_sec
            sent['start'] = start_sec + onset
            sent['end'] = end_sec + onset
            sent['duration'] = seg_sec

            sent['story'] = slice_dict['story']
            sent['story_id'] = slice_dict['story_uid']
            sent['sound_id'] = slice_dict['sound_id']
            sent['speech_rate'] = slice_dict['speech_rate']
            sent['voice'] = slice_dict['voice']

            sent['meg_path'] = tsv_path[:-10] + 'meg.con'
            sent['audio_path'] = audio_file_name
            sentences.append(sent)

    return sentences


def preprocess_eeg_data(data, threshold=10):
    data = data.T

    # 2. Robust scaling
    scaler = RobustScaler()
    scaler.fit(data[:60]) # change 500 => 60
    data = scaler.transform(data).T
    # 3. Clipping outliers

    data[np.abs(data) > threshold] = np.sign(data[np.abs(data) > threshold]) * threshold
    data = data / threshold
    threshold_mask = np.abs(data) > 1
    num_clipped = np.sum(threshold_mask)

    # Calculate proportion
    clipped_ratio = num_clipped / (data.shape[0] * data.shape[1])
    assert clipped_ratio < 0.2, 'clip ratio should below 20%'
    return data, clipped_ratio


def find_files_with_extension(folder_path, extension):
    file_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.abspath(os.path.join(root, file))
                file_paths.append(file_path)

    return file_paths


def write_jsonlines(file_path, json_dicts):
    with jsonlines.open(file_path, mode='w') as writer:
        for json_dict in json_dicts:
            writer.write(json_dict)

def read_jsonlines_from_petrel(file_path):
    # 从 OSS 下载文件到内存（字节流）
    file_data = client.get(file_path)
    
    # 使用 io.BytesIO 创建内存字节流对象
    buffer = io.BytesIO(file_data)

    # 解析 jsonlines 格式的文件
    json_dicts = []
    with jsonlines.Reader(buffer) as reader:
        for json_dict in reader:
            json_dicts.append(json_dict)
    
    return json_dicts

def write_jsonlines_with_petrel(file_path, json_dicts):
    # 使用内存中的字节流模拟文件写入
    buffer = io.BytesIO()

    # 写入 JSONL 数据到字节流
    with jsonlines.Writer(buffer) as writer:
        for json_dict in json_dicts:
            writer.write(json_dict)
    
    # 在写入结束后，重置字节流指针到开头
    buffer.seek(0)

    # 使用 petrel_client 将字节流上传到远程文件
    client.put(file_path, buffer.read())

def read_raw_kit_with_petrel(meg_path, preload=True, verbose=False):
    # 下载 MEG 文件到本地临时目录
    with tempfile.NamedTemporaryFile(suffix=".con", delete=False) as temp_file:
        temp_file.write(client.get(meg_path))  # 将远程文件写入临时文件
        temp_file_path = temp_file.name  # 获取临时文件路径

    # 加载 MEG 数据
    try:
        raw = mne.io.read_raw_kit(temp_file_path, preload=preload, verbose=verbose)
    finally:
        # 删除临时文件
        os.remove(temp_file_path)
    return raw

def save_numpy_with_petrel(file_path, array):
    # 创建字节流作为文件对象
    buffer = io.BytesIO()

    # 使用 numpy.save 将数组写入字节流
    np.save(buffer, array)

    # 确保字节流指针回到开头
    buffer.seek(0)

    # 使用 petrel_client 上传字节流内容到远程存储
    client.put(file_path, buffer.read())

def write_audio_with_petrel(file_path, audio_data, sample_rate):
    # 创建字节流作为文件对象
    buffer = io.BytesIO()

    # 使用 soundfile 将音频写入字节流
    sf.write(buffer, audio_data, sample_rate, format='WAV')

    # 确保字节流指针回到开头
    buffer.seek(0)

    # 使用 petrel_client 上传字节流内容到远程存储
    client.put(file_path, buffer.read())


def process_meg(tsv_path):
    print(tsv_path,'begin')
    sentences = get_sequences(tsv_path)
    save_sentences_path=tsv_path.replace('.tsv','save_sentences_info.jsonl')
    assert save_sentences_path!=tsv_path,' these two have to be different'
    write_jsonlines_with_petrel(save_sentences_path,sentences)
    # print(tsv_path,len(sentences))
    meg_path = sentences[0]['meg_path']
    # import pdb;pdb.set_trace()
    meg = read_raw_kit_with_petrel(meg_path, preload=True, verbose=False)
    picks = mne.pick_types(
        meg.info, meg=True, ref_meg=False, eeg=False, stim=False, eog=False, ecg=False
    )
    meg.pick(picks, verbose=False)
    # meg.notch_filter(60, verbose=False)
    # meg.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    meg.resample(target_meg_sr)
    data = meg.get_data()
    assert data.shape[0] == 208, f'data shape:{data.shape}'
    old_audio_path = None
    lines = []

    for i, sent in enumerate(sentences):
        # split meg
        start_meg_index = int(sent['start'] * target_meg_sr)
        end_meg_index = int(sent['end'] * target_meg_sr)
        seg_meg = data[:, start_meg_index:end_meg_index]
        sent['duration'] = seg_meg.shape[1] / target_meg_sr
        # split audio
        audio_path = sent['audio_path']
        # import pdb;pdb.set_trace()
        if old_audio_path != audio_path:
            audio_file = io.BytesIO(client.get(audio_path))
            speech_data, speech_sr = sf.read(audio_file)
        start_audio_index = int(sent['audio_start'] * speech_sr)
        end_audio_index = int(sent['audio_end'] * speech_sr)
        seg_audio = speech_data[start_audio_index:end_audio_index]
        
        mel = processor(audio=seg_audio, sampling_rate=16000,
                             return_tensors="pt", return_attention_mask=True)
        speech_mel_input_features = mel.input_features
        speech_mel_useful_length = torch.sum(mel.attention_mask).item()
        

        # standardization
        seg_meg, cr = preprocess_eeg_data(seg_meg, threshold=threshold) # change threshold=20 for the same setting as meta
        # Store the processed audio files, EEG files, and annotation files.
        seg_meg_path = tsv_path.replace('download', replace_folder).replace('events.tsv', f'senid_{i}_meg.npy')
        seg_audio_path = seg_meg_path.replace('meg.npy', 'audio.wav')
        seg_mel_path = seg_meg_path.replace('meg.npy', 'mel.npy')
        # makedirs(seg_meg_path)
        # print(f'{i} seg_meg {seg_meg.shape} seg_meg_path {seg_meg_path}')
        save_numpy_with_petrel(seg_meg_path, seg_meg)
        save_numpy_with_petrel(seg_mel_path, speech_mel_input_features)
        # seg_meg = np.load(seg_meg_path)
        write_audio_with_petrel(seg_audio_path, seg_audio, target_speech_sr)

        # Parse other key-value pairs
        selected_keys = ['story', 'story_id', 'sound_id', 'speech_rate', 'voice']

        new_dict = {key: sent[key] for key in selected_keys}
        # whisper json
        line = {
            "speech": {"path": seg_audio_path, 'sr': target_speech_sr},
            "eeg": {"path": seg_meg_path, 'sr': target_meg_sr},
            "mel": {"path": seg_mel_path, 'speech_mel_useful_length': speech_mel_useful_length},
            "audio_start": 0,
            "audio_end": sent['audio_end'] - sent['audio_start'],
            "start": 0,
            "end": sent['duration'],
            "duration": sent['duration'],
            # "delay_sec": delay_sec,
            "language": "English",
            "sentence": sent['text'],
            "sentences": [{"text": sent['text'],
                           "start": 0.0, "end": sent['duration'], "duration": sent['duration'],
                           "words": [{"word": word['word'],
                                      "start": word['start'] - sent['audio_start'] if 'start' in word.keys() else None,
                                      "end": word['end'] - sent['audio_start'] if 'end' in word.keys() else None}
                                     for word in sent['words']]}],
            "subj": int(os.path.basename(tsv_path)[4:6]),
            **new_dict
        }
        lines.append(line)
    if len(lines) != 0:
        seg_jsonl_path = tsv_path.replace('download', replace_folder).replace('events.tsv', 'info.jsonl')
        write_jsonlines_with_petrel(seg_jsonl_path, lines)
    print(tsv_path,'done')
    return lines

def get_info(tsv_path):
    seg_jsonl_path = tsv_path.replace('download', replace_folder).replace('events.tsv', 'info.jsonl')
    lines = read_jsonlines_from_petrel(seg_jsonl_path)
    print(tsv_path,'done')
    return lines


from multiprocessing import Pool


def read_json(file_path):
    return json.loads(client.get(file_path))


def process_file(filename_id):
    lines = get_info(events_tsv_list[filename_id])
    # lines = process_meg(events_tsv_list[filename_id])
    return lines


# python process_dataset/gwilliams2023_process_240411.py
if __name__ == '__main__':
    client = Client()
    np.random.seed(0)
    home_dir = os.path.expanduser("~")
    replace_folder = 'preprocess_10_nofilter'
    folder_path = 's3://MAD/Gwilliams2023/'
    audio_folder_path = f's3://MAD/Gwilliams2023/{replace_folder}/audio'
    base_model = 'openai/whisper-base'
    language = 'en'
    task = 'transcribe'
    timestamps = False
    local_files_only = False
    extension = 'events.tsv'
    original_eeg_sr = 1000
    target_meg_sr = 100 # change 200 => 120
    target_speech_sr = 16000
    threshold = 20
    slide_sec = 5
    seg_sec = 10 # modify 05014
    delay_sec = 0.5
    processes = 32
    l_freq = 1
    h_freq = 40
    hop_length=160
    events_tsv_list = ['s3://MAD/Gwilliams2023/download/sub-14/ses-0/meg/sub-14_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-14/ses-0/meg/sub-14_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-14/ses-0/meg/sub-14_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-14/ses-0/meg/sub-14_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-14/ses-1/meg/sub-14_ses-1_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-14/ses-1/meg/sub-14_ses-1_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-14/ses-1/meg/sub-14_ses-1_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-14/ses-1/meg/sub-14_ses-1_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-18/ses-0/meg/sub-18_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-18/ses-0/meg/sub-18_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-18/ses-0/meg/sub-18_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-18/ses-0/meg/sub-18_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-18/ses-1/meg/sub-18_ses-1_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-18/ses-1/meg/sub-18_ses-1_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-18/ses-1/meg/sub-18_ses-1_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-18/ses-1/meg/sub-18_ses-1_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-02/ses-0/meg/sub-02_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-02/ses-0/meg/sub-02_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-02/ses-0/meg/sub-02_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-02/ses-0/meg/sub-02_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-02/ses-1/meg/sub-02_ses-1_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-02/ses-1/meg/sub-02_ses-1_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-02/ses-1/meg/sub-02_ses-1_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-02/ses-1/meg/sub-02_ses-1_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-03/ses-0/meg/sub-03_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-03/ses-0/meg/sub-03_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-03/ses-0/meg/sub-03_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-03/ses-0/meg/sub-03_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-21/ses-0/meg/sub-21_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-21/ses-0/meg/sub-21_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-21/ses-0/meg/sub-21_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-21/ses-0/meg/sub-21_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-12/ses-0/meg/sub-12_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-12/ses-0/meg/sub-12_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-12/ses-0/meg/sub-12_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-12/ses-0/meg/sub-12_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-25/ses-0/meg/sub-25_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-25/ses-0/meg/sub-25_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-25/ses-0/meg/sub-25_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-25/ses-0/meg/sub-25_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-25/ses-1/meg/sub-25_ses-1_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-25/ses-1/meg/sub-25_ses-1_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-25/ses-1/meg/sub-25_ses-1_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-25/ses-1/meg/sub-25_ses-1_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-01/ses-0/meg/sub-01_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-01/ses-0/meg/sub-01_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-01/ses-0/meg/sub-01_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-01/ses-0/meg/sub-01_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-01/ses-1/meg/sub-01_ses-1_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-01/ses-1/meg/sub-01_ses-1_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-01/ses-1/meg/sub-01_ses-1_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-01/ses-1/meg/sub-01_ses-1_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-04/ses-0/meg/sub-04_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-04/ses-0/meg/sub-04_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-04/ses-0/meg/sub-04_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-04/ses-0/meg/sub-04_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-04/ses-1/meg/sub-04_ses-1_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-04/ses-1/meg/sub-04_ses-1_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-04/ses-1/meg/sub-04_ses-1_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-04/ses-1/meg/sub-04_ses-1_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-27/ses-0/meg/sub-27_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-27/ses-0/meg/sub-27_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-27/ses-0/meg/sub-27_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-27/ses-0/meg/sub-27_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-27/ses-1/meg/sub-27_ses-1_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-27/ses-1/meg/sub-27_ses-1_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-27/ses-1/meg/sub-27_ses-1_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-27/ses-1/meg/sub-27_ses-1_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-22/ses-0/meg/sub-22_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-22/ses-0/meg/sub-22_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-22/ses-0/meg/sub-22_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-22/ses-0/meg/sub-22_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-22/ses-1/meg/sub-22_ses-1_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-22/ses-1/meg/sub-22_ses-1_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-22/ses-1/meg/sub-22_ses-1_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-22/ses-1/meg/sub-22_ses-1_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-10/ses-0/meg/sub-10_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-10/ses-0/meg/sub-10_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-10/ses-0/meg/sub-10_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-10/ses-0/meg/sub-10_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-10/ses-1/meg/sub-10_ses-1_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-10/ses-1/meg/sub-10_ses-1_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-10/ses-1/meg/sub-10_ses-1_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-10/ses-1/meg/sub-10_ses-1_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-09/ses-0/meg/sub-09_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-09/ses-0/meg/sub-09_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-09/ses-0/meg/sub-09_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-09/ses-0/meg/sub-09_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-09/ses-1/meg/sub-09_ses-1_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-09/ses-1/meg/sub-09_ses-1_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-09/ses-1/meg/sub-09_ses-1_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-09/ses-1/meg/sub-09_ses-1_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-16/ses-0/meg/sub-16_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-16/ses-0/meg/sub-16_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-16/ses-0/meg/sub-16_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-16/ses-0/meg/sub-16_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-05/ses-0/meg/sub-05_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-05/ses-0/meg/sub-05_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-05/ses-0/meg/sub-05_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-05/ses-0/meg/sub-05_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-05/ses-1/meg/sub-05_ses-1_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-05/ses-1/meg/sub-05_ses-1_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-05/ses-1/meg/sub-05_ses-1_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-05/ses-1/meg/sub-05_ses-1_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-08/ses-0/meg/sub-08_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-08/ses-0/meg/sub-08_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-08/ses-0/meg/sub-08_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-08/ses-0/meg/sub-08_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-08/ses-1/meg/sub-08_ses-1_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-08/ses-1/meg/sub-08_ses-1_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-08/ses-1/meg/sub-08_ses-1_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-08/ses-1/meg/sub-08_ses-1_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-15/ses-0/meg/sub-15_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-15/ses-0/meg/sub-15_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-15/ses-0/meg/sub-15_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-15/ses-0/meg/sub-15_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-15/ses-1/meg/sub-15_ses-1_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-15/ses-1/meg/sub-15_ses-1_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-15/ses-1/meg/sub-15_ses-1_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-15/ses-1/meg/sub-15_ses-1_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-13/ses-0/meg/sub-13_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-13/ses-0/meg/sub-13_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-13/ses-0/meg/sub-13_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-13/ses-0/meg/sub-13_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-13/ses-1/meg/sub-13_ses-1_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-13/ses-1/meg/sub-13_ses-1_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-13/ses-1/meg/sub-13_ses-1_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-13/ses-1/meg/sub-13_ses-1_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-20/ses-0/meg/sub-20_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-20/ses-0/meg/sub-20_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-20/ses-0/meg/sub-20_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-20/ses-0/meg/sub-20_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-06/ses-0/meg/sub-06_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-06/ses-0/meg/sub-06_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-06/ses-0/meg/sub-06_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-06/ses-0/meg/sub-06_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-06/ses-1/meg/sub-06_ses-1_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-06/ses-1/meg/sub-06_ses-1_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-06/ses-1/meg/sub-06_ses-1_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-06/ses-1/meg/sub-06_ses-1_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-17/ses-0/meg/sub-17_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-17/ses-0/meg/sub-17_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-17/ses-0/meg/sub-17_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-17/ses-0/meg/sub-17_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-17/ses-1/meg/sub-17_ses-1_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-17/ses-1/meg/sub-17_ses-1_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-17/ses-1/meg/sub-17_ses-1_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-17/ses-1/meg/sub-17_ses-1_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-19/ses-0/meg/sub-19_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-19/ses-0/meg/sub-19_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-19/ses-0/meg/sub-19_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-19/ses-0/meg/sub-19_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-19/ses-1/meg/sub-19_ses-1_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-19/ses-1/meg/sub-19_ses-1_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-19/ses-1/meg/sub-19_ses-1_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-19/ses-1/meg/sub-19_ses-1_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-23/ses-0/meg/sub-23_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-23/ses-0/meg/sub-23_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-23/ses-0/meg/sub-23_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-23/ses-0/meg/sub-23_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-23/ses-1/meg/sub-23_ses-1_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-23/ses-1/meg/sub-23_ses-1_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-23/ses-1/meg/sub-23_ses-1_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-23/ses-1/meg/sub-23_ses-1_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-07/ses-0/meg/sub-07_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-07/ses-0/meg/sub-07_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-07/ses-0/meg/sub-07_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-07/ses-0/meg/sub-07_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-07/ses-1/meg/sub-07_ses-1_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-07/ses-1/meg/sub-07_ses-1_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-07/ses-1/meg/sub-07_ses-1_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-07/ses-1/meg/sub-07_ses-1_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-26/ses-0/meg/sub-26_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-26/ses-0/meg/sub-26_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-26/ses-0/meg/sub-26_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-26/ses-0/meg/sub-26_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-26/ses-1/meg/sub-26_ses-1_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-26/ses-1/meg/sub-26_ses-1_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-26/ses-1/meg/sub-26_ses-1_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-26/ses-1/meg/sub-26_ses-1_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-24/ses-0/meg/sub-24_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-24/ses-0/meg/sub-24_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-24/ses-0/meg/sub-24_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-24/ses-0/meg/sub-24_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-24/ses-1/meg/sub-24_ses-1_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-24/ses-1/meg/sub-24_ses-1_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-24/ses-1/meg/sub-24_ses-1_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-24/ses-1/meg/sub-24_ses-1_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-11/ses-0/meg/sub-11_ses-0_task-0_events.tsv', 's3://MAD/Gwilliams2023/download/sub-11/ses-0/meg/sub-11_ses-0_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-11/ses-0/meg/sub-11_ses-0_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-11/ses-0/meg/sub-11_ses-0_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-11/ses-1/meg/sub-11_ses-1_task-1_events.tsv', 's3://MAD/Gwilliams2023/download/sub-11/ses-1/meg/sub-11_ses-1_task-2_events.tsv', 's3://MAD/Gwilliams2023/download/sub-11/ses-1/meg/sub-11_ses-1_task-3_events.tsv', 's3://MAD/Gwilliams2023/download/sub-11/ses-1/meg/sub-11_ses-1_task-0_events.tsv']
    processor = WhisperProcessor.from_pretrained(base_model,
                                                 language=language,
                                                 task=task,
                                                 no_timestamps=not timestamps,
                                                 local_files_only=local_files_only,
                                                 hop_length=hop_length) # equal to Meta
    # results=[process_file(file) for file in events_tsv_list[:2]]
    # process_file(0)


    pool = Pool(processes=processes)
    results = pool.map(process_file, np.arange(len(events_tsv_list)))
    pool.close()
    pool.join()

    all_lines = []
    for lines in results:
        all_lines.extend(lines)

    write_jsonlines_with_petrel(os.path.join(folder_path + replace_folder, 'info.jsonl'), all_lines)
