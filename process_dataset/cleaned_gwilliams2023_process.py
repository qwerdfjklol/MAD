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
from tqdm import tqdm
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


def get_sequences(fif_path):

    sentences = []
    wav_dir = os.path.join(folder_path, 'audio')
    transcription_dir = os.path.join(folder_path, 'transcriptions')

    transcription_file_name = fif_path[-15:-9] + '.json'
    transcription_file_name = os.path.join(transcription_dir, transcription_file_name)
    audio_file_name = fif_path[-15:-9] + '.wav'
    audio_file_name = os.path.join(wav_dir, audio_file_name)
    transcription = read_json(transcription_file_name)
    language = transcription['language']
    onset = 0.0  # the indices of eeg
    words = [word for seg in transcription['segments'] for word in seg['words']]
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
            for i,word in enumerate(words):
                print(i,word)
                print('start',word['start'])
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

        sent['story_id'] = float(fif_path[-10:-9])

        sent['meg_path'] = fif_path
        sent['audio_path'] = audio_file_name
        sentences.append(sent)

    # Read sequentially played audio files from slice_dict
    # Find the corresponding text file
    # Extract and save information at sentence level
    

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

def read_evokeds_with_petrel(meg_path):
    # 下载 MEG 文件到本地临时目录
    with tempfile.NamedTemporaryFile(suffix=".fif", delete=False) as temp_file:
        temp_file.write(client.get(meg_path))  # 将远程文件写入临时文件
        temp_file_path = temp_file.name  # 获取临时文件路径

    # 加载 MEG 数据
    try:
        raw = mne.read_evokeds(temp_file_path)
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


def process_meg(fif_path):
    print(fif_path,'begin')
    sentences = get_sequences(fif_path)
    meg_path = sentences[0]['meg_path']
    # import pdb;pdb.set_trace()
    meg = read_evokeds_with_petrel(meg_path)
    # meg[0].filter(l_freq=0.1, h_freq=50, verbose=False)
    meg[0].resample(target_meg_sr)
    data = meg[0].data
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
        seg_meg_path = fif_path.replace('emeg', replace_folder).replace('.fif', f'_senid_{i}_meg.npy')
        seg_audio_path = seg_meg_path.replace('meg.npy', 'audio.wav')
        seg_mel_path = seg_meg_path.replace('meg.npy', 'mel.npy')
 
        save_numpy_with_petrel(seg_meg_path, seg_meg)
        save_numpy_with_petrel(seg_mel_path, speech_mel_input_features)
        write_audio_with_petrel(seg_audio_path, seg_audio, target_speech_sr)

        # Parse other key-value pairs
        selected_keys = ['story_id']

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
            "subj": int(fif_path[-18:-16]),
            **new_dict
        }
        lines.append(line)
    if len(lines) != 0:
        seg_jsonl_path = fif_path.replace('emeg', replace_folder).replace('.fif', '_info.jsonl')
        write_jsonlines_with_petrel(seg_jsonl_path, lines)
    print(fif_path,'done')
    return lines

def get_info(fif_path):
    seg_jsonl_path = fif_path.replace('.fif', '_info.jsonl')
    lines = read_jsonlines_from_petrel(seg_jsonl_path)
    print(fif_path,'done')
    return lines

def preprocess_eeg_all(seg_meg_path):
    with io.BytesIO(client.get(seg_meg_path)) as f:
        seg_meg = np.load(f)
    # import pdb;pdb.set_trace()
    seg_meg, cr = preprocess_eeg_data(seg_meg, threshold=threshold) # change threshold=20 for the same setting as meta
    save_numpy_with_petrel(seg_meg_path, seg_meg)


from multiprocessing import Pool


def read_json(file_path):
    return json.loads(client.get(file_path))


def process_file(filename_id):
    lines = process_meg(folder_path + 'emeg_wo_ecg/' + fif_list[filename_id])
    # lines = preprocess_eeg_all(folder_path + 'preprocess_10/' + fif_list[filename_id])
    # lines = get_info(folder_path + 'preprocess_10/' + fif_list[filename_id])

    return lines


# python process_dataset/gwilliams2023_process_240411.py
if __name__ == '__main__':
    client = Client()
    np.random.seed(0)
    home_dir = os.path.expanduser("~")
    replace_folder = 'preprocess_10_nofilter_wo_ecg'
    folder_path = 's3://MAD/Gwilliams2023_cleaned/'
    base_model = 'openai/whisper-base'
    language = 'en'
    task = 'transcribe'
    timestamps = False
    local_files_only = False
    original_eeg_sr = 1000
    target_meg_sr = 100 # change 200 => 120
    target_speech_sr = 16000
    threshold = 20
    slide_sec = 5
    seg_sec = 10 # modify 05014
    delay_sec = 0.5
    processes = 32
    hop_length=160
    
    # all_fif = os.listdir('/mnt/petrelfs/zhangchi1/datasets/Gwilliams_cleaned/emeg/')
    all_fif = client.list('s3://MAD/Gwilliams2023_cleaned/emeg_wo_ecg/')
    # fif_list = [file for file in all_fif if file.endswith('meg.npy')]
    fif_list = [file for file in all_fif if file.endswith('rep0.fif') or file.endswith('rep1.fif')]


    # fifff = mne.read_evokeds('/mnt/petrelfs/zhangchi1/datasets/Gwilliams_cleaned/emeg/sub-18_task-3_rep1.fif')

    processor = WhisperProcessor.from_pretrained(base_model,
                                                 language=language,
                                                 task=task,
                                                 no_timestamps=not timestamps,
                                                 local_files_only=local_files_only,
                                                 hop_length=hop_length) # equal to Meta
    # results=[process_file(file) for file in events_tsv_list[:2]]
    # import pdb;pdb.set_trace()
    # process_file(0)

    pool = Pool(processes=processes)
    results = pool.map(process_file, np.arange(len(fif_list)))
    pool.close()
    pool.join()

    all_lines = []
    for lines in results:
        all_lines.extend(lines)

    write_jsonlines_with_petrel(os.path.join(folder_path + replace_folder, 'info.jsonl'), all_lines)
