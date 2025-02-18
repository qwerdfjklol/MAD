import argparse
import logging
logging.getLogger("boto3").setLevel(logging.WARNING)
import boto3;boto3.set_stream_logger(level=logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)
import functools
import gc
import json
import os
import torch.nn as nn
import evaluate
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoConfig
from utils.load_model import BrainWhisperForConditionalGeneration2
from utils.model_utils import projection_module
from peft import PeftModel, AdaLoraConfig, get_peft_model
from utils.data_utils import DataCollatorBrainSpeechSeq2SeqWithPadding,generate_random_string, remove_punctuation, to_simple,contains_valid_letters
from utils.process_str import filter_ascii_text, model_generate, convert_lower_text, list_operation
from utils.reader import BetterDataset
from utils.utils import print_arguments, add_arguments
from utils.generation_helper import GetSequenceBias
import pickle
import re
from metric import Calculate_MCD
import torchaudio
import librosa
import matplotlib.pyplot as plt
import soundfile
from PIL import Image
import seaborn as sns

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("test_data",   type=str, default="/data/johj/MEG/gwilliams2023/preprocess7/split3/cable_spool_fort/lw1/train.jsonl", help="test set")
add_arg("checkpoint_path",  type=str, default="models/whisper-tiny-finetune", help="full model checkpoint path")
add_arg("model_path",    type=str, default="/data/johj/MEG/transformer_whisper_models", help="whisper")
add_arg("modal", type=str, default='speech',  help="输入的模态")
add_arg("sampling_rate", type=int, default=120,  help="输入信号采样率")
add_arg("eeg_ch", type=int, default=208,  help="输入信号通道数")
add_arg("batch_size",  type=int, default=16,        help="评估的batch size")
add_arg("num_workers", type=int, default=8,         help="读取数据的线程数量")
add_arg("language",    type=str, default="Chinese", help="设置语言，可全称也可简写，如果为None则评估的是多语言")
add_arg("remove_pun",  type=bool, default=True,     help="是否移除标点符号")
add_arg("to_simple",   type=bool, default=True,     help="是否转为简体中文")
add_arg("timestamps",  type=bool, default=False,    help="评估时是否使用时间戳数据")
add_arg("min_audio_len",     type=float, default=0.5,  help="最小的音频长度，单位秒")
add_arg("max_audio_len",     type=float, default=30,   help="最大的音频长度，单位秒")
add_arg("local_files_only",  type=bool,  default=True, help="是否只在本地加载模型，不尝试下载")
add_arg("noise",  type=bool,  default=False, help="输入模型的是噪声")
add_arg("filter_dataset",      type=bool,  default=False, help="是否过滤数据集")
add_arg("random_choice",  type=bool,  default=False, help="随机选择标签中的文本,选用这个，等于模型无效，noise无效")
add_arg("task",       type=str, default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
add_arg("random_initialize_whisper", type=bool, default=False,    help="随机初始化whisper")
add_arg("teacher_forcing", type=bool, default=False,    help="使用teacher forcing")
add_arg("extra_name", type=str, default=None,    help="result basename里面增加字符")
add_arg("post_processing", type=bool, default=False,    help="是否使用后处理")
add_arg("config_name", type=str, default='base',    help="使用的模型")
add_arg("add_sequence_bias", type=bool, default=False,    help="是否对生成词增强。")
add_arg("base_model",    type=str, default="/mnt/petrelfs/zhangchi1/.cache/huggingface/hub/models--openai--whisper-base/snapshots/e37978b90ca9030d5170a5c07aadb050351a65bb", help="Whisper的基础模型")
add_arg("device", type=str, default='cuda',    help="device")
add_arg("mmd_input_type",    type=str, default='mean',      help="mmd")

# add_arg("metric",     type=str, default="fulleval",        choices=['cer', 'wer','fulleval'],              help="评估方式")
args = parser.parse_args()


def pearson_torch(y_true, y_pred, axis=1):
    """Pearson correlation function implemented in PyTorch.

    Parameters
    ----------
    y_true: torch.Tensor
        Ground truth labels. Shape is (batch_size, time_steps, n_features)
    y_pred: torch.Tensor
        Predicted labels. Shape is (batch_size, time_steps, n_features)
    axis: int
        Axis along which to compute the pearson correlation. Default is 1.

    Returns
    -------
    torch.Tensor
        Pearson correlation.
        Shape is (batch_size, 1, n_features) if axis is 1.
    """
    # Compute the mean of the true and predicted values
    y_true_mean = torch.mean(y_true, dim=axis, keepdim=True)
    y_pred_mean = torch.mean(y_pred, dim=axis, keepdim=True)

    # Compute the numerator and denominator of the pearson correlation
    numerator = torch.sum(
        (y_true - y_true_mean) * (y_pred - y_pred_mean),
        dim=axis,
        keepdim=True,
    )
    std_true = torch.sum(torch.square(y_true - y_true_mean), dim=axis, keepdim=True)
    std_pred = torch.sum(torch.square(y_pred - y_pred_mean), dim=axis, keepdim=True)
    denominator = torch.sqrt(std_true * std_pred)

    # Compute the pearson correlation
    return torch.mean(torch.where(denominator != 0, numerator / denominator, torch.zeros_like(numerator)), dim=-1)
    
    
    
def pearson_loss(y_true, y_pred, axis=1):
    return -pearson_correlation(y_true, y_pred, axis=axis)
    

def concatenate_images(pred_mel, gt_mel):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    sns.heatmap(pred_mel, cmap='viridis', ax=axs[0])
    axs[0].set_title('Predicted Mel-Spectrogram')

    sns.heatmap(gt_mel, cmap='viridis', ax=axs[1])
    axs[1].set_title('Ground Truth Mel-Spectrogram')

    # Remove axis for a cleaner look
    for ax in axs:
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')

    # Convert to PIL image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return Image.fromarray(img)



print_arguments(args)

# model path checking
assert 'openai' == os.path.dirname(args.model_path) or os.path.exists(args.model_path), \
    f"The model file {args.model_path} does not exist. Please check whether the model has been successfully merged, or if it is a model available on Hugging Face."
# Get Whisper's data processor, which includes feature extractor and tokenizer
print('loading')

os.environ['WORLD_SIZE'] = '1'
device_map = args.device
if device_map != 'cpu':
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if world_size != 1:
        device_index = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"{device_map}:{device_index}")
    else:
        device = torch.device(f"{device_map}:0")
else:
    device = torch.device("cpu")

''' base model load '''
pretrained = WhisperForConditionalGeneration.from_pretrained("/mnt/petrelfs/zhangchi1/.cache/huggingface/hub/models--openai--whisper-base/snapshots/e37978b90ca9030d5170a5c07aadb050351a65bb")
whisper_config = args.base_model + '/config.json'
whisper_config = AutoConfig.from_pretrained(whisper_config)
checkpoint_path =args.checkpoint_path
state_dict = torch.load(checkpoint_path+'full_model.pth')
depth=5
# model = BrainWhisperForConditionalGeneration2(whisper_config, state_dict.config.total_loss, pretrained, state_dict.config.run_name, depth=depth)
model = BrainWhisperForConditionalGeneration2(whisper_config, None, pretrained, None, depth=depth)
model.config.mmd_input_type=args.mmd_input_type
#model = get_peft_model(model, state_dict.peft_config['default'])

''' merge lora '''
contains_lora = any('lora' in key for key in state_dict.state_dict())
if contains_lora:
    print("adaLora was used")
    model = PeftModel.from_pretrained(model, checkpoint_path, local_files_only=args.local_files_only)
    model = model.merge_and_unload()

''' brain model load '''
brain_module_state_dict = {name.replace('base_model.model.', ''): param for name, param in state_dict.state_dict().items() if 'brain_module' in name}
model.load_state_dict(brain_module_state_dict, strict=False)
print(model)

device = torch.device(device_map)
model.to(device)
if args.noise==True:
    print(args.noise)
    results_path = '/mnt/petrelfs/zhangchi1/m2t/results/noise/'+state_dict.config.run_name
    mel_path = '/mnt/petrelfs/zhangchi1/m2t/predmel_output/noise/'+state_dict.config.run_name

else:
    results_path = '/mnt/petrelfs/zhangchi1/m2t/results/'
    mel_path = '/mnt/petrelfs/zhangchi1/m2t/predmel_output/'

if not os.path.exists(results_path):
    os.makedirs(results_path)

if not os.path.exists(mel_path):
    os.makedirs(mel_path)
processor = WhisperProcessor.from_pretrained(args.base_model,
                                             language=args.language,
                                             task=args.task,
                                             no_timestamps=not args.timestamps,
                                             local_files_only=args.local_files_only)
print('loading done')
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language=args.language,
    task=args.task,
    no_timestamps=not args.timestamps,)

# Get model
#model = BrainWhisperForConditionalGeneration2.from_pretrained(args.model_path,
#                                                    device_map="auto",
#                                                    local_files_only=args.local_files_only,)

model.eval()

# Get test data

test_dataset = BetterDataset(
    data_list_path=args.test_data,
    processor=processor,
    modal=args.modal,
    modal_ch=args.eeg_ch,
    sample_rate=args.sampling_rate,
    language=args.language,
    timestamps=args.timestamps,
    min_duration=args.min_audio_len,
    max_duration=args.max_audio_len)
print(f"test set size：{len(test_dataset)}")

# Data padding
data_collator = DataCollatorBrainSpeechSeq2SeqWithPadding(processor=processor)

eval_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, collate_fn=data_collator)

mcd_toolbox = Calculate_MCD(MCD_mode="plain")
mcd_value = 0
total_num = 0
useful_length = 1000

for step, batch in tqdm.tqdm(enumerate(eval_dataloader)):
    batch['subject_index'] = batch['subject_index'].to(device)
    with torch.cuda.amp.autocast():
         with torch.no_grad():
            if not args.random_choice:
                input_features = batch["input_features"].to(device)

                pred_mel, _ = model.predict_mel(input_features=input_features, useful_length=1200, subject_index=batch['subject_index'])
                pred_mel = pred_mel[..., :useful_length].type(torch.float32).cpu()
                gt_mel = batch['mel_spec'][..., :useful_length].type(torch.float32).cpu()
                gt_mel = 10 ** (gt_mel * 4.0 - 4.0).numpy()
                pred_mel = 10 ** (pred_mel * 4.0 - 4.0).numpy()
                mcd_value += torch.sum(torch.mean(pearson_torch(torch.tensor(gt_mel), torch.tensor(pred_mel), axis=-1), axis=-1))
                # import pdb;pdb.set_trace()
                # for i in range(pred_mel.shape[0]):
                #     pred = librosa.feature.inverse.mel_to_audio(pred_mel[i], sr=16000, n_fft=400, hop_length=160, win_length=400, fmin=0, fmax=8000, pad_mode='reflect')
                #     gt = librosa.feature.inverse.mel_to_audio(gt_mel[i], sr=16000, n_fft=400, hop_length=160, win_length=400, fmin=0, fmax=8000, pad_mode='reflect')
                #     soundfile.write('transformer_pred_test_'+ str(i) + '.wav', pred, 16000) 
                #     soundfile.write('300_gt_test_'+ str(i) + '.wav', gt, 16000)
                    
                    # combined_img = concatenate_images(pred_mel[i].numpy(), gt_mel[i].numpy())
                    # combined_img.save('pred_'+ str(i) +'.jpg')
                    # import pdb;pdb.set_trace()

                    # mcd_value += mcd_toolbox.calculate_mcd([gt], [pred])
                total_num += pred_mel.shape[0]
                if total_num % 100 == 0:
                    print(mcd_value / total_num)
                    # print(mcd_value)

print("final:")
print(mcd_value / total_num)
