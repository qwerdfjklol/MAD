import jsonlines
import os
import sys
import io
from petrel_client.client import Client
# This is to filter a certain story in the GWilliams data set
# Get the file path of the current script
# current_path = os.path.abspath(__file__)
# Get the path to the project root directory
# project_root = os.path.dirname(os.path.dirname(current_path))
# Add the project root directory to sys.path
# sys.path.append(project_root)
# import argparse
# import functools
# from utils.utils import add_arguments


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



# python process_dataset/filter_story_jsonl.py
if __name__ == '__main__':
    client = Client()
    home_dir = os.path.expanduser("~")
    val_a_story=True
    # parser = argparse.ArgumentParser(description=__doc__)
    # add_arg = functools.partial(add_arguments, argparser=parser)
    # add_arg("jsonl",    type=str, default=None,       help="jsonl文件路径")
    # add_arg("output_dir",    type=str,  default=None,       help="输出jsonl文件夹")
    # args = parser.parse_args()
    input_jsonl='s3://MAD/Gwilliams2023/preprocess_10_nofilter/info.jsonl'
    output_dir='s3://MAD/Gwilliams2023/preprocess_10_nofilter/split3'
    # import pdb;pdb.set_trace()
    datas = read_jsonlines_from_petrel(input_jsonl)
    story_list=['lw1', 'cable_spool_fort', 'easy_money', 'the_black_willow']
    # 分割四种
    for hold_out_story_id, hold_out_story in enumerate(story_list):
        if val_a_story:
            for val_story in [s for s in story_list if s not in [hold_out_story]]:
                train_list=[data for data in datas if data['story_id'] not in [float(story_list.index(val_story)),float(hold_out_story_id)]]
                val_list=[data for data in datas if data['story_id']==float(story_list.index(val_story))]
                test_list=[data for data in datas if data['story_id']==float(hold_out_story_id)]

                data_dict = {
                    'train': train_list,
                    'val': val_list,
                    'test': test_list,
                }
                print(val_story,hold_out_story,len(train_list),len(val_list),len(test_list))
                for k,v in data_dict.items():
                    json=os.path.join(output_dir,hold_out_story,val_story,f'{k}.jsonl')
                    write_jsonlines_with_petrel(json, v)


        else:
            train_val_list=[data for data in datas if data['story']!=hold_out_story]
            test_list=[data for data in datas if data['story']==hold_out_story]
            split_num=int(len(train_val_list)/9*8)
            train_list=train_val_list[:split_num]
            val_list=train_val_list[split_num:]
            print(len(train_list),len(val_list),len(test_list))
            data_dict={
                'train':train_list,
                'val':val_list,
                'test':test_list,
            }
            for mode in ['train','val','test']:
                json=os.path.join(home_dir,output_dir,hold_out_story,f'{mode}.jsonl')
                write_jsonlines(makedirs(json), data_dict[mode])

