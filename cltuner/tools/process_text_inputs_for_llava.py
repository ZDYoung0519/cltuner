# Copyright (c) OpenMMLab. All rights reserved。
import argparse
import os
import json
from mmengine.config import Config
from mmengine.dist import get_rank, get_world_size, init_dist, all_gather
from mmengine.runner.runner import Runner
from mmengine.model import MMDistributedDataParallel
from mmengine.device import get_device

from xtuner.dataset.huggingface import process_hf_dataset
from xtuner.dataset.utils import expand2square, DEFAULT_IMAGE_TOKEN
from xtuner.registry import BUILDER
from datasets import DatasetDict
from datasets import Dataset as HFDataset
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


def load_jsonl(json_file):
    with open(json_file) as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data


def warp_model(model, distributed=False):
    model = model.to(get_device())
    if not distributed:
        return model
    model = MMDistributedDataParallel(
        module=model,
        device_ids=[int(os.environ["LOCAL_RANK"])],
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    return model


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Extract vision hidden states")
    parser.add_argument("config", help="config file name or path.")
    parser.add_argument("--save-folder", default="", type=str, help="path to save data")

    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    for task_id in range(len(cfg.train_dataset)):
        dataset = cfg.train_dataset[task_id]
        
        data_path = dataset['data_path']
        tokenizer = dataset['tokenizer']
        max_length = dataset['max_length']
        dataset_map_fn = dataset['dataset_map_fn']
        template_map_fn = dataset['template_map_fn']
        max_dataset_length = dataset.get('max_dataset_length', None)
        task_name = os.path.basename(os.path.dirname(data_path))

        if data_path.endswith(".json"):
            json_data = json.load(open(data_path))
        elif data_path.endswith(".jsonl"):
            json_data = load_jsonl(data_path)
        else:
            raise NotImplementedError

        for idx in range(len(json_data)):
            if not hasattr(json_data[idx], "id"):
                json_data[idx]["id"] = str(idx)
            if isinstance(json_data[idx]["id"], int):
                json_data[idx]["id"] = str(json_data[idx]["id"])
        json_data = DatasetDict({"train": HFDataset.from_list(json_data)})

        text_data = process_hf_dataset(
                dataset=json_data,
                tokenizer=tokenizer,
                max_length=max_length,
                dataset_map_fn=dataset_map_fn,
                template_map_fn=template_map_fn,
                split="train",
                max_dataset_length=max_dataset_length,
                remove_unused_columns=False,
                pack_to_max_length=False,
                with_image_token=True,
                map_num_proc=4
            )
        data_folder = os.path.dirname(data_path)
        save_folder = os.path.join(args.save_folder, task_name)
        os.makedirs(save_folder, exist_ok=True)

        text_data.save_to_disk(save_folder)
        print('Save at: {}'.format(save_folder))


if __name__ == "__main__":
    main()