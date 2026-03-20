# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import json
import math
import os.path as osp
import tqdm
from types import FunctionType

import torch
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.dist import (
    collect_results,
    get_dist_info,
    get_rank,
    init_dist,
    master_only,
)
from transformers import (
    GenerationConfig,
)

from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint, prepare_inputs_labels_for_multimodal
from xtuner.registry import MAP_FUNC
from xtuner.utils.device import get_device, get_torch_device
from xtuner.tools.utils import get_stop_criteria, is_cn_string
from xtuner.registry import BUILDER


def parse_args():
    parser = argparse.ArgumentParser(description="Eval model")
    parser.add_argument("config", help="config file name or path.")
    parser.add_argument("--checkpoint", default=None, help="checkpoint file")
    parser.add_argument("--eval-task", type=int, default=-1, help="current task index")
    parser.add_argument("--output-dir", type=str, help="checkpoint file")
    parser.add_argument("--num-chunks", default=1, type=int, help="checkpoint file")
    parser.add_argument("--chunk-idx", default=0, type=int, help="checkpoint file")
    parser.add_argument('--max-new-tokens', default=200, type=int)
    parser.add_argument('--temperature', default=0.2, type=float)
    parser.add_argument(
        "--work-dir",
        help="the directory to save the file containing evaluation metrics",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    return args


def register_function(cfg_dict):
    if isinstance(cfg_dict, dict):
        for key, value in dict.items(cfg_dict):
            if isinstance(value, FunctionType):
                value_str = str(value)
                if value_str not in MAP_FUNC:
                    MAP_FUNC.register_module(module=value, name=value_str)
                cfg_dict[key] = value_str
            else:
                register_function(value)
    elif isinstance(cfg_dict, (list, tuple)):
        for value in cfg_dict:
            register_function(value)


def generate_answers(model, dataset, args):
    device = get_device()

    # rank, world_size = get_dist_info()
    model = model.module if hasattr(model, "module") else model
    model.to(device)
    tokenizer = dataset.tokenizer

    # generate config
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        use_cache=True,
    )
    stop_criteria = get_stop_criteria(tokenizer=tokenizer, stop_words='')

    rank = args.chunk_idx
    world_size = args.num_chunks
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size)
    per_rank_ids = range(
        per_rank_samples * rank, min(n_samples, per_rank_samples * (rank + 1))
    )

    print(f"[CHUNK:{rank}][IDX:{str(per_rank_ids)}][LEN:{len(per_rank_ids)}][TOTAL:{n_samples}]")
    results = []
    for i in tqdm.tqdm(per_rank_ids, desc=f"Rank {rank}"):
        data = dataset[i]
        data["input_ids"] = data["input_ids"].to(device).unsqueeze(0)
        data["pixel_values"] = data["pixel_values"].to(device).unsqueeze(0)

        with torch.inference_mode():
            data = model._prepare(data)
            mm_inputs = prepare_inputs_labels_for_multimodal(
                llm=model.llm,
                input_ids=data["input_ids"],
                pixel_values=data["pixel_values"],
            )
            generation_output = model.generate(
                **mm_inputs,
                generation_config=gen_config,
                bos_token_id=tokenizer.bos_token_id,
                stopping_criteria=stop_criteria,
            )

        prediction = tokenizer.decode(generation_output[0], skip_special_tokens=True).strip()
        cur_result = {}
        cur_result["question_id"] = data["question_id"]
        cur_result["text"] = data.get("text")
        cur_result["image"] = data.get("image", None)
        cur_result["answer"] = data.get("answer")
        cur_result["prediction"] = prediction
        results.append(cur_result)
    # results = collect_results(results, n_samples)
    return results


def main():
    args = parse_args()

    # parse config
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f"Cannot find {args.config}")

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register FunctionType object in cfg to `MAP_FUNC` Registry and
    # change these FunctionType object to str
    register_function(cfg._cfg_dict)

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )

    # disable visualizer
    cfg.visualizer = None

    # build model and dataset
    model = BUILDER.build(cfg.model)
    dataset = BUILDER.build(cfg.test_dataset[args.eval_task])

    # # load checkpoint (we don't need this as we specify `model.pretrained_pth=xxx` in the config file)
    # if args.checkpoint is not None:
    #     state_dict = guess_load_checkpoint(args.checkpoint)
    #     model.load_state_dict(state_dict, strict=False)
    #     print_log(f"Load checkpoint from {args.checkpoint} successfully!", 'current')

    data_name = dataset.metainfo['name']
    print_log("##########################################")
    print_log(f"############ {data_name} ############")
    print_log("##########################################")

    # generate answers 
    results = generate_answers(model, dataset, args)

    # save results
    if not os.path.exists(args.output_dir) and get_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"chunk{args.chunk_idx}.jsonl")
    if get_rank() == 0:
        file = open(output_file, "w", encoding='utf-8')
        for res in results:
            file.write(json.dumps(res, ensure_ascii=False) + "\n")
        file.close()

if __name__ == "__main__":
    main()

