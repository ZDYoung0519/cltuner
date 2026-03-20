# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import numpy as np
import os.path as osp
from mmengine.config import Config
from xtuner.configs import cfgs_name_path
import subprocess
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Eval model")
    parser.add_argument("config", help="config file name or path.")
    parser.add_argument("--cur-task", type=int, default=-1, help="current task index")
    parser.add_argument("--work-dir", help="the directory to save the file containing evaluation metrics")
    args = parser.parse_args()
    return args


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
    if args.cur_task == -1:
        args.cur_task = len(cfg.test_dataset) - 1

    result_matrix = np.ones((args.cur_task + 1, args.cur_task + 1)) * np.nan
    for task_id in range(args.cur_task + 1):
        for eval_task_id in range(task_id + 1):
            metainfo = cfg.test_dataset[eval_task_id].get('metainfo', {})

            task_name = metainfo.get('name', f'task{eval_task_id}')
            eval_script = metainfo.get('eval_script', None)
            anno_file = metainfo.get('anno_file', None)
            result_file = osp.join(args.work_dir, f"eval/task{task_id}/task{eval_task_id}/merge.jsonl")
            output_dir = osp.join(args.work_dir, f"eval/task{task_id}/task{eval_task_id}")
            print(f"####### After Train Task {task_id} | Evaluating Task {eval_task_id}: {task_name} ########")
            cmd = [
                "python", eval_script,
                '--result-file', result_file,
                '--output-dir', output_dir,
                '--annotation-file', anno_file
            ]
            print(f"Running eval script: {' '.join(cmd)}")
            subprocess.run(cmd)
            try:
                with open(osp.join(output_dir, 'result.txt'), 'r') as f:
                    lines = f.readlines()[-1]
                    score = float(lines.strip().split(':')[-1].replace('%', ''))
                    result_matrix[task_id, eval_task_id] = score
            except:
                print(f"WARNING: Failed to read result for Task {task_id} on Eval Task {eval_task_id}")
                result_matrix[task_id, eval_task_id] = np.nan

    dataset_names = [cfg.test_dataset[i].get('metainfo', {}).get('name', f'task{eval_task_id}') for i in range(args.cur_task + 1)]
    df = pd.DataFrame(result_matrix)
    df.columns = dataset_names
    df.index = dataset_names
    df.to_csv(osp.join(args.work_dir, f'result_matrix.csv'))
    print("Final Evaluation Matrix:")
    print(df)


if __name__ == "__main__":
    main()

