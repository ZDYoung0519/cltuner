import argparse
import logging
import os
import os.path as osp

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
from timm.utils.metrics import accuracy

import math
import numpy as np
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.dist import all_gather, is_main_process, get_rank, get_world_size, broadcast_object_list
from mmengine.device import get_device
from xtuner.registry import BUILDER

from sklearn.cluster import KMeans
from tqdm import tqdm

from cltuner.utils.metric import MetricLogger, SmoothedValue


@torch.no_grad()
def collect_features(loader, model):
    rank = get_rank()
    all_features = []
    pbar = tqdm(loader, desc=f"Rank {rank}: collect features", position=rank, leave=False)
    for data in pbar:
        out = model(**data, mode="")
        all_features.append(out['embeddings'].detach().cpu())
    features = torch.cat(all_features, dim=0)
    print("Rank{}: Features shape: {}".format(rank, features.shape))
    all_features = torch.cat(all_gather(features), dim=0)
    print("All features shape: {} {}".format(rank, all_features.shape))
    return all_features


@torch.no_grad()
def collect_test_features(dataset, model):
    rank = get_rank()
    world_size = get_world_size()
    device = get_device()

    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size)
    per_rank_ids = range(
        per_rank_samples * rank, min(n_samples, per_rank_samples * (rank + 1))
    )
    
    pbar = tqdm(per_rank_ids, desc=f"Rank {rank}")
    all_features = []
    for i in pbar:
        data = dataset[i]
        data["input_ids"] = data["input_ids"].to(device).unsqueeze(0)
        data["pixel_values"] = data["pixel_values"].to(device).unsqueeze(0)
        data['text'] = [data['text']]
        out = model(data, None, "")['embeddings'].detach().cpu()
        all_features.append(out)
    all_features = torch.cat(all_features, dim=0)

    print("Rank{}: Features shape: {}".format(rank, all_features.shape))
    all_features = torch.cat(all_gather(all_features), dim=0)
    print("All features shape: {} {}".format(rank, all_features.shape))
    return all_features


def cal_mean_cov_with_kmeans(features, n_clusters, mode='covariance'):
    device = features.device
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(features)

    # cal mean and cov for all clusters
    cluster_lables = kmeans.labels_
    cluster_means = []
    cluster_vars = []
    for i in range(n_clusters):
        cluster_data = features[cluster_lables == i].detach()
        cluster_mean = cluster_data.mean(0).to(device)
        if mode == "var":
            cluster_var = torch.tensor(np.var(cluster_data, axis=0), dtype=torch.float64).to(device)
        elif mode == "cov":
            cluster_var = torch.cov(torch.tensor(cluster_data, dtype=torch.float64).T) + torch.eye(
                cluster_mean.shape[-1]) * 1e-4
            cluster_var = cluster_var.to(device)
        else:
            raise NotImplementedError
        cluster_means.append(cluster_mean)
        cluster_vars.append(cluster_var)
    return cluster_means, cluster_vars


class GaussianMemory(object):
    def __init__(self, num_clusters_per_task, cov_mode, l2_norm=False):
        self.num_clusters_per_task = num_clusters_per_task
        self.cov_mode = cov_mode
        self.l2_norm = l2_norm
        self.means = {}
        self.covs = {}
    
    def update(self, all_features, task_id):
        if self.l2_norm:
            all_features = F.normalize(all_features.detach(), p=2.0, dim=1)
        else:
            all_features = all_features.detach()

        if is_main_process():
            cluster_means, cluster_covs = cal_mean_cov_with_kmeans(
                all_features,
                n_clusters=self.num_clusters_per_task, 
                mode=self.cov_mode
            )
        else:
            cluster_means = None
            cluster_covs = None

        # 广播对象到所有进程
        objects = [cluster_means, cluster_covs] if is_main_process() else [None, None]
        broadcast_object_list(objects, src=0)
        cluster_means, cluster_covs = objects[0], objects[1]

        # 存储（转换为numpy）
        self.means[task_id] = cluster_means
        self.covs[task_id] = cluster_covs
    
    def get_seen_tasks(self):
        return  list(self.means.keys())

    def sample_data(self, num_per_clusters):
        seen_tasks = self.get_seen_tasks()
        num_tasks = len(seen_tasks)
        sampled_data = []
        sampled_label = []
        for task_id in range(num_tasks):
            task_means = self.means[task_id]
            task_covs = self.covs[task_id]
            n_clusters = len(task_means)
            for cluster_id in range(n_clusters):
                if self.cov_mode == 'var':
                    mean = task_means[cluster_id]
                    var = task_covs[cluster_id]
                    cov = (torch.diag(var) + 1e-4 * torch.eye(mean.shape[0]).to(mean.device))
                    m = MultivariateNormal(mean.float(), cov.float())
                    sampled_data_single = m.sample(sample_shape=(num_per_clusters,))
                elif self.cov_mode == 'cov':
                    mean = task_means[cluster_id]
                    cov = task_covs[cluster_id]
                    m = MultivariateNormal(mean.float(), cov.float())
                    sampled_data_single = m.sample(sample_shape=(num_per_clusters,))
                else:
                    raise NotImplementedError
                sampled_data.append(sampled_data_single.detach().cpu())
                sampled_label.extend([task_id] * num_per_clusters)
        sampled_data = torch.cat(sampled_data, dim=0).float()
        sampled_label = torch.tensor(sampled_label).long()
        return sampled_data, sampled_label
    
    def state_dict(self):
        return {'means': self.means, "covs": self.covs}
    
    def load_state_dict(self, state_dict):
        assert 'means' in state_dict
        assert 'covs' in state_dict
        self.means = state_dict['means']
        self.covs = state_dict['covs']

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--cur-task', type=int, default=0, required=True)
    parser.add_argument('--num_clusters_per_task', type=int, default=5, help='the dir to save logs and models')
    parser.add_argument('--cov_mode', type=str, default='cov', help='the dir to save logs and models')
    parser.add_argument("--l2-norm", action='store_true')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')


    # customize
    cfg.model.cur_task = args.cur_task

    
    # set work dir and exp name
    cfg_name = osp.basename(args.config).split(".")[0]
    cfg.visualizer.vis_backends[0].init_kwargs.experiment_name =\
        f'/{cfg_name}/task{args.cur_task}'
    work_dir_parent = cfg.work_dir
    cfg.work_dir = osp.join(work_dir_parent, f'task{args.cur_task}')

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
    
    model = runner.model.module if hasattr(runner.model, 'module') else runner.model
    from projects.tadyra.model.task_adaptive_visual_encoder import TaskAdaptiveVisualEncoder
    assert isinstance(model, TaskAdaptiveVisualEncoder)

    # build optimizer and scheaduler
    runner.optim_wrapper = runner.build_optim_wrapper(runner.optim_wrapper)
    # runner.param_schedulers = runner.build_param_scheduler(runner.param_schedulers)
    optimizer = runner.optim_wrapper.optimizer

    # collect all train features
    all_train_datasets_cfg = cfg.train_dataset
    train_dataloader_cfg = cfg.train_dataloader
    train_features_path = osp.join('./work_dirs', f'train_features.pth')
    os.makedirs(train_features_path, exist_ok=True)
    for task_id in range(len(cfg.train_dataset)):
        task_train_features_path = osp.join(train_features_path, f"task{task_id}.pt")
        if not osp.exists(task_train_features_path):
            train_dataloader_cfg['dataset'] = all_train_datasets_cfg[task_id]
            diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
            train_dataloader = runner.build_dataloader(
                train_dataloader_cfg, seed=runner.seed, diff_rank_seed=diff_rank_seed)

            task_features = collect_features(train_dataloader, model).detach().cpu()
            if is_main_process():
                torch.save(task_features, task_train_features_path)

    # collect all test features
    test_features_path = osp.join('./work_dirs', f'train_features.pth')
    os.makedirs(train_features_path, exist_ok=True)
    for task_id in range(len(cfg.test_dataset)):
        task_test_features_path = osp.join(test_features_path, f"task{task_id}.pt")
        if not osp.exists(task_test_features_path):
            testset = BUILDER.build(cfg.test_dataset[task_id])
            test_features = collect_test_features(testset, model)
            if is_main_process():
                torch.save(test_features, task_test_features_path)

    gaussian_memory = GaussianMemory(args.num_clusters_per_task, args.cov_mode)
    # load previous gaussian_statics
    if args.cur_task > 0:
        pre_ckpt_path = osp.join(work_dir_parent, f'task{args.cur_task-1}/gaussian_memeory_state_dict.pth')
        if os.path.exists(pre_ckpt_path):
            gaussian_memory.load_state_dict(
                torch.load(pre_ckpt_path)
            )
            print_log(f"Load Gaussian Memory from: {pre_ckpt_path}")
    
    # update current gaussian_statics
    task_train_features_path = osp.join(train_features_path, f"task{args.cur_task}.pt")
    task_features = torch.load(task_train_features_path)
    gaussian_memory.update(task_features,task_id=args.cur_task)

    if is_main_process():
        os.makedirs(cfg.work_dir, exist_ok=True)
        torch.save(gaussian_memory.state_dict(), osp.join(cfg.work_dir, f'gaussian_memeory_state_dict.pth'))


    batch_size = cfg.batch_size

    def train_epoch():
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', SmoothedValue(window_size=1, fmt='{value:.4f}'))

        device = get_device()
        sampled_data, sampled_label = gaussian_memory.sample_data(batch_size)
        inputs = sampled_data
        targets = sampled_label
        sf_indexes = torch.randperm(inputs.size(0))
        inputs = inputs[sf_indexes]
        targets = targets[sf_indexes]
        crct_num = args.cur_task * args.num_clusters_per_task
        for _iter in range(crct_num):
            inp = inputs[_iter * batch_size:(_iter + 1) * batch_size].to(device)
            tgt = targets[_iter * batch_size:(_iter + 1) * batch_size].to(device)
            logits = model.dynamic_routing(inp)
            logits = logits[:, :args.cur_task+1]
            loss = F.cross_entropy(logits, tgt)
            with torch.no_grad():
                acc1, acc5 = accuracy(logits, tgt, topk=(1, 5))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_logger.update(Loss=loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=inp.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=inp.shape[0])
        
        metric_logger.synchronize_between_processes()
        print('Epoch: {}, Train Loss: {:.3f}\tAcc@1: {:.2f}\tAcc@5: {:.2f}'.format(
            epoch+1,
            metric_logger.meters['Loss'].global_avg,
            metric_logger.meters['Acc@1'].global_avg,
            metric_logger.meters['Acc@5'].global_avg)
        )
        torch.cuda.synchronize()
        return model
    
    @torch.no_grad()
    def validate_until_now(model, cur_task):
        val_acc1, val_acc5 = 0, 0
        dynamic_routing = model.dynamic_routing
        device = get_device()
        for task_id in range(cur_task+1):
            task_test_features_path = osp.join(test_features_path, f"task{task_id}.pt")
            task_features = torch.load(task_test_features_path).to(device)
            logits = dynamic_routing(task_features)
            logits[:, (cur_task+1):] = float('-inf')

            targets = torch.LongTensor([task_id] * logits.shape[0]).to(device)
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))

            print(f"Valid on task {task_id}, acc1: {acc1}, acc5: {acc5}")
            val_acc1 += acc1
            val_acc5 += acc5
        val_acc1 /= cur_task+1
        val_acc5 /= cur_task+1

        return val_acc1, val_acc5

    # retrain the task head
    if args.cur_task > 0:
        # load previous ckpt
        ckpt = osp.join(work_dir_parent, f'task{args.cur_task-1}/dynamic_routing_best.pth')
        if osp.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=False)

        best_acc = 0.
        for epoch in range(cfg.max_epochs):
            train_epoch()
            if is_main_process():
                val_acc1, _ = validate_until_now(model, args.cur_task)
                print(f"Valid until task {args.cur_task}, acc1: {val_acc1}, best: {best_acc}")
                if val_acc1 > best_acc:
                    best_acc = val_acc1
                    state_dict = model.state_dict()
                    to_return = {k: state_dict[k] for k in state_dict if "dynamic_routing" in k}
                    torch.save(to_return, osp.join(work_dir_parent, f'task{args.cur_task}/dynamic_routing_best.pth'))


        if is_main_process():
            state_dict = model.state_dict()
            to_return = {k: state_dict[k] for k in state_dict if "dynamic_routing" in k}
            torch.save(to_return, osp.join(work_dir_parent, f'task{args.cur_task}/dynamic_routing_last.pth'))


if __name__ == '__main__':
    main()

