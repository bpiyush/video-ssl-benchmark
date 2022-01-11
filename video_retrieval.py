"""
Feature-based retrieval for a video.
"""

from os.path import basename
import argparse
import time
from torch._C import EnumType
import yaml
import torch
import wandb

import utils.logger
from utils import main_utils, eval_utils
import torch.multiprocessing as mp

from utils.debug import debug

import warnings
warnings.filterwarnings("ignore")

from aot import get_model, distribute_model_to_cuda


parser = argparse.ArgumentParser(description='Video retrieval')
parser.add_argument('cfg', metavar='CFG', help='config file')
parser.add_argument('model_cfg', metavar='CFG', help='config file')
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--test-only', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--port', default='1234')
parser.add_argument('--pretext-model', default='rspnet')
parser.add_argument('--no_wandb', action='store_true')
parser.add_argument('--freeze_backbone', action='store_true')
parser.add_argument('-w', '--wandb_run_name', default="base", type=str, help='name of run on W&B')


def main():
    ngpus = torch.cuda.device_count()
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))

    if not args.no_wandb:
        name = basename(args.cfg).split(".yaml")[0] + "-" + args.wandb_run_name
        wandb.init(name=name, project="video-ssl", entity="uva-vislab", config=cfg)
        wandb.config.update(args)
        wandb.config.update(cfg)

    if args.test_only:
        cfg['test_only'] = True
    if args.resume:
        cfg['resume'] = True
    if args.debug:
        cfg['num_workers'] = 1
        cfg['dataset']['batch_size'] = 4

    if args.distributed:
        mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, cfg['dataset']['fold'], args, cfg))
    else:
        main_worker(None, ngpus, cfg['dataset']['fold'], args, cfg)


def model_features_for_given_dataset(model, dataloader):
    """Computes model features for a given dataset."""
    from tqdm import tqdm

    results = {
        "features": [],
        "labels": [],
    }

    iterator = tqdm(
        dataloader,
        desc=f"Computing features for {type(model).__name__} on {dataloader.dataset.name}",
        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
    )
    with torch.no_grad():
        for batch in iterator:
            # import ipdb; ipdb.set_trace()
            frames = batch["frames"]
            labels = batch["label"]
            # frames, labels = batch

            # remove time dimension since number of frames 1
            frames = frames.squeeze(2)

            # forward pass
            features = model(frames)

            results["features"].append(features.cpu())
            results["labels"].append(labels.cpu())

    results["features"] = torch.cat(results["features"], dim=0)
    results["labels"] = torch.cat(results["labels"], dim=0)

    return results


def main_worker(gpu, ngpus, fold, args, cfg):
    args.gpu = gpu
    args.world_size = ngpus

    # Prepare folder and logger
    eval_dir, model_cfg, logger = eval_utils.prepare_environment(args, cfg, fold)

    # create pretext model
    model, ckp_manager = get_model(model_cfg, cfg, eval_dir, args, logger) 
    
    # freeze the backbone
    if args.freeze_backbone:
        model = eval_utils.freeze_backbone(model, args.pretext_model)
    
    # replace fully connected layer with a identity layer since we only want to extract features
    model.fc = torch.nn.Identity()
    model = model.eval()

    # Optimizer
    optimizer, scheduler = main_utils.build_optimizer(model.parameters(), cfg['optimizer'], logger)

    # Datasets
    train_loader, test_loader, dense_loader = eval_utils.build_dataloaders(
        cfg['dataset'], fold, cfg['num_workers'], args.distributed, logger)
    # train_loader, test_loader = build_dataloaders_debug(cfg)

    # Distribute
    model = distribute_model_to_cuda(model, args, cfg)

    # get features
    train_results = model_features_for_given_dataset(model, train_loader)
    debug()


if __name__ == '__main__':
    main()

