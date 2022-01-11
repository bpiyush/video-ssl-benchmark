"""
Feature-based retrieval for a video.
"""
from genericpath import exists
import os
from os.path import basename
from collections import defaultdict
import argparse
import time
from torch._C import EnumType
import yaml
import torch
import wandb
import numpy as np
from tqdm import tqdm

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
parser.add_argument('--ignore_cache', action='store_true')
parser.add_argument('-w', '--wandb_run_name', default="base", type=str, help='name of run on W&B')


def get_model_only_backbone(model_cfg, cfg, eval_dir, args, logger):

    if args.pretext_model== 'rspnet':
            model, ckp_manager =  eval_utils.build_model_rsp(
                model_cfg, cfg, eval_dir, args, logger,
            )
            model.fc = torch.nn.Identity()
    elif args.pretext_model== 'tclr':
            model, ckp_manager =  eval_utils.build_model_tclr(
                model_cfg, cfg, eval_dir, args, logger,
            )
            model.fc = torch.nn.Identity()
    elif args.pretext_model== 'avid_cma':
            model, ckp_manager =  eval_utils.build_model_avid_cma(
                model_cfg, cfg, eval_dir, args, logger,
            )
            model.classifier = torch.nn.Identity()
    elif args.pretext_model== 'pretext_contrast':
            model, ckp_manager =  eval_utils.build_model_pretext_contrast(
                model_cfg, cfg, eval_dir, args, logger,
            )
            model.linear = torch.nn.Identity()
    elif args.pretext_model== 'gdt':
            model, ckp_manager =  eval_utils.build_model_gdt(
                model_cfg, cfg, eval_dir, args, logger,
            )
            model.classifier = torch.nn.Identity()
    elif args.pretext_model== 'ctp':
            model, ckp_manager =  eval_utils.build_model_ctp(
                model_cfg, cfg, eval_dir, args, logger,
            )
            model.fc = torch.nn.Identity()
    elif args.pretext_model== 'video_moco':
            model, ckp_manager =  eval_utils.build_model_video_moco(
                model_cfg, cfg, eval_dir, args, logger,
            )
            model.fc = torch.nn.Identity()
    elif args.pretext_model== 'selavi':
            model, ckp_manager =  eval_utils.build_model_selavi(
                model_cfg, cfg, eval_dir, args, logger,
            )
            model.classifier = torch.nn.Identity()
    elif args.pretext_model== 'full_supervision':
            model, ckp_manager =  eval_utils.build_model_full_supervision(
                model_cfg, cfg, eval_dir, args, logger,
            )
            model.fc = torch.nn.Identity()

    return model, ckp_manager


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


def model_features_for_given_dataset(model, dataloader, mode="train", name="sample", use_cached=True):
    """Computes model features for a given dataset."""

    results_path = f"./cache/features/{name}_{mode}.pt"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    if use_cached and os.path.exists(results_path):
        return torch.load(results_path, map_location="cpu")

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
            frames = batch["frames"]
            labels = batch["label"]

            # forward pass
            features = model(frames)

            results["features"].append(features.cpu())
            results["labels"].append(labels.cpu())

    results["features"] = torch.cat(results["features"], dim=0)
    results["labels"] = torch.cat(results["labels"], dim=0)
    
    assert results["features"].shape == torch.Size([len(results["labels"]), 512])

    if not exists(results_path):
        torch.save(results, results_path)

    return results


def retrieval(
    train_features,
    train_labels,
    train_vid_indices,
    val_features,
    val_labels,
    val_vid_indices,
    train_aud_features=None,
    val_aud_features=None,
    task='v-v',
):
    """
    Computes retrieval scores for a given dataset.
    
    Function borrowed from Facebook's SeLaVi codebase.
    Link: https://github.com/facebookresearch/selavi/
    """
    from sklearn.neighbors import NearestNeighbors

    assert task in ['v-a', 'a-v', 'v-v', 'a-a']
    if task in ['v-a', 'a-v', 'a-a']:
        assert(train_aud_features is not None)
        assert(val_aud_features is not None)

    if task == 'v-v':
        feat_val = val_features
        feat_train = train_features
    elif task == 'v-a':
        feat_val = val_features
        feat_train = train_aud_features
    elif task == 'a-v':
        feat_val = val_aud_features
        feat_train = train_features
    elif task == 'a-a':
        feat_val = val_aud_features
        feat_train = train_aud_features

    # Create 
    neigh = NearestNeighbors(n_neighbors=50)
    neigh.fit(feat_train)
    recall_dict = defaultdict(list)
    retrieval_dict = {}

    iterator = tqdm(
        range(len(feat_val)),
        desc=f'Computing retrieval results per video',
        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
    )
    for i in iterator:
        feat = np.expand_dims(feat_val[i], 0)
        vid_idx = val_vid_indices[i]
        vid_label = val_labels[i]
        retrieval_dict[vid_idx] = {
            'label': vid_label,
            'recal_acc': {
                '1': 0, '5': 0, '10': 0, '20': 0, '50': 0
            },
            'neighbors': {
                '1': [], '5':[], '10': [], '20': [], '50': []
            }
        }
        for recall_treshold in [1, 5, 10, 20, 50]:
            neighbors = neigh.kneighbors(feat, recall_treshold)
            neighbor_indices = neighbors[1]
            neighbor_indices = neighbor_indices.flatten()
            neighbor_labels = set([train_labels[vid_index] for vid_index in neighbor_indices])
            recall_value = 100 if vid_label in neighbor_labels else 0
            acc_value = len([1 for neigh_label in neighbor_labels if neigh_label == vid_label]) / float(len(neighbor_labels))
            retrieval_dict[vid_idx]['recal_acc'][str(recall_treshold)] = acc_value
            retrieval_dict[vid_idx]['neighbors'][str(recall_treshold)] = neighbor_indices
            recall_dict[recall_treshold].append(recall_value)
        # print(f'Computing retrieval benchmarks for video: {i} / {len(feat_val)}', end='\r')

    # Calculate mean recall values
    for recall_treshold in [1, 5, 10, 20, 50]:
        mean_recall = np.mean(recall_dict[recall_treshold]) 
        print(f"{task}: Recall @ {recall_treshold}: {mean_recall}")
    return retrieval_dict


def main_worker(gpu, ngpus, fold, args, cfg, norm_feat=True):
    args.gpu = gpu
    args.world_size = ngpus

    # Prepare folder and logger
    eval_dir, model_cfg, logger = eval_utils.prepare_environment(args, cfg, fold)

    # create pretext model
    model, ckp_manager = get_model_only_backbone(model_cfg, cfg, eval_dir, args, logger) 
    
    # freeze the backbone
    if args.freeze_backbone:
        model = eval_utils.freeze_backbone(model, args.pretext_model)
    
    # eval mode
    model = model.eval()

    # Optimizer
    optimizer, scheduler = main_utils.build_optimizer(model.parameters(), cfg['optimizer'], logger)

    # Datasets
    train_loader, test_loader, dense_loader = eval_utils.build_dataloaders(
        cfg['dataset'], fold, cfg['num_workers'], args.distributed, logger)

    # Distribute
    model = distribute_model_to_cuda(model, args, cfg)

    # get features
    train_results = model_features_for_given_dataset(model, train_loader, mode="train", name=args.wandb_run_name, use_cached=(not args.ignore_cache))
    test_results = model_features_for_given_dataset(model, test_loader, mode="test", name=args.wandb_run_name, use_cached=(not args.ignore_cache))

    # normalize features
    if norm_feat:
        train_results["features"] /= train_results["features"].norm(dim=1, p=2, keepdim=True)
        test_results["features"] /= test_results["features"].norm(dim=1, p=2, keepdim=True)

    # TODO: average features across clips of a video
    # NOTE: currently, not doing averaging since I am using a single clip per video

    # Get retrieval benchmarks
    retrieval_dict = retrieval(
        train_results["features"].numpy(),
        train_results["labels"].numpy(),
        list(range(len(train_results["features"]))),
        test_results["features"].numpy(),
        test_results["labels"].numpy(),
        list(range(len(test_results["features"]))),
        train_aud_features=None, 
        val_aud_features=None, 
        task='v-v',
    )



if __name__ == '__main__':
    main()

