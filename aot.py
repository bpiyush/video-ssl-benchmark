"""
Arrow of time prediction for a video.
"""

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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


parser = argparse.ArgumentParser(description='Evaluation on ESC Sound Classification')
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


def distribute_model_to_cuda(model, args, cfg):
    if torch.cuda.device_count() == 1:
        model = model.cuda()
    elif args.distributed:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        cfg['dataset']['batch_size'] = max(cfg['dataset']['batch_size'] // args.world_size, 1)
        cfg['num_workers'] = max(cfg['num_workers'] // args.world_size, 1)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        model = torch.nn.DataParallel(model).cuda()
    return model


def get_model(model_cfg, cfg, eval_dir, args, logger):

    if args.pretext_model== 'rspnet':
            model, ckp_manager =  eval_utils.build_model_rsp(model_cfg, cfg, eval_dir, args, logger)
    elif args.pretext_model== 'avid_cma':
            model, ckp_manager =  eval_utils.build_model_avid_cma(model_cfg, cfg, eval_dir, args, logger)
    elif args.pretext_model== 'pretext_contrast':
            model, ckp_manager =  eval_utils.build_model_pretext_contrast(model_cfg, cfg, eval_dir, args, logger)
    elif args.pretext_model== 'gdt':
            model, ckp_manager =  eval_utils.build_model_gdt(model_cfg, cfg, eval_dir, args, logger)
    elif args.pretext_model== 'ctp':
            model, ckp_manager =  eval_utils.build_model_ctp(model_cfg, cfg, eval_dir, args, logger)
    elif args.pretext_model== 'video_moco':
            model, ckp_manager =  eval_utils.build_model_video_moco(model_cfg, cfg, eval_dir, args, logger)
    elif args.pretext_model== 'selavi':
            model, ckp_manager =  eval_utils.build_model_selavi(model_cfg, cfg, eval_dir, args, logger)
    elif args.pretext_model== 'full_supervision':
            model, ckp_manager =  eval_utils.build_model_full_supervision(model_cfg, cfg, eval_dir, args, logger)

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


def build_dataloaders_debug(cfg):
    from torchvision.transforms import (
        Compose,
        Lambda,
        CenterCrop,
    )
    from pytorchvideo.transforms import (
        Normalize,
        Permute,
        ShortSideScale,
    )
    from torch.utils.data import DataLoader
    from datasets.ucf_debug import UCF101Debug

    # Constants
    # For ResNet3D (torchvision): https://pytorch.org/vision/stable/models.html#video-classification
    _MEAN = [0.43216, 0.394666, 0.37645]
    _STD = [0.22803, 0.22145, 0.216989]
    _CROP_SIZE = 112
    BATCH_SIZE = cfg["dataset"]["batch_size"]
    # BATCH_SIZE = 4
    NUM_WORKERS = cfg["num_workers"]

    # define the datasets
    transform = Compose(
        [
            Lambda(lambda x: x / 255.0),
            Permute((3, 0, 1, 2)),
            Normalize(_MEAN, _STD),
            ShortSideScale(_CROP_SIZE),
            CenterCrop(_CROP_SIZE),
        ]
    )
    print("Loading train dataset")
    train_dataset = UCF101Debug(
        root="/ssd/pbagad/datasets/ucf101/",
        video_prefix="videos",
        annotation_path="versions/aot_classification_train_01_v3.0.csv",
        clip_len=8,
        train_mode=True,
        transform=transform,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    print("Loading valid dataset")
    valid_dataset = UCF101Debug(
        root="/ssd/pbagad/datasets/ucf101/",
        video_prefix="videos",
        annotation_path="versions/aot_classification_test_01_v3.0.csv",
        clip_len=8,
        train_mode=False,
        transform=transform,
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
    return train_dataloader, valid_dataloader


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

    # Optimizer
    optimizer, scheduler = main_utils.build_optimizer(model.parameters(), cfg['optimizer'], logger)

    # Datasets
    train_loader, test_loader, dense_loader = eval_utils.build_dataloaders(
        cfg['dataset'], fold, cfg['num_workers'], args.distributed, logger)
    # train_loader, test_loader = build_dataloaders_debug(cfg)
    # debug()

    # Distribute
    model = distribute_model_to_cuda(model, args, cfg)

    ################################ Test only ################################
    if cfg['test_only']:
        start_epoch = ckp_manager.restore(model, optimizer, scheduler, restore_best=True)
        logger.add_line("Loaded checkpoint '{}' (epoch {})".format(ckp_manager.best_checkpoint_fn(), start_epoch))

    ################################ Train ################################
    start_epoch, end_epoch = 0, cfg['optimizer']['num_epochs']
    if cfg['resume'] and ckp_manager.checkpoint_exists(last=True):
        start_epoch = ckp_manager.restore(model, optimizer, scheduler, restore_last=True)
        logger.add_line("Loaded checkpoint '{}' (epoch {})".format(ckp_manager.last_checkpoint_fn(), start_epoch))

    if not cfg['test_only']:
        logger.add_line("=" * 30 + "   Training   " + "=" * 30)

        # Warmup. Train classifier for a few epochs.
        if start_epoch == 0 and 'warmup_classifier' in cfg['optimizer'] and cfg['optimizer']['warmup_classifier']:
            n_wu_epochs = cfg['optimizer']['warmup_epochs'] if 'warmup_epochs' in cfg['optimizer'] else 5
            cls_opt, _ = main_utils.build_optimizer(
                params=[p for n, p in model.named_parameters() if 'feature_extractor' not in n],
                cfg={'lr': {'base_lr': cfg['optimizer']['lr']['base_lr'], 'milestones': [n_wu_epochs,], 'gamma': 1.},
                     'weight_decay': cfg['optimizer']['weight_decay'],
                     'name': cfg['optimizer']['name']}
            )
            print("class opts",cls_opt)
            for epoch in range(n_wu_epochs):
                logger.add_line('LR: {}'.format(scheduler.get_last_lr()))
                run_phase('train', train_loader, model, cls_opt, epoch, args, cfg, logger)
                top1, _ = run_phase('test', test_loader, model, None, epoch, args, cfg, logger)

        # Main training loop
        for epoch in range(start_epoch, end_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                test_loader.sampler.set_epoch(epoch)

            logger.add_line('='*30 + ' Epoch {} '.format(epoch) + '='*30)
            logger.add_line('LR: {}'.format(scheduler._last_lr))
            run_phase('train', train_loader, model, optimizer, epoch, args, cfg, logger)
            top1, _, classwise_top1, _ = run_phase('test', test_loader, model, None, epoch, args, cfg, logger)
            ckp_manager.save(model, optimizer, scheduler, epoch, eval_metric=top1)
            scheduler.step(epoch=None)

    ################################ Eval ################################
    logger.add_line('\n' + '=' * 30 + ' Final evaluation ' + '=' * 30)
    cfg['dataset']['test']['clips_per_video'] = 5  # Evaluate clip-level predictions with 25 clips per video for metric stability
    train_loader, test_loader, dense_loader = eval_utils.build_dataloaders(cfg['dataset'], fold, cfg['num_workers'], args.distributed, logger)
    top1, top5, mean_top1, mean_top5 = run_phase('test', test_loader, model, None, end_epoch, args, cfg, logger)
    # top1_dense, top5_dense = run_phase('test_dense', dense_loader, model, None, end_epoch, args, cfg, logger)

    logger.add_line('\n' + '=' * 30 + ' Evaluation done ' + '=' * 30)
    logger.add_line('Clip@1: {:6.2f}'.format(top1))
    logger.add_line('Clip@MeanTop1: {:6.2f}'.format(mean_top1))
    logger.add_line('Clip@5: {:6.2f}'.format(top5))
    logger.add_line('Clip@MeanTop5: {:6.2f}'.format(mean_top5))
    # logger.add_line('Video@1: {:6.2f}'.format(top1_dense))
    # logger.add_line('Video@5: {:6.2f}'.format(top5_dense))


def run_phase(phase, loader, model, optimizer, epoch, args, cfg, logger):
    from utils import metrics_utils
    batch_time = metrics_utils.AverageMeter('Time', ':6.3f', window_size=100)
    data_time = metrics_utils.AverageMeter('Data', ':6.3f', window_size=100)
    loss_meter = metrics_utils.AverageMeter('Loss', ':.4e')
    top1_meter = metrics_utils.AverageMeter('Acc@1', ':6.2f')
    # mean_top1_meter = metrics_utils.AverageMeter('Acc@1', ':6.2f')
    # top5_meter = metrics_utils.AverageMeter('Acc@5', ':6.2f')
    # mean_top5_meter = metrics_utils.AverageMeter('Acc@1', ':6.2f')
    progress = utils.logger.ProgressMeter(len(loader), meters=[batch_time, data_time, loss_meter, top1_meter],
                                          phase=phase, epoch=epoch, logger=logger)

    # switch to train/test mode
    model.train(phase == 'train')
    if phase in {'test_dense', 'test'}:
        model = eval_utils.BatchWrapper(model, cfg['dataset']['batch_size'])

    criterion = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)

    end = time.time()
    logger.add_line('\n{}: Epoch {}'.format(phase, epoch))

    # collect all predictions and targets
    all_outputs = []
    all_targets = []
    
    # wandb_logs = {}

    for it, sample in enumerate(loader):
        data_time.update(time.time() - end)

        video_forward = sample['frames']
        label_forward = torch.ones_like(sample['label']).cuda()

        video_reverse = torch.flip(video_forward, [2])
        label_reverse = torch.zeros_like(sample['label']).cuda()

        video = torch.cat((video_forward, video_reverse), dim=0)
        target = torch.cat((label_forward, label_reverse), dim=0)
        # target = torch.cat([sample['label'], sample['label']]).cuda()

        if args.gpu is not None:
            video = video.cuda(args.gpu, non_blocking=True)
        if torch.cuda.device_count() == 1 and args.gpu is None:
            video = video.cuda()
        #print(video.size())

        # compute outputs
        if phase == 'test_dense':
            batch_size, clips_per_sample = video.shape[0], video.shape[1]
            video = video.flatten(0, 1).contiguous()
        if phase == 'train':
            logits = model(video)
        else:
            with torch.no_grad():
                logits = model(video)

        # compute loss and accuracy
        if phase == 'test_dense':
            confidence = softmax(logits).view(batch_size, clips_per_sample, -1).mean(1)
            labels_tiled = target.unsqueeze(1).repeat(1, clips_per_sample).view(-1)
            loss = criterion(logits, labels_tiled)
        else:
            confidence = softmax(logits)
            loss = criterion(logits, target)
        
        # print(confidence)
        # print(torch.argmax(confidence, dim=1))
        # print(target)

        all_outputs.append(confidence)
        all_targets.append(target)

        with torch.no_grad():
            # acc1, acc5 = metrics_utils.accuracy(confidence, target, topk=(1, 5))
            acc1, = metrics_utils.accuracy(confidence, target, topk=(1,))
            # mean_acc1, mean_acc5 = metrics_utils.classwise_mean_accuracy(confidence, target, topk=(1, 5))
            # debug()
            loss_meter.update(loss.item(), target.size(0))
            top1_meter.update(acc1[0], target.size(0))
            
            if phase == "train" and not args.no_wandb:
                wandb.log(
                    {
                        phase: {
                            f"batch-{loss_meter.name}": loss_meter.val,
                            f"batch-{top1_meter.name}": top1_meter.val,
                        },
                    },
                )
            
            # top5_meter.update(acc5[0], target.size(0))
            # mean_top1_meter.update(mean_acc1[0], target.size(0))
            # mean_top5_meter.update(mean_acc5[0], target.size(0))

        # compute gradient and do SGD step
        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (it + 1) % 100 == 0 or it == 0 or it + 1 == len(loader):
            progress.display(it+1)
    
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    classes = all_targets.unique()

    # classwise_top1 = [0 for c in classes]
    # # classwise_top5 = [0 for c in classes]
    # for c in classes:
    #     indices = all_targets == c
    #     # mean_top1, mean_top5 = metrics_utils.accuracy(all_outputs[indices], all_targets[indices], topk=(1, 5))
    #     mean_top1, = metrics_utils.accuracy(all_outputs[indices], all_targets[indices], topk=(1,))
    #     classwise_top1[c] = mean_top1
    #     # classwise_top5[c] = mean_top5
    # classwise_top1 = torch.cat(classwise_top1).mean()
    # # classwise_top5 = torch.cat(classwise_top5).mean()
    
    if not args.no_wandb:
        wandb.log(
            {
                phase: {
                    f"epoch-{loss_meter.name}": loss_meter.avg,
                    f"epoch-{top1_meter.name}": top1_meter.avg,
                },
                "epoch": epoch,
            },
        )

    if args.distributed:
        progress.synchronize_meters(args.gpu)
        progress.display(len(loader) * args.world_size)

    return top1_meter.avg, 0, 0, 0


if __name__ == '__main__':
    main()
