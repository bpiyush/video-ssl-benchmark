# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import torch
from torch import nn
import torch.distributed as dist

import utils.logger
from utils import main_utils
import yaml
import os
import numpy as np

def parameter_description(model):
    desc = ''
    for n, p in model.named_parameters():
        desc += "{:70} | {:10} | {:30} | {}\n".format(
            n, 'Trainable' if p.requires_grad else 'Frozen',
            ' x '.join([str(s) for s in p.size()]), str(np.prod(p.size())))
    return desc

def freeze_backbone(model,pretext_model):

    for n, p in model.named_parameters():
        if 'classifier' in n or 'fc' in n or 'linear' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

    if pretext_model == 'gdt':
          print("fida")
          model.classifier.weight.data.normal_(mean=0.0, std=0.01)
          model.classifier.bias.data.zero_()
    elif pretext_model == 'ctp':
          print("fida")
          model.fc.weight.data.normal_(mean=0.0, std=0.01)
          model.fc.bias.data.zero_()
    elif pretext_model== 'video_moco':
          print("fida")
          model.fc.weight.data.normal_(mean=0.0, std=0.01)
          model.fc.bias.data.zero_()
    elif pretext_model== 'pretext_contrast':
          print("fida")
          model.linear.weight.data.normal_(mean=0.0, std=0.01)
          model.linear.bias.data.zero_()
    elif pretext_model== 'rspnet':
          model.fc.weight.data.normal_(mean=0.0, std=0.01)
          model.fc.bias.data.zero_()
    elif pretext_model== 'full_supervision':
          model.fc.weight.data.normal_(mean=0.0, std=0.01)
          model.fc.bias.data.zero_()

    return model

def load_checkpoint_ctp(model,pretrained):
    # load from pre-trained, before DistributedDataParallel constructor
    if pretrained:
        if os.path.isfile(pretrained):
            print("=> loading checkpoint '{}'".format(pretrained))
            checkpoint = torch.load(pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for q,k in zip(list(model.state_dict().keys()),list(state_dict.keys())):
                # retain only encoder_q up to before the embedding layer
                #print(q,k[len("backbone."):])
                if k.startswith('backbone.'):# and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("backbone."):]] = state_dict[k]
                    #state_dict[q] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            #args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(pretrained))

def load_rsp_checkpoint(model,checkpoint_path: str):
        cp = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in cp and 'arch' in cp:
            print('Loading MoCo checkpoint from %s (epoch %d)', checkpoint_path, cp['epoch'])
            moco_state = cp['model']
            prefix = 'encoder_q.'
        else:
            # This checkpoint is from third-party
            #logger.info('Loading third-party model from %s', checkpoint_path)
            print('Loading third-party model from %s', checkpoint_path)
            if 'state_dict' in cp:
                moco_state = cp['state_dict']
            else:
                # For c3d
                moco_state = cp
                #logger.warning('if you are not using c3d sport1m, maybe you use wrong checkpoint')
                print('if you are not using c3d sport1m, maybe you use wrong checkpoint')
            if next(iter(moco_state.keys())).startswith('module'):
                prefix = 'module.'
            else:
                prefix = ''

        """
        fc -> fc. for c3d sport1m. Beacuse fc6 and fc7 is in use.
        """
        blacklist = ['fc.', 'linear', 'head', 'new_fc', 'fc8']
        blacklist += ['encoder_fuse']

        def filter(k):
            return k.startswith(prefix) and not any(k.startswith(f'{prefix}{fc}') for fc in blacklist)

        model_state = {k[len(prefix):]: v for k, v in moco_state.items() if filter(k)}
        msg = model.load_state_dict(model_state, strict=False)
        # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"} or \
        #        set(msg.missing_keys) == {"linear.weight", "linear.bias"} or \
        #        set(msg.missing_keys) == {'head.projection.weight', 'head.projection.bias'} or \
        #        set(msg.missing_keys) == {'new_fc.weight', 'new_fc.bias'},\
        #     msg

        #logger.warning(f'Missing keys: {msg.missing_keys}, Unexpected keys: {msg.unexpected_keys}')
        print(f'Missing keys: {msg.missing_keys}, Unexpected keys: {msg.unexpected_keys}')

def load_pretext_contrast_checkpoint(model,checkpoint_fn):

     def load_pretrained_weights(ckpt_path):
         """load pretrained weights and adjust params name."""
         adjusted_weights = {}
         pretrained_weights = torch.load(ckpt_path)
         for name, params in pretrained_weights.items():
             if 'module' in name:
                 name = name[name.find('module')+7:]
                 adjusted_weights[name] = params
                 print('Pretrained weight name: [{}]'.format(name))
         return adjusted_weights

     if os.path.isfile(checkpoint_fn):
            print("=> loading checkpoint '{}'".format(checkpoint_fn))
            pretrained_weights = load_pretrained_weights(checkpoint_fn)        
            #pretrained_weights = torch.load(args.pretrained)
            msg = model.load_state_dict(pretrained_weights, strict=False) # Set True to check whether loaded successfully
            print(msg)
            #assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(checkpoint_fn))
     else:
            print("=> no checkpoint found at '{}'".format(checkpoint_fn))

def load_video_moco_checkpoint(model,pretrained):
    # load from pre-trained, before DistributedDataParallel constructor
    if pretrained:
        if os.path.isfile(pretrained):
            print("=> loading checkpoint '{}'".format(pretrained))
            checkpoint = torch.load(pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            #args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(pretrained))

def prepare_environment(args, cfg, fold):
    if args.distributed:
        while True:
            try:
                dist.init_process_group(backend='nccl', init_method='tcp://localhost:{}'.format(args.port), world_size=args.world_size, rank=args.gpu)
                break
            except RuntimeError:
                args.port = str(int(args.port) + 1)

    model_cfg = yaml.safe_load(open(args.model_cfg))['model']
    eval_dir = '{}/{}/eval-{}/fold-{:02d}'.format(model_cfg['model_dir'], model_cfg['name'], cfg['benchmark']['name'], fold)
    os.makedirs(eval_dir, exist_ok=True)
    yaml.safe_dump(cfg, open('{}/config.yaml'.format(eval_dir), 'w'))

    logger = utils.logger.Logger(quiet=args.quiet, log_fn='{}/eval.log'.format(eval_dir), rank=args.gpu)
    if any(['SLURM' in env for env in list(os.environ.keys())]):
        logger.add_line("=" * 30 + "   SLURM   " + "=" * 30)
        for env in os.environ.keys():
            if 'SLURM' in env:
                logger.add_line('{:30}: {}'.format(env, os.environ[env]))
    logger.add_line("=" * 30 + "   Config   " + "=" * 30)
    def print_dict(d, ident=''):
        for k in d:
            if isinstance(d[k], dict):
                logger.add_line("{}{}".format(ident, k))
                print_dict(d[k], ident='  '+ident)
            else:
                logger.add_line("{}{}: {}".format(ident, k, str(d[k])))
    print_dict(cfg)
    logger.add_line("=" * 30 + "   Model Config   " + "=" * 30)
    print_dict(model_cfg)

    return eval_dir, model_cfg, logger


def build_dataloader(db_cfg, split_cfg, fold, num_workers, distributed):
    import torch.utils.data as data

    ####################################################
    from datasets import preprocessing
    if db_cfg['transform'] == 'msc+color':
        video_transform = preprocessing.VideoPrep_MSC_CJ(
            crop=(db_cfg['crop_size'], db_cfg['crop_size']),
            num_frames=int(db_cfg['video_fps'] * db_cfg['clip_duration']),
            pad_missing=True,
            augment=split_cfg['use_augmentation'],
            min_area=db_cfg['min_area'],
            color=db_cfg['color'],
        )
    elif db_cfg['transform'] == 'crop+color':
        video_transform = preprocessing.VideoPrep_Crop_CJ(
            crop=(db_cfg['crop_size'], db_cfg['crop_size']),
            num_frames=int(db_cfg['video_fps'] * db_cfg['clip_duration']),
            pad_missing=True,
            augment=split_cfg['use_augmentation'],
        )
    else:
        raise ValueError

    import datasets
    if db_cfg['name'] == 'ucf101':
        dataset = datasets.UCF
    elif db_cfg['name'] == 'hmdb51':
        dataset = datasets.HMDB
    elif db_cfg['name'] == 'kinetics':
        dataset = datasets.Kinetics
    elif db_cfg['name'] == 'something':
        dataset = datasets.SOMETHING
    elif db_cfg['name'] == 'ntu60':
        dataset = datasets.NTU
    elif db_cfg['name'] == 'gym99':
        dataset = datasets.GYM99
    elif db_cfg['name'] == 'gym288':
        dataset = datasets.GYM288
    elif db_cfg['name'] == 'gym_event':
        dataset = datasets.GYM_event
    else:
        raise ValueError('Unknown dataset')

    ######### Transforms from debug repo #########
    # from utils.videotransforms import video_transforms, volume_transforms, tensor_transforms
    # from pytorchvideo.transforms import (
    #     Normalize,
    #     Permute,
    #     ShortSideScale,
    # )
    # from torchvision.transforms import (
    #     Compose,
    #     Lambda,
    #     CenterCrop,
    # )
    # _MEAN = [0.43216, 0.394666, 0.37645]
    # _STD = [0.22803, 0.22145, 0.216989]
    # _CROP_SIZE = 112

    # video_transform = [
    #     # Lambda(lambda x: x / 255.0),
    #     # Permute((3, 0, 1, 2)),
    #     volume_transforms.ClipToTensor(),
    #     Normalize(_MEAN, _STD),
    #     ShortSideScale(_CROP_SIZE),
    #     CenterCrop(_CROP_SIZE),
    # ]
    # video_transform = Compose(video_transform)
    # from utils.debug import debug
    ##############################################

    db = dataset(
        subset=split_cfg['split'].format(fold=fold),
        return_video=True,
        video_clip_duration=db_cfg['clip_duration'],
        video_fps=db_cfg['video_fps'],
        video_transform=video_transform,
        return_audio=False,
        return_labels=True,
        mode=split_cfg['mode'],
        frame_sampling=split_cfg.get("frame_sampling", None),
        clips_per_video=split_cfg['clips_per_video'],
    )
    # debug()
    ####################################################

    ####################################################
    # # use dataset object from video-classification/ repo 
    # from datasets.ucf101 import UCF101
    # from pytorchvideo.transforms import (
    #     Normalize,
    #     Permute,
    #     RandomShortSideScale,
    #     ShortSideScale,
    # )
    # from torchvision.transforms import (
    #     Compose,
    #     Lambda,
    #     RandomCrop,
    #     RandomHorizontalFlip,
    #     CenterCrop,
    # )

    # from utils.debug import debug

    # _MEAN = [0.43216, 0.394666, 0.37645]
    # _STD = [0.22803, 0.22145, 0.216989]
    # _CROP_SIZE = 112
    # _FRAME_INTERVAL = 2
    # _CLIP_LEN = 16

    # transform = [
    #     Lambda(lambda x: x / 255.0),
    #     Permute((3, 0, 1, 2)),
    #     Normalize(_MEAN, _STD),
    #     ShortSideScale(_CROP_SIZE),
    #     CenterCrop(_CROP_SIZE),
    # ]
    # train_mode = "train" in split_cfg["split"]
    # # if train_mode:
    # #     # define the train dataset
    # #     transform += [
    # #         # RandomShortSideScale(min_size=256, max_size=320),
    # #         # RandomCrop(_CROP_SIZE),
    # #         # RandomHorizontalFlip(p=0.5),
    # #     ]
    # # else:
    # #     transform += [CenterCrop(_CROP_SIZE)]
    # transform = Compose(transform)
    
    # phase = "train" if train_mode else "test"

    # db = UCF101(
    #     root="/ssd/pbagad/datasets/ucf101/",
    #     video_prefix="videos",
    #     annotation_path=f"versions/aot_classification_{phase}_01_v3.0.csv",
    #     num_workers=num_workers,
    #     transform=transform,
    #     train_mode=train_mode,
    #     clip_len=_CLIP_LEN,
    #     frame_interval=_FRAME_INTERVAL,
    #     sampler="FrameSampler",
    #     # debug=db_cfg["overfit_tiny"],
    #     # debug_samples=db_cfg["overfit_samples"],
    # )
    # # debug()
    ####################################################
    
    # this is to enable debug mode (i.e. fitting on small number of samples)
    overfit_tiny = db_cfg.get('overfit_tiny', False)
    if overfit_tiny:
        overfit_samples = db_cfg.get('overfit_samples', 10)

        from torch.utils.data import Subset
        np.random.seed(db_cfg.get('overfit_seed', 0))
        indices = np.random.choice(len(db), overfit_samples, replace=False)
        print('::::: Overfitting on {} samples'.format(len(indices)))
        db = Subset(db, indices)

    print(distributed)
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(db)
    else:
        sampler = None

    drop_last = split_cfg['drop_last'] if 'drop_last' in split_cfg else True
    loader = data.DataLoader(
        db,
        batch_size=db_cfg['batch_size']  if split_cfg['mode'] == 'clip' else max(1, db_cfg['batch_size']//split_cfg['clips_per_video']),
        num_workers=num_workers,
        pin_memory=True,
        shuffle=(sampler is None) and split_cfg['use_shuffle'],
        sampler=sampler,
        drop_last=drop_last
    )
    return loader


def build_dataloaders(cfg, fold, num_workers, distributed, logger):
    logger.add_line("=" * 30 + "   Train DB   " + "=" * 30)
    train_loader = build_dataloader(cfg, cfg['train'], fold, num_workers, distributed)
    logger.add_line(str(train_loader.dataset))

    logger.add_line("=" * 30 + "   Test DB   " + "=" * 30)
    test_loader = build_dataloader(cfg, cfg['test'], fold, num_workers, distributed)
    logger.add_line(str(test_loader.dataset))

    logger.add_line("=" * 30 + "   Dense DB   " + "=" * 30)
    dense_loader = build_dataloader(cfg, cfg['test_dense'], fold, num_workers, distributed)
    logger.add_line(str(dense_loader.dataset))

    return train_loader, test_loader, dense_loader


class CheckpointManager(object):
    def __init__(self, checkpoint_dir, rank=0):
        self.checkpoint_dir = checkpoint_dir
        self.best_metric = 0.
        self.rank = rank

    def save(self, model, optimizer, scheduler, epoch, eval_metric=0.):
        if self.rank is not None and self.rank != 0:
            return
        is_best = False
        if eval_metric > self.best_metric:
            self.best_metric = eval_metric
            is_best = True

        main_utils.save_checkpoint(state={
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best=is_best, model_dir=self.checkpoint_dir)

    def last_checkpoint_fn(self):
        return '{}/checkpoint.pth.tar'.format(self.checkpoint_dir)

    def best_checkpoint_fn(self):
        return '{}/model_best.pth.tar'.format(self.checkpoint_dir)

    def checkpoint_fn(self, last=False, best=False):
        assert best or last
        assert not (last and best)
        if last:
            return self.last_checkpoint_fn()
        if best:
            return self.best_checkpoint_fn()

    def checkpoint_exists(self, last=False, best=False):
        return os.path.isfile(self.checkpoint_fn(last, best))

    def restore(self, model, optimizer, scheduler, restore_last=False, restore_best=False):
        checkpoint_fn = self.checkpoint_fn(restore_last, restore_best)
        ckp = torch.load(checkpoint_fn, map_location={'cuda:0': 'cpu'})
        start_epoch = ckp['epoch']
        model.load_state_dict(ckp['state_dict'])
        optimizer.load_state_dict(ckp['optimizer'])
        scheduler.load_state_dict(ckp['scheduler'])
        #for i in range(start_epoch):
        #    scheduler.step(epoch=None)
        return start_epoch


class ClassificationWrapper(torch.nn.Module):
    def __init__(self, feature_extractor, n_classes, feat_dim ,freeze_backbone=False):
        super(ClassificationWrapper, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = torch.nn.Linear(feat_dim, n_classes)

        if freeze_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

    def forward(self, *inputs):
        emb_pool = self.feature_extractor(*inputs)
        emb_pool = emb_pool.view(inputs[0].shape[0], -1)
        logit = self.classifier(emb_pool)
        return logit


def build_model_avid_cma(feat_cfg, eval_cfg, eval_dir, args, logger):
    import networks.avid_cma.models as avid_cma_models
    pretrained_net = avid_cma_models.__dict__[feat_cfg['arch']](**feat_cfg['args'])

    # Load from checkpoint
    checkpoint_fn = '{}/{}'.format(feat_cfg['model_dir'] ,feat_cfg['checkpoint'])
    ckp = torch.load(checkpoint_fn, map_location='cpu')
    pretrained_net.load_state_dict({k.replace('module.', ''): ckp['model'][k] for k in ckp['model']})

    # Wrap with linear-head classifiers
    model = ClassificationWrapper(feature_extractor=pretrained_net.video_model, **eval_cfg['model']['args'])
    ckp_manager = CheckpointManager(eval_dir, rank=args.gpu)

    # Log model description
    logger.add_line("=" * 30 + "   Model   " + "=" * 30)
    logger.add_line(str(model))
    logger.add_line("=" * 30 + "   Parameters   " + "=" * 30)
    logger.add_line(main_utils.parameter_description(model))
    logger.add_line("=" * 30 + "   Pretrained model   " + "=" * 30)
    logger.add_line("File: {}\nEpoch: {}".format(checkpoint_fn, ckp['epoch']))

    ckp_manager = CheckpointManager(eval_dir, rank=args.gpu)

    return model, ckp_manager



def build_model_video_moco(feat_cfg, eval_cfg, eval_dir, args, logger):

    import torchvision.models.video as models
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    # create model
    print("=============> creating model '{}'".format(feat_cfg['arch']))
    model = models.__dict__[feat_cfg['arch']]()

    # Load from checkpoint
    checkpoint_fn = '{}/{}'.format(feat_cfg['model_dir'] ,feat_cfg['checkpoint'])
    load_video_moco_checkpoint(model,checkpoint_fn)

    # Wrap with linear-head classifiers
    model.fc = nn.Linear(eval_cfg['model']['args']['feat_dim'], eval_cfg['model']['args']['n_classes'])

    # Log model description
    logger.add_line("=" * 30 + "   Model   " + "=" * 30)
    logger.add_line(str(model))
    logger.add_line("=" * 30 + "   Parameters   " + "=" * 30)
    logger.add_line(main_utils.parameter_description(model))
    logger.add_line("=" * 30 + "   Pretrained model   " + "=" * 30)
    logger.add_line("File: {}".format(checkpoint_fn))

    ckp_manager = CheckpointManager(eval_dir, rank=args.gpu)
    return model, ckp_manager

def build_model_pretext_contrast(feat_cfg, eval_cfg, eval_dir, args, logger):

    from networks.pretext_contrast.network import R21D

    # create model
    model = R21D(with_classifier= True,num_classes = eval_cfg['model']['args']['n_classes'])

    # Load from checkpoint
    checkpoint_fn = '{}/{}'.format(feat_cfg['model_dir'] ,feat_cfg['checkpoint'])
    load_pretext_contrast_checkpoint(model,checkpoint_fn)

    # Log model description
    logger.add_line("=" * 30 + "   Model   " + "=" * 30)
    logger.add_line(str(model))
    logger.add_line("=" * 30 + "   Parameters   " + "=" * 30)
    logger.add_line(main_utils.parameter_description(model))
    logger.add_line("=" * 30 + "   Pretrained model   " + "=" * 30)
    logger.add_line("File: {}".format(checkpoint_fn))

    ckp_manager = CheckpointManager(eval_dir, rank=args.gpu)
    return model, ckp_manager

def build_model_ctp(feat_cfg, eval_cfg, eval_dir, args, logger):

    #CTP IMPORTS
    from networks.ctp.models.backbones.r3d import R3D, R2Plus1D

    # create ctp model
    model = R2Plus1D( 
        depth=18,
        num_class = eval_cfg['model']['args']['n_classes'],
        num_stages=4,
        stem=dict(
            temporal_kernel_size=3,
            temporal_stride=1,
            in_channels=3,
            with_pool=False,
        ),
        down_sampling=[False, True, True, True],
        channel_multiplier=1.0,
        bottleneck_multiplier=1.0,
        with_bn=True,
        zero_init_residual=False,
        pretrained=None,
    )

    # Load from checkpoint
    checkpoint_fn = '{}/{}'.format(feat_cfg['model_dir'], feat_cfg['checkpoint'])
    load_checkpoint_ctp(model,checkpoint_fn)

    # Log model description
    logger.add_line("=" * 30 + "   Model   " + "=" * 30)
    logger.add_line(str(model))
    logger.add_line("=" * 30 + "   Parameters   " + "=" * 30)
    logger.add_line(main_utils.parameter_description(model))
    logger.add_line("=" * 30 + "   Pretrained model   " + "=" * 30)
    logger.add_line("File: {}".format(checkpoint_fn))

    ckp_manager = CheckpointManager(eval_dir, rank=args.gpu)
    return model, ckp_manager

def build_model_rsp(feat_cfg, eval_cfg, eval_dir, args, logger):
    #rspnet imports
    from networks.rspnet.models import ModelFactory

    # rspnet model
    model_factory = ModelFactory()
    model = model_factory.build_multitask_wrapper(feat_cfg['arch'], eval_cfg['model']['args']['n_classes'])

    # Load from checkpoint
    checkpoint_fn = '{}/{}'.format(feat_cfg['model_dir'], feat_cfg['checkpoint'])
    load_rsp_checkpoint(model,checkpoint_fn)

    # Log model description
    logger.add_line("=" * 30 + "   Model   " + "=" * 30)
    logger.add_line(str(model))
    logger.add_line("=" * 30 + "   Parameters   " + "=" * 30)
    logger.add_line(main_utils.parameter_description(model))
    logger.add_line("=" * 30 + "   Pretrained model   " + "=" * 30)
    logger.add_line("File: {}".format(checkpoint_fn))

    ckp_manager = CheckpointManager(eval_dir, rank=args.gpu)
    return model, ckp_manager


def build_model_tclr(feat_cfg, eval_cfg, eval_dir, args, logger):
    #tclr imports
    from networks.tclr.model import build_r2p1d_classifier

    checkpoint_fn = '{}/{}'.format(feat_cfg['model_dir'], feat_cfg['checkpoint'])

    # tclr model
    model = build_r2p1d_classifier(num_classes = eval_cfg['model']['args']['n_classes'],saved_model_file = checkpoint_fn)
    # Load from checkpoint
    #load_rsp_checkpoint(model,checkpoint_fn)

    # Log model description
    logger.add_line("=" * 30 + "   Model   " + "=" * 30)
    logger.add_line(str(model))
    logger.add_line("=" * 30 + "   Parameters   " + "=" * 30)
    logger.add_line(main_utils.parameter_description(model))
    logger.add_line("=" * 30 + "   Pretrained model   " + "=" * 30)
    logger.add_line("File: {}".format(checkpoint_fn))

    ckp_manager = CheckpointManager(eval_dir, rank=args.gpu)
    return model, ckp_manager


def build_model_gdt(feat_cfg, eval_cfg, eval_dir, args, logger):

    #GDT  imports
    from networks.GDT.model import load_model, Identity
    from networks.GDT.model import load_model_parameters,load_model_finetune

    # GDT  model
    # Load model
    print("Loading model")
    model = load_model(
        model_type=feat_cfg['model_type'],
        vid_base_arch=feat_cfg['vid_base_arch'],
        aud_base_arch=feat_cfg['aud_base_arch'],
        pretrained=False,
        norm_feat=False,
        use_mlp=False,
        num_classes=256,
        args=None,
    )

    # Load from checkpoint
    checkpoint_fn = '{}/{}'.format(feat_cfg['model_dir'] ,feat_cfg['checkpoint'])
    if os.path.exists(checkpoint_fn):
        print("Loading model weights")
        ckpt_dict = torch.load(checkpoint_fn)
        try:
            model_weights = ckpt_dict["state_dict"]
        except:
            model_weights = ckpt_dict["model"]
        epoch = ckpt_dict["epoch"]
        print(f"Epoch checkpoint: {epoch}")
        load_model_parameters(model, model_weights)
        print(f"Loading model done")
    else:
        print(f"Training from scratch")

    # Add FC layer to model for fine-tuning or feature extracting
    agg_model = False
    model = load_model_finetune(
        None,
        model.video_network.base,
        pooling_arch=model.video_pooling if agg_model else None,
        num_ftrs=model.encoder_dim,
        num_classes=eval_cfg['model']['args']['n_classes'],
        use_dropout=False, 
        use_bn=False,
        use_l2_norm=False,
        dropout=0.9,
        agg_model=agg_model,
    )


    # Log model description
    logger.add_line("=" * 30 + "   Model   " + "=" * 30)
    logger.add_line(str(model))
    logger.add_line("=" * 30 + "   Parameters   " + "=" * 30)
    logger.add_line(main_utils.parameter_description(model))
    logger.add_line("=" * 30 + "   Pretrained model   " + "=" * 30)
    logger.add_line("File: {}".format(checkpoint_fn))

    ckp_manager = CheckpointManager(eval_dir, rank=args.gpu)
    return model, ckp_manager


def build_model_selavi(feat_cfg, eval_cfg, eval_dir, args, logger):

    #selavi  imports
    from networks.selavi.model import load_model, get_video_dim
    from networks.selavi.model import Finetune_Model,load_model_parameters

    # selavi  model
    # Load model
    print("Loading model")

    model = load_model(
        vid_base_arch=feat_cfg['vid_base_arch'],
        aud_base_arch=feat_cfg['aud_base_arch'],
        pretrained=False,
        num_classes=400,
        norm_feat=False,
        use_mlp=True,
        headcount=10,
    )

    # Load from checkpoint
    checkpoint_fn = '{}/{}'.format(feat_cfg['model_dir'] ,feat_cfg['checkpoint'])
    if os.path.exists(checkpoint_fn):
        print("Loading model weights")
        ckpt_dict = torch.load(checkpoint_fn)
        try:
            model_weights = ckpt_dict["state_dict"]
        except:
            model_weights = ckpt_dict["model"]
        epoch = ckpt_dict["epoch"]
        print(f"Epoch checkpoint: {epoch}")
        load_model_parameters(model, model_weights)
        print(f"Loading model done")
    else:
        print(f"Training from scratch")

    # Add FC layer to model for fine-tuning 

    model = Finetune_Model(
        model.video_network.base, 
        get_video_dim(vid_base_arch=feat_cfg['vid_base_arch']),
        eval_cfg['model']['args']['n_classes'],
        use_dropout=False, 
        use_bn=False, 
        use_l2_norm=False, 
        dropout=0.7
    )


    # Log model description
    logger.add_line("=" * 30 + "   Model   " + "=" * 30)
    logger.add_line(str(model))
    logger.add_line("=" * 30 + "   Parameters   " + "=" * 30)
    logger.add_line(main_utils.parameter_description(model))
    logger.add_line("=" * 30 + "   Pretrained model   " + "=" * 30)
    logger.add_line("File: {}".format(checkpoint_fn))

    ckp_manager = CheckpointManager(eval_dir, rank=args.gpu)
    return model, ckp_manager


def build_model_full_supervision(feat_cfg, eval_cfg, eval_dir, args, logger):

    import torchvision.models.video as models
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    # create model
    print("=============> creating model '{}'".format(feat_cfg['arch']))
    model = models.__dict__[feat_cfg['arch']](pretrained=False)

    # Load from checkpoint
    checkpoint_fn = '{}/{}'.format(feat_cfg['model_dir'] ,feat_cfg['checkpoint'])
    if os.path.exists(checkpoint_fn):
        print("Loading model weights")
        ckpt_dict = torch.load(checkpoint_fn)
        msg = model.load_state_dict(ckpt_dict, strict=True)
        print(msg)
        print("=> loaded pre-trained model '{}'".format(checkpoint_fn))
    else:
        print(f"Training from scratch")

    # Wrap with linear-head classifiers
    model.fc = nn.Linear(eval_cfg['model']['args']['feat_dim'], eval_cfg['model']['args']['n_classes'])

    # Log model description
    logger.add_line("=" * 30 + "   Model   " + "=" * 30)
    logger.add_line(str(model))
    logger.add_line("=" * 30 + "   Parameters   " + "=" * 30)
    logger.add_line(main_utils.parameter_description(model))
    logger.add_line("=" * 30 + "   Pretrained model   " + "=" * 30)
    logger.add_line("File: {}".format(checkpoint_fn))

    ckp_manager = CheckpointManager(eval_dir, rank=args.gpu)
    return model, ckp_manager


class BatchWrapper:
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def __call__(self, x):
        outs = []
        for i in range(0, x.shape[0], self.batch_size):
            outs += [self.model(x[i:i + self.batch_size])]
        return torch.cat(outs, 0)
