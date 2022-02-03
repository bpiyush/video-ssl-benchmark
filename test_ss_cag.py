"""Tests performance on contrastive action groups on the SS dataset."""

import os
import argparse
import time
from turtle import st
import yaml
import json
import numpy as np
import torch

import utils.logger
from utils import main_utils, eval_utils
import torch.multiprocessing as mp

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from pprint import pprint

# from utils.debug import debug


parser = argparse.ArgumentParser(description='Evaluation on ESC Sound Classification')
parser.add_argument('cfg', metavar='CFG', help='config file')
parser.add_argument('model_cfg', metavar='CFG', help='config file')
parser.add_argument('--ckpt', type=str, help="path to the checkpoint of a finetuned model")
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--test-only', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--port', default='1234')
parser.add_argument('--pretext-model', default='rspnet')


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
    elif args.pretext_model== 'coclr':
            model, ckp_manager =  eval_utils.build_model_coclr(model_cfg, cfg, eval_dir, args, logger)
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
    elif args.pretext_model== 'tclr':
            model, ckp_manager =  eval_utils.build_model_tclr(model_cfg, cfg, eval_dir, args, logger)
    elif args.pretext_model== 'full_supervision':
            model, ckp_manager =  eval_utils.build_model_full_supervision(model_cfg, cfg, eval_dir, args, logger)
    elif args.pretext_model== 'selavi':
            model, ckp_manager =  eval_utils.build_model_selavi(model_cfg, cfg, eval_dir, args, logger)

    return model, ckp_manager


def fetch_contrastive_action_group_mapping(
        filepath="./assets/contrastive_groups_list.txt",
    ):
    class_to_idx_file ="/local-ssd/fmthoker/20bn-something-something-v2/"\
        "something-something-v2-annotations/something-something-v2-labels.json"
    with open(class_to_idx_file) as f:
        class_to_idx = json.load(f)

    label_to_action_grp_dict = {}
    action_grp_to_label_dict = {}
    action_grp_to_target_dict = {}
    merge_grp_dict = {}

    grp_id = 0
    flag = 0

    with open(filepath, "r", encoding='utf-8') as fp:
        for row in fp:
            if row.startswith('# '):
    #             import pdb; pdb.set_trace()
                if row[2].isdigit():
                    mapping = []
                    for c in row[2:].strip():
                        if c.isdigit():
                            mapping.append(int(c))
                    merge_grp_dict[grp_id] = mapping
                continue
            elif not row.strip():
                if flag == 0:
                    flag = 1
                    continue
                else:
                    flag = 0
                    grp_id += 1
            elif row.startswith('##'):
                break
            else:
                label = row.strip().strip(",").strip("\"").strip("'")
                label_to_action_grp_dict[class_to_idx[label]] = grp_id
                if grp_id not in action_grp_to_label_dict:
                    action_grp_to_label_dict[grp_id] = [label]
                    action_grp_to_target_dict[grp_id] = [class_to_idx[label]]                
                else:                
                    action_grp_to_label_dict[grp_id].append(label)
                    action_grp_to_target_dict[grp_id].append(class_to_idx[label])
    
    return (
        label_to_action_grp_dict, 
        action_grp_to_label_dict,
        action_grp_to_target_dict,
        merge_grp_dict,
        class_to_idx,
    )


def get_argmax_over_predefined_targets(targets, logits):
    return logits[targets].argmax()


def merge_classes(data, target_names, mapping_list):
    mapping = {key: val for key, val in enumerate(mapping_list)}
    
    y_pred_new = []
    for val in data['y_pred']:
        y_pred_new.append(mapping[val])
    
    y_true_new = []
    for val in data['y_true']:
        y_true_new.append(mapping[val])
    
    data['y_pred'] = y_pred_new
    data['y_true'] = y_true_new
    
    target_list_done = []
    target_names_new = []
    for i, elem in enumerate(mapping_list):
        if elem not in target_list_done:
            target_names_new.append(target_names[i])
            target_list_done.append(elem)

    return data, target_names_new


def compute_cag_score(logits_matrix, targets_list):
    (
        label_to_action_grp_dict, 
        action_grp_to_label_dict,
        action_grp_to_target_dict,
        merge_grp_dict,
        class_to_idx,
    ) = fetch_contrastive_action_group_mapping()

    action_grp_preds_and_true = []
    for i in range(len(action_grp_to_target_dict)):
        true_ag_label = []
        pred_ag_label = []
        
        targets_action_grp = action_grp_to_target_dict[i]
        for logits, target in zip(logits_matrix, targets_list):
            if str(target.item()) not in targets_action_grp:
                continue
            else:
                print("hi")
                true_ag_label.append(targets_action_grp.index(str(target.item())))
                pred_ag = get_argmax_over_predefined_targets([int(x) for x in targets_action_grp], logits)
                pred_ag_label.append(str(pred_ag.item()))
        
        action_grp_preds_and_true.append({'y_true': true_ag_label, 'y_pred': pred_ag_label})


    idx_to_class = {v: k for k, v in class_to_idx.items()}
    action_grp_ap_average = 0
    action_grp_ap_most_probable_average = 0
    for action_grp, data in enumerate(action_grp_preds_and_true):
        
        target_names = []
        for ind in action_grp_to_target_dict[action_grp]:
            # target_names.append(class_to_idx[ind])
            target_names.append(idx_to_class[ind])
        
        data['y_true'] = [int(x) for x in data['y_true']]
        data['y_pred'] = [int(x) for x in data['y_pred']]

        ## pre-process potential class clubbing
        if action_grp in merge_grp_dict:
            data, target_names = merge_classes(data, target_names, merge_grp_dict[action_grp])

        confusion_mat = confusion_matrix(data['y_true'], data['y_pred'])
        report = classification_report(data['y_true'], data['y_pred'], target_names=target_names)
        
        """
        Calculate metrics for each label, and find their average, weighted by support 
        (the number of true instances for each label). This alters `macro` to account 
        for label imbalance; it can result in an F-score that is not between precision and recall.
        """
        
        precision_recall_fscore_support_val = precision_recall_fscore_support(
                                                        data['y_true'],
                                                        data['y_pred'],
                                                        average='weighted'
                                                        )
        accuracy_group = accuracy_score(data['y_true'], data['y_pred'])
        
        print("#" * 80)
        print("{}: ACTION GROUP".format(action_grp + 1))
        print("\nConfusion Matrix:")
        print(confusion_mat)
        
        for i, name in enumerate(target_names):
            print("{} --> {}".format(i, name))

        ap_most_probable = np.max(np.sum(confusion_mat, axis=1)) / np.sum(confusion_mat)
        print("Average precision = {:.2f}% ({:.2f}%)\n".format(
                            accuracy_group * 100,
                            ap_most_probable * 100))

        action_grp_ap_average += accuracy_group
        action_grp_ap_most_probable_average += ap_most_probable
        
    action_grp_ap_average /= len(action_grp_preds_and_true)
    action_grp_ap_most_probable_average /= len(action_grp_preds_and_true)

    metric = 100 * (action_grp_ap_average - action_grp_ap_most_probable_average) / (1 - action_grp_ap_most_probable_average)
    print("\nCAG score: {:.2f}%".format(metric))
    
    return metric


def main():
    ngpus = torch.cuda.device_count()
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))

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


def main_worker(gpu, ngpus, fold, args, cfg):
    args.gpu = gpu
    args.world_size = ngpus

    # Prepare folder and logger
    eval_dir, model_cfg, logger = eval_utils.prepare_environment(args, cfg, fold)

    # create pretext model
    model, ckp_manager = get_model(model_cfg, cfg, eval_dir, args, logger) 

    # Optimizer
    optimizer, scheduler = main_utils.build_optimizer(model.parameters(), cfg['optimizer'], logger)

    # Datasets
    # train_loader, test_loader, dense_loader = eval_utils.build_dataloaders(
    #     cfg['dataset'], fold, cfg['num_workers'], args.distributed, logger)

    # Distribute
    model = distribute_model_to_cuda(model, args, cfg)

    # load given checkpoint (of a finetuned model)
    path = "/var/scratch/fmthoker/ssl_benchmark/checkpoints/CTP_2/Kinetics/downstream/eval-something-full_finetune_112X112x32/fold-01/model_best.pth.tar"
    # path = args.ckpt
    print(":: Loading checkpoint: ", path)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location='cpu')
        ckpt_state_dict = ckpt['state_dict']
        # ckpt_state_dict = {k.replace('module.', ''): v for k, v in ckpt_state_dict.items()}
        model.load_state_dict(ckpt_state_dict, strict=True)
    else:
        raise Exception("No checkpoint found at: ", path)

    # ################################ Test only ################################
    # if cfg['test_only']:
    #     #start_epoch = ckp_manager.restore(model, optimizer, scheduler, restore_last=True)
    #     start_epoch = ckp_manager.restore(model, optimizer, scheduler, restore_best=True)
    #     logger.add_line("Loaded checkpoint '{}' (epoch {})".format(ckp_manager.best_checkpoint_fn(), start_epoch))

    # ################################ Train ################################
    start_epoch, end_epoch = 0, cfg['optimizer']['num_epochs']
    # if cfg['resume'] and ckp_manager.checkpoint_exists(last=True):
    #     start_epoch = ckp_manager.restore(model, optimizer, scheduler, restore_last=True)
    #     logger.add_line("Loaded checkpoint '{}' (epoch {})".format(ckp_manager.last_checkpoint_fn(), start_epoch))

    # if not cfg['test_only']:
    #     logger.add_line("=" * 30 + "   Training   " + "=" * 30)

    #     # Warmup. Train classifier for a few epochs.
    #     if start_epoch == 0 and 'warmup_classifier' in cfg['optimizer'] and cfg['optimizer']['warmup_classifier']:
    #         n_wu_epochs = cfg['optimizer']['warmup_epochs'] if 'warmup_epochs' in cfg['optimizer'] else 5
    #         cls_opt, _ = main_utils.build_optimizer(
    #             params=[p for n, p in model.named_parameters() if 'feature_extractor' not in n],
    #             cfg={'lr': {'base_lr': cfg['optimizer']['lr']['base_lr'], 'milestones': [n_wu_epochs,], 'gamma': 1.},
    #                  'weight_decay': cfg['optimizer']['weight_decay'],
    #                  'name': cfg['optimizer']['name']}
    #         )
    #         print("class opts",cls_opt)
    #         for epoch in range(n_wu_epochs):
    #             logger.add_line('LR: {}'.format(scheduler.get_last_lr()))
    #             run_phase('train', train_loader, model, cls_opt, epoch, args, cfg, logger)
    #             top1, _ = run_phase('test', test_loader, model, None, epoch, args, cfg, logger)

    #     # Main training loop
    #     for epoch in range(start_epoch, end_epoch):
    #         if args.distributed:
    #             train_loader.sampler.set_epoch(epoch)
    #             test_loader.sampler.set_epoch(epoch)

    #         logger.add_line('='*30 + ' Epoch {} '.format(epoch) + '='*30)
    #         logger.add_line('LR: {}'.format(scheduler._last_lr))
    #         run_phase('train', train_loader, model, optimizer, epoch, args, cfg, logger)
    #         top1, _ = run_phase('test', test_loader, model, None, epoch, args, cfg, logger)
    #         ckp_manager.save(model, optimizer, scheduler, epoch, eval_metric=top1)
    #         scheduler.step(epoch=None)

    ################################ Eval ################################
    logger.add_line('\n' + '=' * 30 + ' Final evaluation ' + '=' * 30)
    cfg['dataset']['test']['clips_per_video'] = 5  # Evaluate clip-level predictions with 25 clips per video for metric stability
    train_loader, test_loader, dense_loader = eval_utils.build_dataloaders(cfg['dataset'], fold, cfg['num_workers'], args.distributed, logger)
    #top1, top5, mean_top1, mean_top5 = run_phase('test', test_loader, model, None, end_epoch, args, cfg, logger)
    outputs, targets = run_phase('test_dense', dense_loader, model, None, end_epoch, args, cfg, logger)

    cag_score = compute_cag_score(outputs, targets)

    from utils.debug import debug
    debug()

    # outputs, targets = run_phase('test', test_loader, model, None, end_epoch, args, cfg, logger)


    # logger.add_line('\n' + '=' * 30 + ' Evaluation done ' + '=' * 30)

    # logger.add_line('Video@1: {:6.2f}'.format(top1_dense))
    # logger.add_line('Video@MeanTop1: {:6.2f}'.format(mean_top1))
    # logger.add_line('Video@5: {:6.2f}'.format(top5_dense))
    # logger.add_line('Video@MeanTop5: {:6.2f}'.format(mean_top5))


def run_phase(phase, loader, model, optimizer, epoch, args, cfg, logger):
    from utils import metrics_utils
    batch_time = metrics_utils.AverageMeter('Time', ':6.3f', window_size=100)
    data_time = metrics_utils.AverageMeter('Data', ':6.3f', window_size=100)
    loss_meter = metrics_utils.AverageMeter('Loss', ':.4e')
    top1_meter = metrics_utils.AverageMeter('Acc@1', ':6.2f')
    # mean_top1_meter = metrics_utils.AverageMeter('Acc@1', ':6.2f')
    top5_meter = metrics_utils.AverageMeter('Acc@5', ':6.2f')
    # mean_top5_meter = metrics_utils.AverageMeter('Acc@1', ':6.2f')
    progress = utils.logger.ProgressMeter(len(loader), meters=[batch_time, data_time, loss_meter, top1_meter, top5_meter],
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

    for it, sample in enumerate(loader):
        data_time.update(time.time() - end)

        video = sample['frames']
        target = sample['label'].cuda()
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
            #print(confidence.size(),labels_tiled.size())
        else:
            confidence = softmax(logits)
            loss = criterion(logits, target)
        #print(confidence.size(),target.size())
        
        all_outputs.append(confidence)
        all_targets.append(target)

        # with torch.no_grad():
        #     acc1, acc5 = metrics_utils.accuracy(confidence, target, topk=(1, 5))
        #     loss_meter.update(loss.item(), target.size(0))
        #     top1_meter.update(acc1[0], target.size(0))
        #     top5_meter.update(acc5[0], target.size(0))

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
    # classwise_top5 = [0 for c in classes]
    # for c in classes:
    #     indices = all_targets == c
    #     mean_top1, mean_top5 = metrics_utils.accuracy(all_outputs[indices], all_targets[indices], topk=(1, 5))
    #     classwise_top1[c] = mean_top1
    #     classwise_top5[c] = mean_top5
    # classwise_top1 = torch.cat(classwise_top1).mean()
    # classwise_top5 = torch.cat(classwise_top5).mean()

    # if args.distributed:
    #     progress.synchronize_meters(args.gpu)
    #     progress.display(len(loader) * args.world_size)

    # return top1_meter.avg, top5_meter.avg, classwise_top1, classwise_top5
    return all_outputs, all_targets


if __name__ == '__main__':
    main()
