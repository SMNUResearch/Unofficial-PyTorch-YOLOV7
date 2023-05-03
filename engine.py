import os
import json
import yaml
import time
import torch
import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from copy import deepcopy
from torch.cuda import amp
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.nn.parallel import DistributedDataParallel

import utils as tool
from model.yolov7 import Model
from loss import ComputeLossOTA
from dataset import create_trainloader, create_testloader

def train(opt):
    # set config
    config = yaml.load(open(opt.CONFIG), Loader=yaml.SafeLoader)

    # set DDP environment: multi-GPU (single-machine)
    world_size = int(os.environ['WORLD_SIZE']) # number of GPUs
    local_rank = int(os.environ['LOCAL_RANK']) # equal to os.environ['RANK']: [0, world_size - 1]
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.distributed.init_process_group(backend='nccl' if torch.distributed.is_nccl_available() else 'gloo')
    tool.init_seed(opt.SEED + local_rank)

    # set cudnn
    if torch.cuda.is_available():
        cudnn.benchmark = False
        cudnn.deterministic = True

    # define model
    model = Model(config['ANCHORS'], config['NUM_CLASSES'], config['STRIDE']).to(device)

    total_batch_size = world_size * opt.BATCH_SIZE
    accumulate = max(round(config['NORMINAL_BATCH_SIZE'] / total_batch_size), 1)
    training_info = config['TRAINING_INFO']
    scale_weight_decay = training_info['WEIGHT_DECAY'] * (total_batch_size * accumulate / config['NORMINAL_BATCH_SIZE'])
    start_epoch = 0

    # define optimizer
    optimizer = tool.smart_optimizer(model, training_info['LR_INIT'], training_info['MOMENTUM'], scale_weight_decay)

    # define scheduler
    lr_f = tool.cosine_annealing(1, training_info['LR_F'], opt.NUM_EPOCHS)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_f)
    scheduler.last_epoch = start_epoch - 1

    # define EMA
    if local_rank == 0:
        ema = tool.EMA(model, config['EMA_INFO'])
    else:
        ema = None

    # SyncBatchNorm
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # define dataset
    grid_size = max(int(max(config['STRIDE'])), 32)
    train_loader = create_trainloader(config['DATA_INFO'], config['AUGMENT_INFO'], config['RECTANGULAR_INFO'], opt.BATCH_SIZE, opt.IMAGE_SIZE, grid_size, local_rank, opt.SEED, opt.NUM_WORKERS)

    # pre-reduce anchor precision
    if local_rank == 0:
        model.half().float()

    # DDP mode
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    scaler = amp.GradScaler(enabled=True)
    num_batch = len(train_loader)
    num_warmup_iterations = max(round(training_info['WARMUP_EPOCHS'] * num_batch), 1000)

    # define loss
    criterion = ComputeLossOTA(next(model.parameters()).device, config['LOSS_INFO'], config['ANCHORS'], config['STRIDE'], config['NUM_CLASSES'])

    # start training
    model.train()
    previous_time = time.time()
    for epoch in range(start_epoch, opt.NUM_EPOCHS):
        total_loss = torch.zeros(4)
        train_loader.sampler.set_epoch(epoch)
        optimizer.zero_grad()
        for i, (image, labels, path, index) in enumerate(train_loader):
            n_i = i + num_batch * epoch
            image = image.to(device, non_blocking=True).float() / 255

            # warmup
            if n_i <= num_warmup_iterations:
                x_i = [0, num_warmup_iterations]
                accumulate = max(1, np.interp(n_i, x_i, [1, config['NORMINAL_BATCH_SIZE'] / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(n_i, x_i, [training_info['WARMUP_BIAS_LR'] if j == 2 else 0.0, x['initial_lr'] * lr_f(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(n_i, x_i, [training_info['WARMUP_MOMENTUM'], training_info['MOMENTUM']])

            # forward
            with amp.autocast(enabled=True):
                output = model(image)
                targets = labels.to(device)
                loss, loss_items = criterion(output, targets, image)
                total_loss += loss_items.cpu().data
                loss *= world_size

            # backward
            scaler.scale(loss).backward()

            # optimize
            if n_i % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

        # scheduler
        lr = [x['lr'] for x in optimizer.param_groups]
        scheduler.step()

        # get remaining time
        current_time = time.time()
        time_left = datetime.timedelta(seconds=(current_time - previous_time)*(opt.NUM_EPOCHS - epoch - 1))
        previous_time = current_time
        if local_rank == 0:
            # save checkpoint
            if not os.path.exists(opt.WEIGHT_PATH):
                os.mkdir(opt.WEIGHT_PATH)

            checkpoint = {'model': deepcopy(model.module).half().state_dict(), 'ema': deepcopy(ema.ema).half().state_dict()}
            weight_file = os.path.join(opt.WEIGHT_PATH, str(epoch + 1) + '.pt')
            torch.save(checkpoint, weight_file, _use_new_zipfile_serialization=False)

            script = ('Epoch:[ %d | %d ]    box loss: %.4f    object loss: %.4f    class loss: %.4f    total loss: %.4f') % (epoch + 1, opt.NUM_EPOCHS, total_loss[0], total_loss[1], total_loss[2], total_loss[3])
            print(script)
            print('Remaining Time:', time_left)

            del checkpoint

    return 0

def cocoAPI_evaluation(opt):
    # set config
    config = yaml.load(open(opt.CONFIG), Loader=yaml.SafeLoader)

    # set cudnn
    if torch.cuda.is_available():
        cudnn.benchmark = False
        cudnn.deterministic = True

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # define model
    model = Model(config['ANCHORS'], config['NUM_CLASSES'], config['STRIDE']).to(device)
    model.load_state_dict(torch.load(opt.TEST_WEIGHT, map_location=device)['ema'])
    model.half()
    model.eval()

    # define dataset
    grid_size = max(int(max(config['STRIDE'])), 32)
    test_loader = create_testloader(config['DATA_INFO'], config['AUGMENT_INFO'], config['RECTANGULAR_INFO'], opt.BATCH_SIZE, opt.IMAGE_SIZE, grid_size, opt.NUM_WORKERS)

    # start test
    output_file = []
    coco_class_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
                     35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                     64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

    with torch.no_grad():
        for i, (image, shapes, path, image_index) in enumerate(test_loader):
            image = image.to(device, non_blocking=True).half() / 255
            out, prediction = model(image)
            output = tool.NMS(prediction, config['NMS_INFO'])
            for index, p in enumerate(output):
                if len(p) == 0:
                    continue

                p_copy = p.clone()
                tool.scale_coords(image[index].shape[1:], p_copy[:, :4], shapes[index][0], shapes[index][1])
                box = tool.xyxy2xywh(p_copy[:, :4])
                box[:, :2] -= box[:, 2:] / 2 # xy center to top-left corner
                image_id = int(path[index].strip().split('.')[0])

                for j, b in zip(p.tolist(), box.tolist()):
                    output_file.append({'image_id': image_id, 'category_id': coco_class_id[int(j[5])], 'bbox': [round(x, 3) for x in b], 'score': round(j[4], 5)})

    # coco API
    if not os.path.exists(opt.OUTPUT_PATH):
        os.mkdir(opt.OUTPUT_PATH)

    predictions_file = os.path.join(opt.OUTPUT_PATH, 'COCOAPI_PRED.json')
    with open(predictions_file, 'w') as f:
        json.dump(output_file, f)

    annotations = COCO(config['DATA_INFO']['ANNOTATIONS_TEST'])
    predictions = annotations.loadRes(predictions_file)
    coco_eval = COCOeval(annotations, predictions, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return 0

