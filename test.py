import os
import sys
import json
import yaml
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.dont_write_bytecode = True

import utils as tool
from model.yolov7 import Model
from dataset import create_testloader

def coco_evaluation(opt):
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
    predictions_file = os.path.join(opt.OUTPUT_PATH, 'predictions.json')
    with open(predictions_file, 'w') as f:
        json.dump(output_file, f)

    annotations = COCO(config['DATA_INFO']['ANNOTATIONS_TEST'])
    predictions = annotations.loadRes(predictions_file)
    coco_eval = COCOeval(annotations, predictions, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-BATCH_SIZE', type=int, default=1)
    parser.add_argument('-IMAGE_SIZE', type=int, default=640)
    parser.add_argument('-NUM_WORKERS', type=int, default=4)
    parser.add_argument('-CONFIG', type=str, default='./config/YOLOV7_COCO.yaml')
    parser.add_argument('-OUTPUT_PATH', type=str, default='./outputs')
    parser.add_argument('-TEST_WEIGHT', type=str, default='./YOLOV7_COCO.pt')
    opt = parser.parse_args()

    # PATH: save predictions
    if not os.path.exists(opt.OUTPUT_PATH):
        os.mkdir(opt.OUTPUT_PATH)

    coco_evaluation(opt)
    exit()

