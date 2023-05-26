import os
import sys
import argparse

sys.dont_write_bytecode = True

import utils as tool
from engine import cocoAPI_evaluation, extract_predictions
from evaluation import Evaluation

if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-BATCH_SIZE', type=int, default=1)
    parser.add_argument('-IMAGE_SIZE', type=int, default=640)
    parser.add_argument('-NUM_WORKERS', type=int, default=4)
    parser.add_argument('-CONFIG', type=str, default='./config/YOLOV7_COCO.yaml')
    parser.add_argument('-COCO_GT', type=str, default='./data/COCO/COCO_VALID_2017.json')
    parser.add_argument('-OUTPUT_PATH', type=str, default='./outputs')
    parser.add_argument('-TEST_WEIGHT', type=str, default='./YOLOV7_COCO.pt')
    parser.add_argument('-USE_COCOAPI', action='store_true')
    parser.add_argument('-AREA_MIN', type=float, default=0)
    parser.add_argument('-AREA_MAX', type=float, default=1e10)
    parser.add_argument('-MAX_DETECTIONS_CLASS', type=int, default=100)
    parser.add_argument('-MERGE_TYPE', type=str, default='COCO')
    opt = parser.parse_args()

    if opt.USE_COCOAPI:
        cocoAPI_evaluation(opt)
    else:
        gt_file = os.path.join(opt.OUTPUT_PATH, 'COCO_GT.json')
        pred_file = os.path.join(opt.OUTPUT_PATH, 'PRED.json')
        if not os.path.exists(gt_file):
            tool.transform_coco_annotation(opt.COCO_GT, opt.OUTPUT_PATH)

        extract_predictions(opt)
        evaluator = Evaluation(gt_file, pred_file, area_range=[opt.AREA_MIN, opt.AREA_MAX], max_detections_class=opt.MAX_DETECTIONS_CLASS, merge_type=opt.MERGE_TYPE)
        evaluator.evaluate()

    exit()

