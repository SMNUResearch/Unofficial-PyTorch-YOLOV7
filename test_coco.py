import sys
import argparse

sys.dont_write_bytecode = True

from engine import cocoAPI_evaluation

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

    cocoAPI_evaluation(opt)

    exit()

