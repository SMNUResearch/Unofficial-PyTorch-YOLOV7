import sys
import argparse

sys.dont_write_bytecode = True

from engine import train

if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-SEED', type=int, default=2)
    parser.add_argument('-BATCH_SIZE', type=int, default=32)
    parser.add_argument('-IMAGE_SIZE', type=int, default=640)
    parser.add_argument('-NUM_WORKERS', type=int, default=4)
    parser.add_argument('-NUM_EPOCHS', type=int, default=400)
    parser.add_argument('-CONFIG', type=str, default='./config/YOLOV7_COCO.yaml')
    parser.add_argument('-WEIGHT_PATH', type=str, default='./weights')
    opt = parser.parse_args()

    train(opt)

    exit()

