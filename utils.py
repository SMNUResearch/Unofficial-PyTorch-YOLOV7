import os
import json
import math
import torch
import random
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from contextlib import contextmanager
from torch.utils.data import DataLoader

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2

    return y

def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]

    return y

def clip_coords(boxes, image_shape):
    boxes[:, 0].clamp_(0, image_shape[1])
    boxes[:, 1].clamp_(0, image_shape[0])
    boxes[:, 2].clamp_(0, image_shape[1])
    boxes[:, 3].clamp_(0, image_shape[0])

def scale_coords(image1_shape, coords, image0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(image1_shape[0] / image0_shape[0], image1_shape[1] / image0_shape[1])
        pad = (image1_shape[1] - image0_shape[1] * gain) / 2, (image1_shape[0] - image0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, image0_shape)

    return coords

def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

def box_iou(box1, box2):
    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(dim=2)

    return inter / (area1[:, None] + area2 - inter)

def box_iou_loss(box1, box2, iou_type, eps):
    box1 = box1.T
    box2 = box2.T
    # transform from xywh to xyxy
    b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
    b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
    b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    w1, h1 = box1[2], box1[3]
    w2, h2 = box2[2], box2[3]
    union = w1 * h1 + w2 * h2 - inter
    iou = inter / (union + eps)

    if iou_type == 'IOU':
        return iou
    else:
        c_w = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        c_h = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if iou_type == 'GIOU':
            c_area = c_w * c_h

            return iou - (c_area - union) / (c_area + eps)
        else:
            diagonal = c_w ** 2 + c_h ** 2
            distance = (box2[0] - box1[0]) ** 2 + (box2[1] - box1[1]) ** 2
            if iou_type == 'DIOU':
                return iou - distance / (diagonal + eps)
            else: # CIOU
                v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
                with torch.no_grad():
                    alpha = v / ((1 - iou) + v + eps)

                return iou - (distance / diagonal + v * alpha)

def NMS(prediction, nms_info):
    num_classes = prediction.shape[2] - 5
    candidates = prediction[:, :, 4] > nms_info['CONFIDENCE_THRESHOLD']
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]

    for index, x in enumerate(prediction):
        x = x[candidates[index]]

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        i, j = (x[:, 5:] > nms_info['CONFIDENCE_THRESHOLD']).nonzero(as_tuple=False).T
        x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), dim=1)

        if not x.shape[0]:
            continue
        elif x.shape[0] > nms_info['MAX_BOX']:
            x = x[x[:, 4].argsort(descending=True)[:nms_info['MAX_BOX']]]

        c = x[:, 5:6] * nms_info['CLASS_OFFSET']
        boxes, scores = x[:, :4] + c, x[:, 4]
        k = torchvision.ops.nms(boxes, scores, nms_info['IOU_THRESHOLD'])
        if k.shape[0] > nms_info['MAX_DETECTION']:
            k = k[:nms_info['MAX_DETECTION']]

        output[index] = x[k]

    return output

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    if local_rank >= 1:
        torch.distributed.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        torch.distributed.barrier(device_ids=[0])

def smart_optimizer(model, lr_init, momentum, weight_decay):
    g = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g[2].append(v.bias)

        if isinstance(v, nn.BatchNorm2d):
            g[0].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g[1].append(v.weight)

        if hasattr(v, 'implicit_add'):
            for x in v.implicit_add:
                g[0].append(x.implicit)

        if hasattr(v, 'implicit_mul'):
            for x in v.implicit_mul:
                g[0].append(x.implicit)

    optimizer = optim.SGD(g[0], lr=lr_init, momentum=momentum, nesterov=True)
    optimizer.add_param_group({'params': g[1], 'weight_decay': weight_decay})
    optimizer.add_param_group({'params': g[2]})

    del g

    return optimizer

def cosine_annealing(y1, y2, steps):
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

def transform_coco_annotation(annotation_file, output_path):
    # key 0: ['info']; key 1: ['licenses']; key 2: ['images']; key 3: ['annotations']; key 4: ['categories'];
    annotation = json.load(open(annotation_file, 'r'))
    coco_class_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
                     35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                     64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    output_file = []
    for image_info in annotation['images']:
        image_id = image_info['id']
        image_name = image_info['file_name']
        for GT in annotation['annotations']:
            if GT['image_id'] == image_id:
                bbox = GT['bbox'] # xmin, ymin, W, H
                category_id = coco_class_id.index(GT['category_id'])
                # real bbox_area is bbox[2] * bbox[3]; COCO uses the segmentation area;
                output_file.append({'image_name': image_name, 'category_id': category_id, 'bbox': bbox, 'bbox_area': GT['area'], 'ignore': GT['iscrowd'], 'iscrowd': GT['iscrowd']})

    GT_file = os.path.join(output_path, 'COCO_GT.json')
    with open(GT_file, 'w') as f:
        json.dump(output_file, f)

    return 0

class RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class EMA(object):
    def __init__(self, model, ema_info):
        self.ema_info = ema_info
        self.ema = deepcopy(model).eval()
        self.updates = self.ema_info['UPDATES']
        self.decay = lambda x: self.ema_info['DECAY'] * (1 - math.exp(-x / self.ema_info['TAU']))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            model_state = model.module.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * model_state[k].detach()

