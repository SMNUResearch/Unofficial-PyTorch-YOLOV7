import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

import utils as tool
import augmentation as aug

class TrainDataset(Dataset):
    def __init__(self, data_info, augment_info, rectangular_info, batch_size, image_size, stride):
        super(TrainDataset, self).__init__()

        self.data_info = data_info
        self.augment_info = augment_info
        self.rectangular_info = rectangular_info
        self.batch_size = batch_size
        self.image_size = image_size
        self.stride = stride

        # load annotations
        self.annotations = self.load_annotations(self.data_info['ANNOTATION_FILE_TRAIN'])
        self.data_root = self.data_info['DATA_ROOT_TRAIN']
        if os.path.isfile(self.data_info['CACHE_FILE_TRAIN']):
            self.cache = torch.load(self.data_info['CACHE_FILE_TRAIN'])
        else:
            self.cache = self.cache_labels(self.data_info['CACHE_FILE_TRAIN'])

        labels, shapes = zip(*self.cache.values())
        self.labels = labels
        self.shapes = np.array(shapes, dtype=np.float64) # H W

        # set rectangular training
        self.rectangular_training = self.rectangular_info['RECTANGULAR_TRAINING']
        if self.rectangular_training:
            self.batch_index = np.floor(np.arange(len(self.annotations)) / self.batch_size).astype(np.int)
            self.batch_num = self.batch_index[-1] + 1
            aspect_ratio = self.shapes[:, 0] / self.shapes[:, 1] # H / W
            aspect_ratio_index = aspect_ratio.argsort()
            self.annotations = [self.annotations[i] for i in aspect_ratio_index]
            self.labels = [self.labels[i] for i in aspect_ratio_index]
            self.shapes = self.shapes[aspect_ratio_index]
            self.aspect_ratio = aspect_ratio[aspect_ratio_index]
            batch_shapes = [[1, 1]] * self.batch_num
            for i in range(0, self.batch_num):
                batch_aspect_ratio = self.aspect_ratio[self.batch_index == i]
                min_batch_aspect_ratio, max_batch_aspect_ratio = batch_aspect_ratio.min(), batch_aspect_ratio.max()
                if max_batch_aspect_ratio < 1:
                    batch_shapes[i] = [max_batch_aspect_ratio, 1]
                elif min_batch_aspect_ratio > 1:
                    batch_shapes[i] = [1, 1 / min_batch_aspect_ratio]

            self.shape_padding = self.rectangular_info['SHAPE_PADDING_TRAIN']
            self.batch_shapes = np.ceil(np.array(batch_shapes) * self.image_size / self.stride + self.shape_padding).astype(np.int) * self.stride

        # set data augmentation
        self.augmentation = self.augment_info['AUGMENTATION']
        self.mosaic = (self.augmentation) and (not self.rectangular_training)
        self.mosaic_border = [-self.image_size // 2, -self.image_size // 2]

    def load_annotations(self, annotation_file):
        with open(annotation_file, 'r') as f:
            annotations = list(filter(lambda x:len(x) > 0, f.readlines()))

        return annotations

    def cache_labels(self, cache_file):
        cache = {}
        for index in range(0, len(self.annotations)):
            annotation = self.annotations[index].strip().split(' ')
            image_path = self.data_root + annotation[0]
            image = cv2.imread(image_path)
            shape = image.shape[:2]
            labels = np.array([list(map(float, label.split(','))) for label in annotation[1:]])
            cache[image_path] = [labels, shape]

        torch.save(cache, cache_file)

        return cache

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # mosaic augmentation scheme
        use_mosaic = (self.mosaic) and (random.random() < self.augment_info['MOSAIC_P'])
        if use_mosaic:
            # mosaic
            if random.random() < self.augment_info['MOSAIC_FOUR_P']:
                image, labels = aug.mosaic_four(index, self.labels, self.annotations, self.data_root, self.image_size, self.mosaic_border, self.augmentation, self.augment_info['BORDER_COLOR'][0])
            else:
                image, labels = aug.mosaic_nine(index, self.labels, self.annotations, self.data_root, self.image_size, self.mosaic_border, self.augmentation, self.augment_info['BORDER_COLOR'][0])

            # random perspective
            image, labels = aug.random_perspective(image, labels,
                                                   self.augment_info['DEGREE'], self.augment_info['TRANSLATE'], self.augment_info['SCALE'], self.augment_info['SHEAR'], self.augment_info['PERSPECTIVE'],
                                                   self.mosaic_border, self.augment_info['BORDER_COLOR'],
                                                   self.augment_info['WH_THRESHOLD'], self.augment_info['ASPECT_RATIO_THRESHOLD'], self.augment_info['AREA_THRESHOLD'], self.augment_info['AVOID_ZERO_EPS'])

            # assign weights to labels
            labels = np.concatenate([labels, np.full((len(labels), 1), 1.0)], axis=1)
            # mixup
            if random.random() < self.augment_info['MIXUP_P']:
                mix_index = random.randint(0, len(self.labels) - 1)
                # mosaic
                if random.random() < self.augment_info['MOSAIC_FOUR_P']:
                    image_mix, labels_mix = aug.mosaic_four(mix_index, self.labels, self.annotations, self.data_root, self.image_size, self.mosaic_border, self.augmentation, self.augment_info['BORDER_COLOR'][0])
                else:
                    image_mix, labels_mix = aug.mosaic_nine(mix_index, self.labels, self.annotations, self.data_root, self.image_size, self.mosaic_border, self.augmentation, self.augment_info['BORDER_COLOR'][0])

                # random perspective
                image_mix, labels_mix = aug.random_perspective(image_mix, labels_mix,
                                                               self.augment_info['DEGREE'], self.augment_info['TRANSLATE'], self.augment_info['SCALE'], self.augment_info['SHEAR'], self.augment_info['PERSPECTIVE'],
                                                               self.mosaic_border, self.augment_info['BORDER_COLOR'],
                                                               self.augment_info['WH_THRESHOLD'], self.augment_info['ASPECT_RATIO_THRESHOLD'], self.augment_info['AREA_THRESHOLD'], self.augment_info['AVOID_ZERO_EPS'])
                # assign weights to labels
                labels_mix = np.concatenate([labels_mix, np.full((len(labels_mix), 1), 1.0)], axis=1)
                # perform mixup
                image, labels = aug.mix_up(image, labels, image_mix, labels_mix, self.augment_info['MIXUP_BETA'])
        else:
            if self.rectangular_training:
                shape = self.batch_shapes[self.batch_index[index]]
            else:
                shape = (self.image_size, self.image_size)
                shape = np.array(shape)

            image, labels = aug.get_image_labels(index, self.annotations, self.labels, self.data_root, self.image_size, self.augmentation, shape, self.augment_info['BORDER_EPS'], self.augment_info['BORDER_COLOR'], self.stride)
            # assign weights to labels
            labels = np.concatenate([labels, np.full((len(labels), 1), 1.0)], axis=1)

        # general augmentation scheme
        if self.augmentation:
            if not use_mosaic:
                # random perspective
                image, labels = aug.random_perspective(image, labels,
                                                       self.augment_info['DEGREE'], self.augment_info['TRANSLATE'], self.augment_info['SCALE'], self.augment_info['SHEAR'], self.augment_info['PERSPECTIVE'],
                                                       (0, 0), self.augment_info['BORDER_COLOR'],
                                                       self.augment_info['WH_THRESHOLD'], self.augment_info['ASPECT_RATIO_THRESHOLD'], self.augment_info['AREA_THRESHOLD'], self.augment_info['AVOID_ZERO_EPS'])
                # mixup
                if random.random() < self.augment_info['MIXUP_P']:
                    if self.rectangular_training:
                        valid_batch = np.logical_and((self.batch_shapes == shape)[:, 0], (self.batch_shapes == shape)[:, 1]).astype('int')
                        valid_batch_index = np.where(valid_batch == 1)[0]
                        min_valid_batch_index = valid_batch_index[0]
                        max_valid_batch_index = valid_batch_index[-1]
                        valid_mixup_index = np.where((self.batch_index >= min_valid_batch_index) & (self.batch_index <= max_valid_batch_index))[0]
                        min_valid_mixup_index = valid_mixup_index[0]
                        max_valid_mixup_index = valid_mixup_index[-1]
                        mix_index = random.randint(min_valid_mixup_index, max_valid_mixup_index) # [min, max]
                        mix_shape = self.batch_shapes[self.batch_index[mix_index]]
                    else:
                        mix_index = random.randint(0, len(self.labels) - 1)
                        mix_shape = np.array((self.image_size, self.image_size))

                    image_mix, labels_mix = aug.get_image_labels(mix_index, self.annotations, self.labels, self.data_root, self.image_size, self.augmentation, mix_shape, self.augment_info['BORDER_EPS'], self.augment_info['BORDER_COLOR'], self.stride)
                    # assign weights to labels
                    labels_mix = np.concatenate([labels_mix, np.full((len(labels_mix), 1), 1.0)], axis=1)
                    # random perspective
                    image_mix, labels_mix = aug.random_perspective(image_mix, labels_mix,
                                                                   self.augment_info['DEGREE'], self.augment_info['TRANSLATE'], self.augment_info['SCALE'], self.augment_info['SHEAR'], self.augment_info['PERSPECTIVE'],
                                                                   (0, 0), self.augment_info['BORDER_COLOR'],
                                                                   self.augment_info['WH_THRESHOLD'], self.augment_info['ASPECT_RATIO_THRESHOLD'], self.augment_info['AREA_THRESHOLD'], self.augment_info['AVOID_ZERO_EPS'])
                    # perform mixup
                    image, labels = aug.mix_up(image, labels, image_mix, labels_mix, self.augment_info['MIXUP_BETA'])

            # colorspace: hue, saturation, value;
            aug.augment_hsv(image, self.augment_info['H_GAIN'], self.augment_info['S_GAIN'], self.augment_info['V_GAIN'])

        # transform and normalize labels
        if len(labels) > 0:
            H, W, C = image.shape
            labels_xywh = np.copy(labels)
            labels_xywh[:, 0:4] = tool.xyxy2xywh(labels[:, 0:4])
            labels_xywh[:, [1, 3]] /= H
            labels_xywh[:, [0, 2]] /= W

        if self.augmentation:
            # flip up-down
            if random.random() < self.augment_info['FLIPUD_P']:
                image = np.flipud(image)
                if len(labels) > 0:
                    labels_xywh[:, 1] = 1 - labels_xywh[:, 1]
            # flip left-right
            if random.random() < self.augment_info['FLIPLR_P']:
                image = np.fliplr(image)
                if len(labels) > 0:
                    labels_xywh[:, 0] = 1 - labels_xywh[:, 0]

        # transform image
        image = image[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB; HWC to CHW;
        image = np.ascontiguousarray(image)
        # final labels
        labels_out = torch.zeros((len(labels), 6))
        if len(labels) > 0:
            labels_out[:, 2:] = torch.from_numpy(labels_xywh[:, 0:4])
            labels_out[:, 1:2] = torch.from_numpy(labels_xywh[:, 4:5])

        return torch.from_numpy(image), labels_out, self.annotations[index].strip().split(' ')[0], index

    @staticmethod
    def collate_fn(batch):
        image, labels, path, image_index = zip(*batch)
        for i, label in enumerate(labels):
            label[:, 0] = i

        return torch.stack(image, dim=0), torch.cat(labels, dim=0), path, image_index

def create_trainloader(data_info, augment_info, rectangular_info, batch_size, image_size, stride, rank, seed, num_workers):
    with tool.torch_distributed_zero_first(rank):
        dataset = TrainDataset(data_info, augment_info, rectangular_info, batch_size, image_size, stride)

    rectangular_training = rectangular_info['RECTANGULAR_TRAINING']
    if rectangular_training: # incompatible with shuffle
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = DistributedSampler(dataset, shuffle=True)

    generator = torch.Generator()
    generator.manual_seed(997 + seed + rank)
    train_loader = tool.InfiniteDataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True, sampler=sampler, collate_fn=dataset.collate_fn, worker_init_fn=tool.seed_worker, generator=generator)

    return train_loader

class TestDataset(Dataset):
    def __init__(self, data_info, augment_info, rectangular_info, batch_size, image_size, stride):
        super(TestDataset, self).__init__()

        self.data_info = data_info
        self.augment_info = augment_info
        self.rectangular_info = rectangular_info
        self.batch_size = batch_size
        self.image_size = image_size
        self.stride = stride

        # load image list
        self.image_list = self.load_image_list(self.data_info['IMAGE_LIST_TEST'])
        self.data_root = self.data_info['DATA_ROOT_TEST']
        if os.path.isfile(self.data_info['CACHE_FILE_TEST']):
            self.cache = torch.load(self.data_info['CACHE_FILE_TEST'])
        else:
            self.cache = self.cache_labels(self.data_info['CACHE_FILE_TEST'])

        shapes = list(self.cache.values())
        self.shapes = np.array(shapes, dtype=np.float64) # H W

        # set rectangular test
        self.rectangular_test = self.rectangular_info['RECTANGULAR_TEST']
        if self.rectangular_test:
            self.batch_index = np.floor(np.arange(len(self.image_list)) / self.batch_size).astype(np.int)
            self.batch_num = self.batch_index[-1] + 1
            aspect_ratio = self.shapes[:, 0] / self.shapes[:, 1] # H / W
            aspect_ratio_index = aspect_ratio.argsort()
            self.image_list = [self.image_list[i] for i in aspect_ratio_index]
            self.shapes = self.shapes[aspect_ratio_index]
            self.aspect_ratio = aspect_ratio[aspect_ratio_index]
            batch_shapes = [[1, 1]] * self.batch_num
            for i in range(0, self.batch_num):
                batch_aspect_ratio = self.aspect_ratio[self.batch_index == i]
                min_batch_aspect_ratio, max_batch_aspect_ratio = batch_aspect_ratio.min(), batch_aspect_ratio.max()
                if max_batch_aspect_ratio < 1:
                    batch_shapes[i] = [max_batch_aspect_ratio, 1]
                elif min_batch_aspect_ratio > 1:
                    batch_shapes[i] = [1, 1 / min_batch_aspect_ratio]

            self.shape_padding = self.rectangular_info['SHAPE_PADDING_TEST']
            self.batch_shapes = np.ceil(np.array(batch_shapes) * self.image_size / self.stride + self.shape_padding).astype(np.int) * self.stride

    def load_image_list(self, list_file):
        with open(list_file, 'r') as f:
            image_list = list(filter(lambda x:len(x) > 0, f.readlines()))

        return image_list

    def cache_labels(self, cache_file):
        cache = {}
        for index in range(0, len(self.image_list)):
            image_path = self.data_root + self.image_list[index].strip().split(' ')[0]
            image = cv2.imread(image_path)
            shape = image.shape[:2]
            cache[image_path] = shape

        torch.save(cache, cache_file)

        return cache

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        if self.rectangular_test:
            shape = self.batch_shapes[self.batch_index[index]]
        else:
            shape = (self.image_size, self.image_size)
            shape = np.array(shape)

        image, (h_base, w_base), (h, w) = aug.load_image(index, self.image_list, self.data_root, self.image_size, False)
        image, ratio, pad = aug.letterbox(image, shape, self.augment_info['BORDER_EPS'], self.augment_info['BORDER_COLOR'], self.stride, augmentation=False, auto=False)
        shapes = (h_base, w_base), ((h / h_base, w / w_base), pad)
        # transform image
        image = image[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB; HWC to CHW;
        image = np.ascontiguousarray(image)

        return torch.from_numpy(image), shapes, self.image_list[index].strip().split(' ')[0], index

    @staticmethod
    def collate_fn(batch):
        image, shapes, path, image_index = zip(*batch)

        return torch.stack(image, dim=0), shapes, path, image_index

def create_testloader(data_info, augment_info, rectangular_info, batch_size, image_size, stride, num_workers):
    dataset = TestDataset(data_info, augment_info, rectangular_info, batch_size, image_size, stride)
    test_loader = tool.InfiniteDataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True, collate_fn=dataset.collate_fn, worker_init_fn=tool.seed_worker)

    return test_loader

