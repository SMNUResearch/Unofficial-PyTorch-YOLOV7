import json
import numpy as np
from collections import defaultdict

class Evaluation(object):
    def __init__(self, gt_file, pred_file, area_range=[0 ** 2, 1e5 ** 2], max_detections_class=100, merge_type='COCO'):
        # prepare ground truths and predictions
        gt = json.load(open(gt_file, 'r'))
        pred = json.load(open(pred_file, 'r'))
        self.gt = defaultdict(list)
        self.pred = defaultdict(list)
        self.image_list = []
        self.category_list = []
        self.area_range = area_range
        self.max_detections_class = max_detections_class
        self.merge_type = merge_type

        for x in gt:
            self.gt[x['image_name']].append(x)
            if x['image_name'] not in self.image_list:
                self.image_list.append(x['image_name'])

            if x['category_id'] not in self.category_list:
                self.category_list.append(x['category_id'])

        self.image_list.sort()
        self.category_list.sort()

        for x in pred:
            self.pred[x['image_name']].append(x)

        # set parameters
        self.bg_threshold = 0.1
        self.IoU_T = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)

    def box_iou(self, box1, box2, pred, gt):
        area1 = box1[:, 2] * box1[:, 3]
        area2 = box2[:, 2] * box2[:, 3]
        box1[:, 2:4] = box1[:, 2:4] + box1[:, 0:2]
        box2[:, 2:4] = box2[:, 2:4] + box2[:, 0:2]
        inter_xmin = np.maximum(box1[:, 0][:, None], box2[:, 0])
        inter_ymin = np.maximum(box1[:, 1][:, None], box2[:, 1])
        inter_xmax = np.minimum(box1[:, 2][:, None], box2[:, 2])
        inter_ymax = np.minimum(box1[:, 3][:, None], box2[:, 3])
        inter = np.maximum(inter_xmax - inter_xmin, 0) * np.maximum(inter_ymax - inter_ymin, 0)
        IoU = inter / (area1[:, None] + area2 - inter)

        for i in range(0, IoU.shape[0]):
            p_class = pred[i]['category_id']
            for j in range(0, IoU.shape[1]):
                g_class = gt[j]['category_id']
                if p_class == g_class:
                    if gt[j]['iscrowd'] == 1:
                        IoU[i, j] = inter[i, j] / area1[i]

        return IoU

    def image_evaluation(self, image_name):
        gt = self.gt[image_name]
        pred = self.pred[image_name]
        T = len(self.IoU_T)
        G = len(gt)
        gt_match = np.zeros((T, G))
        gt_info = np.zeros((G, 4))

        if (G == 0) and (len(pred) == 0):
            return np.zeros((T, 0)), gt_match, pred, np.zeros((0, 3)), gt_info

        for g_index, g in enumerate(gt):
            if (g['bbox_area'] < self.area_range[0]) or (g['bbox_area'] > self.area_range[1]):
                g['ignore'] = 1

        gt_index = np.argsort([x['ignore'] for x in gt], kind='mergesort')
        gt = [gt[i] for i in gt_index]
        gt_ignore = np.array([g['ignore'] for g in gt])
        gt_iscrowd = np.array([g['iscrowd'] for g in gt])

        pred_index = np.argsort([-x['score'] for x in pred], kind='mergesort')
        class_count = np.zeros(len(self.category_list))
        remain_index = []
        for i in pred_index:
            if class_count[pred[i]['category_id']] < self.max_detections_class:
                class_count[pred[i]['category_id']] += 1
                remain_index.append(i)

        pred = [pred[i] for i in remain_index]
        P = len(pred)
        pred_mark = np.zeros((T, P))
        pred_info = np.zeros((P, 4))

        bbox_g = [g['bbox'] for g in gt]
        bbox_p = [p['bbox'] for p in pred]
        bbox_g = np.array(bbox_g)
        bbox_p = np.array(bbox_p)

        for p_index, p in enumerate(pred):
            pred_info[p_index, 0] = p['score']
            pred_info[p_index, 1] = p['category_id']
            pred_info[p_index, 2] = p['bbox_area']
            pred_info[p_index, 3] = self.image_list.index(p['image_name'])

        for g_index, g in enumerate(gt):
            gt_info[g_index, 0] = g['ignore']
            gt_info[g_index, 1] = g['category_id']
            gt_info[g_index, 2] = g['bbox_area']
            gt_info[g_index, 3] = self.image_list.index(g['image_name'])

        if G == 0: # no GT and all predictions are FP
            for t_index, t in enumerate(self.IoU_T):
                for p_index, p in enumerate(pred):
                    pred_mark[t_index, p_index] = 5 # background error

            return pred_mark, gt_match, pred, pred_info, gt_info

        if P == 0: # no prediction and all GT are missed
            return pred_mark, gt_match, pred, pred_info, gt_info

        if (G * P) > 0:
            bbox_IoU = self.box_iou(bbox_p, bbox_g, pred, gt)
            for t_index, t in enumerate(self.IoU_T):
                for p_index, p in enumerate(pred):
                    p_class = p['category_id']
                    max_IoU = t
                    max_IoU_index = -1
                    max_IoU_class = -1
                    skip = 0
                    for g_index, g in enumerate(gt):
                        g_class = g['category_id']
                        if g_class == p_class:
                            IoU = bbox_IoU[p_index, g_index]
                            if IoU > max_IoU_class:
                                max_IoU_class = IoU

                            if (gt_match[t_index, g_index] == 1) and (gt_iscrowd[g_index] == 0):
                                # case 1: IoU < max_IoU and the final max_IoU_index = -1 ---> Not affect the error type;
                                # case 2: IoU < max_IoU and the final max_IoU_index > -1 ---> Not affect the error type;
                                # case 3: IoU >= max_IoU and the final max_IoU_index > -1 ---> Not affect the error type;
                                # case 4: IoU >= max_IoU and the final max_IoU_index = -1 ---> affect the error type: duplicate error -> no GT candidate
                                if IoU >= max_IoU:
                                    skip = 1

                                continue

                            if (max_IoU_index > -1) and (gt_ignore[max_IoU_index] == 0) and (gt_ignore[g_index] == 1):
                                break

                            if IoU >= max_IoU:
                                max_IoU = IoU
                                max_IoU_index = g_index

                    if max_IoU_index > -1: # find a GT candidate
                        if gt[max_IoU_index]['ignore'] == 1:
                            gt_match[t_index, max_IoU_index] = 1
                            pred_mark[t_index, p_index] = -1 # ignore
                        else:
                            if gt_match[t_index, max_IoU_index] == 0: # match GT
                                gt_match[t_index, max_IoU_index] = 1
                                pred_mark[t_index, p_index] = 100 # TP
                            else: # duplicate error
                                pred_mark[t_index, p_index] = 4 # FP
                    else: # no GT candidate
                        if skip == 1: # duplicate error
                            pred_mark[t_index, p_index] = 4 # FP
                        else:
                            if max_IoU_class >= self.bg_threshold: # localization error
                                pred_mark[t_index, p_index] = 2 # FP
                            elif max_IoU_class > -1: # background error
                                pred_mark[t_index, p_index] = 5 # FP
                            else: # p_class is not covered in g_class
                                if np.max(bbox_IoU[p_index]) > t: # classification error
                                    pred_mark[t_index, p_index] = 1 # FP
                                elif np.max(bbox_IoU[p_index]) >= self.bg_threshold: # both classification and localization error
                                    pred_mark[t_index, p_index] = 3 # FP
                                else: # background error
                                    pred_mark[t_index, p_index] = 5 # FP

                # set unmatched predictions outside of area range to ignore
                for p_index, p in enumerate(pred):
                    if (p['bbox_area'] < self.area_range[0]) or (p['bbox_area'] > self.area_range[1]):
                        if pred_mark[t_index, p_index] < 100:
                            pred_mark[t_index, p_index] = -1 # ignore

            return pred_mark, gt_match, pred, pred_info, gt_info

    def evaluate(self):
        T = len(self.IoU_T)
        total_pred_mark = np.zeros((T, 0))
        total_gt_match = np.zeros((T, 0))
        total_pred_info = np.zeros((0, 4))
        total_gt_info = np.zeros((0, 4))
        total_AP = np.zeros((T, len(self.category_list)))
        total_AR = np.zeros((T, len(self.category_list)))

        for image_name in self.image_list:
            pred_mark, gt_match, pred, pred_info, gt_info = self.image_evaluation(image_name)
            if gt_match.shape[1] > 0:
                total_gt_match = np.concatenate((total_gt_match, gt_match), axis=1)

            if pred_mark.shape[1] > 0:
                total_pred_mark = np.concatenate((total_pred_mark, pred_mark), axis=1)

            if gt_info.shape[0] > 0:
                total_gt_info = np.concatenate((total_gt_info, gt_info), axis=0)

            if pred_info.shape[0] > 0:
                total_pred_info = np.concatenate((total_pred_info, pred_info), axis=0)

        sort_index = np.argsort(-total_pred_info[:, 0], kind='mergesort')
        total_pred_info = total_pred_info[sort_index]
        total_pred_mark = total_pred_mark[:, sort_index]

        for category_id in self.category_list:
            class_index_pred = np.where(total_pred_info[:, 1] == float(category_id))[0]
            class_pred_info = total_pred_info[class_index_pred]
            class_pred_mark = total_pred_mark[:, class_index_pred]
            class_index_gt = np.where(total_gt_info[:, 1] == float(category_id))[0]
            class_gt_info = total_gt_info[class_index_gt]

            for i in range(0, T):
                tp = np.where(class_pred_mark[i] == 100, 1, 0)
                fp = np.logical_and(np.where(class_pred_mark[i] == 100, 0, 1), np.where(class_pred_mark[i] == -1, 0, 1))
                fp = np.cumsum(fp)
                tp = np.cumsum(tp)
                num_gt = len(class_gt_info[np.where(class_gt_info[:, 0] == 0)])
                recall = tp / num_gt
                precision = tp / (tp + fp + np.finfo(np.float64).eps)
                AP = self.merge(recall, precision)
                total_AP[i, category_id] = AP
                total_AR[i, category_id] = recall[-1]

        self.summary(total_AP, total_AR)

    def merge(self, recall, precision):
        if self.merge_type == 'COCO':
            recall_T = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)
            for i in range(len(precision) - 1, 0, -1):
                if precision[i] > precision[i - 1]:
                    precision[i - 1] = precision[i]

            recall_index = np.searchsorted(recall, recall_T, side='left')
            AP = np.zeros(len(recall_T))
            try:
                for r_i, p_i in enumerate(recall_index):
                    AP[r_i] = precision[p_i]
            except:
                pass

            return np.mean(AP)
        elif self.merge_type == 'VOC_OLD':
            AP = 0
            for i in np.arange(0., 1.1, 0.1):
                if np.sum(recall >= i) == 0:
                    p = 0
                else:
                    p = np.max(precision[recall >= i])

                AP += p / 11.0

            return AP
        elif self.merge_type == 'VOC_NEW':
            r = np.concatenate(([0.], recall, [1.]))
            p = np.concatenate(([0.], precision, [0.]))

            for i in range(len(p) - 1, 0, -1):
                p[i - 1] = np.maximum(p[i - 1], p[i])

            index = np.where(r[1:] != r[:-1])[0]
            AP = np.sum((r[index + 1] - r[index]) * p[index + 1])

            return AP

    def summary(self, total_AP, total_AR):
        # custom implementation
        print('AP:', np.mean(total_AP))
        print('AP50:', np.mean(total_AP[0]))
        print('AR:', np.mean(total_AR))

