import torch
import torch.nn as nn
import torch.nn.functional as F

import utils as tool

class ComputeLossOTA(nn.Module):
    def __init__(self, device, loss_info, anchors, stride, num_classes):
        super(ComputeLossOTA, self).__init__()

        self.device = device
        self.loss_info = loss_info
        self.num_layers = len(anchors)
        self.num_anchors = len(anchors[0]) // 2
        transform_anchors = torch.FloatTensor(anchors).view(self.num_layers, -1, 2).to(self.device)
        self.stride = torch.FloatTensor(stride).to(self.device)
        self.num_classes = num_classes
        self.anchors = transform_anchors / self.stride.view(-1, 1, 1)
        self.BCE_class = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.loss_info['CLASS_POSITIVE_WEIGHT']], device=self.device))
        self.BCE_object = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.loss_info['OBJECT_POSITIVE_WEIGHT']], device=self.device))
        self.balance = self.loss_info['LOSS_BALANCE']

    def forward(self, output, targets, image):
        loss_class = torch.zeros(1, device=self.device)
        loss_box = torch.zeros(1, device=self.device)
        loss_object = torch.zeros(1, device=self.device)
        matching_image_index, matching_anchor_index, matching_grid_j, matching_grid_i, matching_target, matching_A = self.build_targets(output, targets, image)
        t_gain = [torch.tensor(out.shape, device=self.device)[[3, 2, 3, 2]] for out in output]

        for i, out in enumerate(output):
            image_index = matching_image_index[i]
            anchor_index = matching_anchor_index[i]
            grid_j = matching_grid_j[i]
            grid_i = matching_grid_i[i]
            t_object = torch.zeros_like(out[:, :, :, :, 0], device=self.device)
            num_target = image_index.shape[0]
            if num_target > 0:
                prediction = out[image_index, anchor_index, grid_j, grid_i]
                grid = torch.stack([grid_i, grid_j], dim=1)
                p_xy = prediction[:, :2].sigmoid() * 2 - 0.5
                p_wh = ((prediction[:, 2:4].sigmoid() * 2) ** 2) * matching_A[i]
                p_box = torch.cat((p_xy, p_wh), dim=1)
                t_box = matching_target[i][:, 2:6] * t_gain[i]
                t_box[:, :2] -= grid
                iou = tool.box_iou_loss(p_box, t_box, self.loss_info['IOU_LOSS_TYPE'], self.loss_info['AVOID_ZERO_EPS'])
                loss_box += (1.0 - iou).mean()
                t_object[image_index, anchor_index, grid_j, grid_i] = (1.0 - self.loss_info['IOU_LOSS_RATIO']) + self.loss_info['IOU_LOSS_RATIO'] * iou.detach().clamp(0).type(t_object.dtype)
                t_class = matching_target[i][:, 1].long()
                if self.num_classes > 1:
                    t = torch.full_like(prediction[:, 5:], 0, device=self.device)
                    t[range(num_target), t_class] = 1
                    loss_class += self.BCE_class(prediction[:, 5:], t)

            loss_object += self.BCE_object(out[:, :, :, :, 4], t_object) * self.balance[i]

        loss_box *= (self.loss_info['BOX_GAIN'] * (3.0 / self.num_layers))
        loss_object *= (self.loss_info['OBJECT_GAIN'] * (image.shape[2] / 640) * (image.shape[3] / 640) * (3.0 / self.num_layers))
        loss_class *= (self.loss_info['CLASS_GAIN'] * (self.num_classes / 80.0) * (3.0 / self.num_layers))
        batch_size = image.shape[0]
        loss = loss_box + loss_object + loss_class

        return loss * batch_size, torch.cat((loss_box, loss_object, loss_class, loss)).detach()

    def build_targets(self, output, targets, image):
        indices, A = self.find_positive(output, targets)
        matching_image_index = [[] for out in output]
        matching_anchor_index = [[] for out in output]
        matching_grid_j = [[] for out in output]
        matching_grid_i = [[] for out in output]
        matching_target = [[] for out in output]
        matching_A = [[] for out in output]
        for batch_index in range(0, output[0].shape[0]):
            b_index = (targets[:, 0] == batch_index)
            target = targets[b_index]
            if target.shape[0] == 0:
                continue

            t_xywh = torch.clone(target[:, 2:6])
            t_xywh[:, [0, 2]] *= image[batch_index].shape[2]
            t_xywh[:, [1, 3]] *= image[batch_index].shape[1]
            t_xyxy = tool.xywh2xyxy(t_xywh)
            all_image_index = []
            all_anchor_index = []
            all_grid_j = []
            all_grid_i = []
            all_A = []
            from_which_layer = []
            p_class = []
            p_object = []
            p_xyxy = []

            for i, out in enumerate(output):
                image_index, anchor_index, grid_j, grid_i = indices[i]
                index = (image_index == batch_index)
                image_index, anchor_index, grid_j, grid_i = image_index[index], anchor_index[index], grid_j[index], grid_i[index]
                all_image_index.append(image_index)
                all_anchor_index.append(anchor_index)
                all_grid_j.append(grid_j)
                all_grid_i.append(grid_i)
                all_A.append(A[i][index])
                from_which_layer.append((torch.ones(size=(len(image_index),)) * i).to(self.device))

                fg_pred = out[image_index, anchor_index, grid_j, grid_i]
                p_object.append(fg_pred[:, 4:5])
                p_class.append(fg_pred[:, 5:])
                grid = torch.stack([grid_i, grid_j], dim=1)
                p_xy = (fg_pred[:, :2].sigmoid() * 2 - 0.5 + grid) * self.stride[i]
                p_wh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * A[i][index] * self.stride[i]
                p_xywh = torch.cat([p_xy, p_wh], dim=1)
                p_xyxy.append(tool.xywh2xyxy(p_xywh))

            p_xyxy = torch.cat(p_xyxy, dim=0)
            if p_xyxy.shape[0] == 0:
                continue

            p_object = torch.cat(p_object, dim=0)
            p_class = torch.cat(p_class, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_image_index = torch.cat(all_image_index, dim=0)
            all_anchor_index = torch.cat(all_anchor_index, dim=0)
            all_grid_j = torch.cat(all_grid_j, dim=0)
            all_grid_i = torch.cat(all_grid_i, dim=0)
            all_A = torch.cat(all_A, dim=0)

            pairwise_iou = tool.box_iou(t_xyxy, p_xyxy)
            pairwise_iou_loss = -torch.log(pairwise_iou + self.loss_info['AVOID_ZERO_EPS'])
            top_k = torch.topk(pairwise_iou, min(self.loss_info['PAIRWISE_IOU_K'], pairwise_iou.shape[1]), dim=1)[0]
            dynamic_k = torch.clamp(top_k.sum(dim=1).int(), min=1)

            gt_class_per_image = F.one_hot(target[:, 1].to(torch.int64), self.num_classes).float().unsqueeze(1).repeat(1, p_xyxy.shape[0], 1)
            num_gt = target.shape[0]
            class_pred = p_class.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid() * p_object.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid()
            y = class_pred.sqrt()
            pairwise_class_loss = F.binary_cross_entropy_with_logits(torch.log(y / (1 - y)), gt_class_per_image, reduction='none').sum(dim=2)
            del class_pred
            cost = pairwise_class_loss + self.loss_info['COST_PAIRWISE_IOU_LOSS_WEIGHT'] * pairwise_iou_loss

            matching_matrix = torch.zeros_like(cost, device=self.device)
            for gt_index in range(0, num_gt):
                pos_index = torch.topk(cost[gt_index], k=dynamic_k[gt_index].item(), largest=False)[1]
                matching_matrix[gt_index][pos_index] = 1.0

            del top_k, dynamic_k
            anchor_matching_gt = matching_matrix.sum(dim=0)
            if (anchor_matching_gt > 1).sum() > 0:
                cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)[1]
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0

            fg_mask_inbox = (matching_matrix.sum(dim=0) > 0.0).to(self.device)
            matched_gt_index = matching_matrix[:, fg_mask_inbox].argmax(dim=0)

            from_which_layer = from_which_layer[fg_mask_inbox]
            all_image_index = all_image_index[fg_mask_inbox]
            all_anchor_index = all_anchor_index[fg_mask_inbox]
            all_grid_j = all_grid_j[fg_mask_inbox]
            all_grid_i = all_grid_i[fg_mask_inbox]
            all_A = all_A[fg_mask_inbox]
            target = target[matched_gt_index]

            for i in range(0, self.num_layers):
                layer_index = (from_which_layer == i)
                matching_image_index[i].append(all_image_index[layer_index])
                matching_anchor_index[i].append(all_anchor_index[layer_index])
                matching_grid_j[i].append(all_grid_j[layer_index])
                matching_grid_i[i].append(all_grid_i[layer_index])
                matching_target[i].append(target[layer_index])
                matching_A[i].append(all_A[layer_index])

        for i in range(0, self.num_layers):
            if matching_target[i] != []:
                matching_image_index[i] = torch.cat(matching_image_index[i], dim=0)
                matching_anchor_index[i] = torch.cat(matching_anchor_index[i], dim=0)
                matching_grid_j[i] = torch.cat(matching_grid_j[i], dim=0)
                matching_grid_i[i] = torch.cat(matching_grid_i[i], dim=0)
                matching_target[i] = torch.cat(matching_target[i], dim=0)
                matching_A[i] = torch.cat(matching_A[i], dim=0)
            else:
                matching_image_index[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchor_index[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_grid_j[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_grid_i[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_target[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_A[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_image_index, matching_anchor_index, matching_grid_j, matching_grid_i, matching_target, matching_A

    def find_positive(self, output, targets):
        num_targets = targets.shape[0]
        indices = []
        A = []
        gain = torch.ones(7, device=self.device).long()
        anchor_indices = torch.arange(self.num_anchors, device=self.device).float().view(self.num_anchors, 1).repeat(1, num_targets) # shape: [num_anchors, num_targets]
        targets = torch.cat((targets.repeat(self.num_anchors, 1, 1), anchor_indices[:, :, None]), dim=2) # shape: [num_anchors, num_targets, 7]
        g = self.loss_info['GRID_BIAS']
        offset = (torch.FloatTensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]).to(self.device)) * g

        for i in range(0, self.num_layers):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(output[i].shape)[[3, 2, 3, 2]] # shape: [1, 1, W / s, H / s, W / s, H / s, 1]
            t = targets * gain
            r = t[:, :, 4:6] / anchors[:, None] # WH ratio between box and anchor; shape: [num_anchors, num_targets, 2]
            j = torch.max(r, 1.0 / r).max(dim=2)[0] < self.loss_info['ANCHOR_T']
            t = t[j]

            grid_xy = t[:, 2:4]
            grid_xy_i = gain[[2, 3]] - grid_xy
            j, k = ((grid_xy % 1.0 < g) & (grid_xy > 1.0)).T
            l, m = ((grid_xy_i % 1.0 < g) & (grid_xy_i > 1.0)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(grid_xy)[None] + offset[:, None])[j]

            image_index, class_index = t[:, :2].long().T
            grid_xy = t[:, 2:4]
            grid_wh = t[:, 4:6]
            grid_ij = (grid_xy - offsets).long()
            grid_i, grid_j = grid_ij.T
            anchor_index = t[:, 6].long()
            indices.append((image_index, anchor_index, grid_j.clamp_(0, gain[3] - 1), grid_i.clamp_(0, gain[2] - 1)))
            A.append(anchors[anchor_index])

        return indices, A

