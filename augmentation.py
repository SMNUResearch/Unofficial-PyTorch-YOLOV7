import cv2
import math
import random
import numpy as np

def load_image(index, annotations, data_root, image_size, augmentation):
    annotation = annotations[index].strip().split(' ')
    image_path = data_root + annotation[0]
    image = cv2.imread(image_path) # H*W*C & BGR
    h_base, w_base = image.shape[:2]
    ratio = image_size / max(h_base, w_base)

    if augmentation or ratio > 1:
        mode = cv2.INTER_LINEAR # faster in the training inference (augmentation); enlarge the image: prefer to use INTER_LINEAR;
    else:
        mode = cv2.INTER_AREA # shrink the image: prefer to use INTER_AREA; better results in the test inference (without augmentation);

    image = cv2.resize(image, (int(w_base * ratio), int(h_base * ratio)), interpolation=mode)

    return image, (h_base, w_base), image.shape[:2]

def box_candidates(box1, box2, wh_threshold, aspect_ratio_threshold, area_threshold, eps):
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    aspect_ratio = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))

    return (w2 > wh_threshold) & (h2 > wh_threshold) & (w2 * h2 / (w1 * h1 + eps) > area_threshold) & (aspect_ratio < aspect_ratio_threshold)

def letterbox(image, new_shape, border_eps, border_color, stride, augmentation=True, auto=True):
    shape = image.shape[:2]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not augmentation:
        ratio = min(ratio, 1.0) # better AP

    ratios = (ratio, ratio) # W, H
    new_wh = (int(round(shape[1] * ratios[0])), int(round(shape[0] * ratios[1])))
    dw = new_shape[1] - new_wh[0]
    dh = new_shape[0] - new_wh[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    dw /= 2
    dh /= 2

    if shape[::-1] != new_wh:
        image = cv2.resize(image, new_wh, interpolation=cv2.INTER_LINEAR)

    top, bottom = (int(round(dh - border_eps)), int(round(dh + border_eps)))
    left, right = (int(round(dw - border_eps)), int(round(dw + border_eps)))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_color)

    return image, ratios, (dw, dh)

def get_image_labels(index, annotations, total_labels, data_root, image_size, augmentation, shape, border_eps, border_color, stride):
    image, (h_base, w_base), (h, w) = load_image(index, annotations, data_root, image_size, augmentation)
    image, ratio, pad = letterbox(image, shape, border_eps, border_color, stride, augmentation=augmentation, auto=False)
    labels = total_labels[index].copy()
    labels[:, 0] = ratio[0] * (w / w_base) * labels[:, 0] + pad[0]
    labels[:, 1] = ratio[1] * (h / h_base) * labels[:, 1] + pad[1]
    labels[:, 2] = ratio[0] * (w / w_base) * labels[:, 2] + pad[0]
    labels[:, 3] = ratio[1] * (h / h_base) * labels[:, 3] + pad[1]

    return image, labels

def mosaic_four(index, labels, annotations, data_root, s, mosaic_border, augmentation, color_value):
    new_labels = []
    y_center, x_center = [int(random.uniform(-x, 2 * s + x)) for x in mosaic_border]
    indices = [index] + random.choices(range(len(annotations)), k=3)
    new_image = np.full((s * 2, s * 2, 3), color_value, dtype=np.uint8)
    for i in range(0, len(indices)):
        image, (h_base, w_base), (h, w) = load_image(indices[i], annotations, data_root, s, augmentation)
        if i == 0: # top left
            x1a, y1a, x2a, y2a = max(x_center - w, 0), max(y_center - h, 0), x_center, y_center
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1: # top right
            x1a, y1a, x2a, y2a = x_center, max(y_center - h, 0), min(x_center + w, s * 2), y_center
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2: # bottom left
            x1a, y1a, x2a, y2a = max(x_center - w, 0), y_center, x_center, min(s * 2, y_center + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3: # bottom right
            x1a, y1a, x2a, y2a = x_center, y_center, min(x_center + w, s * 2), min(s * 2, y_center + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        new_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]

        pad_w = x1a - x1b
        pad_h = y1a - y1b
        old_labels = labels[indices[i]].copy()
        old_labels[:, 0] = (w / w_base) * old_labels[:, 0] + pad_w
        old_labels[:, 1] = (h / h_base) * old_labels[:, 1] + pad_h
        old_labels[:, 2] = (w / w_base) * old_labels[:, 2] + pad_w
        old_labels[:, 3] = (h / h_base) * old_labels[:, 3] + pad_h
        new_labels.append(old_labels)

    new_labels = np.concatenate(new_labels, axis=0)
    np.clip(new_labels[:, 0:4], 0, 2 * s, out=new_labels[:, 0:4])
    # remove invalid boxes after the clip operation: w=0 or h=0
    box_w_valid = (new_labels[:, 2] - new_labels[:, 0]) > 0
    box_h_valid = (new_labels[:, 3] - new_labels[:, 1]) > 0
    valid_index = np.logical_and(box_w_valid, box_h_valid)
    new_labels = new_labels[valid_index]

    return new_image, new_labels

def mosaic_nine(index, labels, annotations, data_root, s, mosaic_border, augmentation, color_value):
    new_labels = []
    y_center, x_center = [int(random.uniform(0, s)) for _ in mosaic_border]
    indices = [index] + random.choices(range(len(annotations)), k=8)
    new_image = np.full((s * 3, s * 3, 3), color_value, dtype=np.uint8)
    for i in range(0, len(indices)):
        image, (h_base, w_base), (h, w) = load_image(indices[i], annotations, data_root, s, augmentation)
        if i == 0: # center
            h0, w0 = h, w
            c = s, s, s + w, s + h
        elif i == 1: # top
            c = s, s - h, s + w, s
        elif i == 2: # top right
            c = s + w_p, s - h, s + w_p + w, s
        elif i == 3: # right
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4: # bottom right
            c = s + w0, s + h_p, s + w0 + w, s + h_p + h
        elif i == 5: # bottom
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6: # bottom left
            c = s + w0 - w_p - w, s + h0, s + w0 - w_p, s + h0 + h
        elif i == 7: # left
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8: # top left
            c = s - w, s + h0 - h_p - h, s, s + h0 - h_p

        pad_x, pad_y = c[:2]
        x1, y1, x2, y2 = [max(x, 0) for x in c]
        new_image[y1:y2, x1:x2] = image[y1 - pad_y:, x1 - pad_x:]
        h_p, w_p = h, w
        old_labels = labels[indices[i]].copy()
        old_labels[:, 0] = (w / w_base) * old_labels[:, 0] + pad_x
        old_labels[:, 1] = (h / h_base) * old_labels[:, 1] + pad_y
        old_labels[:, 2] = (w / w_base) * old_labels[:, 2] + pad_x
        old_labels[:, 3] = (h / h_base) * old_labels[:, 3] + pad_y
        new_labels.append(old_labels)

    new_image = new_image[y_center:(y_center + 2 * s), x_center:(x_center + 2 * s)]
    new_labels = np.concatenate(new_labels, axis=0)
    new_labels[:, [0, 2]] -= x_center
    new_labels[:, [1, 3]] -= y_center
    np.clip(new_labels[:, 0:4], 0, 2 * s, out=new_labels[:, 0:4])
    # remove invalid boxes after the clip operation: w=0 or h=0
    box_w_valid = (new_labels[:, 2] - new_labels[:, 0]) > 0
    box_h_valid = (new_labels[:, 3] - new_labels[:, 1]) > 0
    valid_index = np.logical_and(box_w_valid, box_h_valid)
    new_labels = new_labels[valid_index]

    return new_image, new_labels

def random_perspective(image, labels, degree, translate, scale, shear, perspective, border, border_color, wh_threshold, aspect_ratio_threshold, area_threshold, eps):
    height = image.shape[0] + border[0] * 2
    width = image.shape[1] + border[1] * 2
    # center
    C = np.eye(3)
    C[0, 2] = -image.shape[1] / 2
    C[1, 2] = -image.shape[0] / 2
    # perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)
    # rotation and scale
    R = np.eye(3)
    R_angle = random.uniform(-degree, degree)
    R_scale = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=R_angle, center=(0, 0), scale=R_scale)
    # shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    # translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    M = T @ S @ R @ P @ C  # the order of operations (right to left) is IMPORTANT; @: matrix multiply;

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            image = cv2.warpPerspective(image, M, dsize=(width, height), borderValue=border_color)
        else:
            image = cv2.warpAffine(image, M[:2], dsize=(width, height), borderValue=border_color)

    # transform labels
    xy = np.ones((len(labels) * 4, 3))
    xy[:, :2] = labels[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(len(labels) * 4, 2)
    xy = xy @ M.T
    if perspective:
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(len(labels), 8)
    else:
        xy = xy[:, :2].reshape(len(labels), 8)

    # create new boxes
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    new_bboxes = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, len(labels)).T
    # clip
    new_bboxes[:, [0, 2]] = new_bboxes[:, [0, 2]].clip(0, width)
    new_bboxes[:, [1, 3]] = new_bboxes[:, [1, 3]].clip(0, height)
    # filter candidates
    index = box_candidates(labels[:, 0:4].T * R_scale, new_bboxes.T, wh_threshold, aspect_ratio_threshold, area_threshold, eps)
    labels = labels[index]
    labels[:, 0:4] = new_bboxes[index]

    return image, labels

def mix_up(image, labels, image_mix, labels_mix, beta):
    ratio = np.random.beta(beta[0], beta[1])
    labels[:, 5:6] = ratio
    labels_mix[:, 5:6] = 1 - ratio
    new_image = (image * ratio + image_mix * (1 - ratio)).astype(np.uint8)
    new_labels = np.concatenate((labels, labels_mix), axis=0)

    return new_image, new_labels

def augment_hsv(image, h_gain, s_gain, v_gain):
    random_gains = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    dtype = image.dtype
    x = np.arange(0, 256, dtype=np.int16)
    lut_h = ((x * random_gains[0]) % 180).astype(dtype)
    lut_s = np.clip(x * random_gains[1], 0, 255).astype(dtype)
    lut_v = np.clip(x * random_gains[2], 0, 255).astype(dtype)
    image_hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v))).astype(dtype)

    cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR, dst=image)

