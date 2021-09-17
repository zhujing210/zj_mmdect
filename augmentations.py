# -*- coding:utf-8 -*-
import torch
from torchvision import transforms
import cv2
import numpy as np
import types, math
from numpy import random
import collections

from lib.utils.visualize_utils import vis_img_box

def mask_img(img, boxes, fill_color=[104.0, 117.0, 123.0]):
    if len(img.shape) == 2:
        fill_color = [0]
    for box in boxes:
        xmin, ymin, xmax, ymax = box[0], box[1], box[0] + box[2], box[1] + box[3]
        img[int(ymin):int(ymax), int(xmin):int(xmax)] = np.array(fill_color).astype(img.dtype)
    return img

def random_affine(img, targets, labels, degrees=10, translate=.1, scale=.1, shear=10, border=0):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # targets = [cls, xyxy]#has convert to [xyxy]
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale) if scale is not None else 1.0
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    if translate is not None:
        T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
        T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    if shear is not None:
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)
    
    # Combined rotation matrix
    M = np.dot(np.dot(S, T), R)
    # M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # print("dddd", len(targets), targets.shape)
    # Transform label coordinates
    n = len(targets) if targets is not None else 0
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1 ##[1, 2, 3, 4, 1, 4, 3, 2]
        # xy = (xy @ M.T)[:, :2].reshape(n, 8)
        xy = np.dot(xy, M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T
        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 2] - targets[:, 0]) * (targets[:, 3] - targets[:, 1])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 0:4] = xy[i]
        labels = labels[i]
        if targets.size == 0:
            targets, labels = None, None
        # print("ssssbbb22", len(targets), targets.shape, labels.shape)

    return img, targets, labels


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b, return_zero=False):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    R = inter / area_a
    if not return_zero:
        return R[R!=0]
    else:
        return R

    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class RandomRotate():
    # def __init__(self, degrees=5, translate=0.1, scale=.1, shear=5, border=0):
    def __init__(self, degrees=10, translate=None, scale=None, shear=None, border=0):
        self.degrees = degrees
        self.translate = translate #random(-translate, translate) * edge
        self.scale = scale #random(1-scale, 1+scale)
        self.shear = shear
        self.border = border
    
    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):
            return random_affine(img, boxes, labels, self.degrees, self.translate,
                                                    self.scale, self.shear, self.border)
        else:
            return img, boxes, labels

class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        # print(self.mean, 'debug...')
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        if boxes is None: return image, boxes, labels
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    def __init__(self, size=(300, 300), ratio=1):
        """size: height width"""
        self.size = size
        self.ratio = ratio

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (int(self.size[1]*self.ratio), int(self.size[0]*self.ratio)))
        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels

class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels

class RandomHSV(object):
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
    
    def __call__(self, img, boxes=None, labels=None):
        # if random.randint(2):
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)  # dst=img, no return needed
        return img, boxes, labels

class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'BGR' and self.transform == 'GRAY':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif self.current == 'GRAY' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            #delta = random.uniform(-self.delta, 0)
            #image = np.uint8(np.clip((image*1.0+delta), 0, 255))
            image += delta
        return image, boxes, labels

         
class RandomBlur(object):
    def __init__(self, sigma=5):
        self.sigma = sigma

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            height, width, channels = image.shape
            s = np.random.normal(0, self.sigma, height * width * \
                        channels).reshape(height, width, channels).astype(image.dtype)  #mean, sigma, num
            image = image + s
            # sigma = random.uniform(0, 3.0)
            # image = cv2.GaussianBlur(image, (3, 3), sigma)
        return image, boxes, labels

class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class RandomSampleCrop(object):
    """
    expand , iou_range_random_crop, photometricdistort
    """
    def __init__(self, cfg, size, mean, use_distort=False):
        self.size = size
        self.sample_options = (
            # using entire original input image
            # None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            #(0.7, None, 0.55, 1), #zoom in #0.4 0.6  0.55
            #(0.9, None, 0.4, 1),
            # (0.7, 1, 0.7, 1), #0.8
            (0.9, 1, 0.7, 1), #0.9
            (1, 1, 0.7, 1), 
            None,
        )
        #face
        # self.sample_options = (
        #     (0.95, 1, 0.3, 1),
        #     (1, 1, 0.3, 1), #1
        #     None,
        # )
        self.distort = PhotometricDistort() if use_distort else None
        self.mean = mean
    
    def __call__(self, image, boxes=None, labels=None):
        """
        zoom_in_only 为 true 则 不进行expand。此处的expand即expandtocavas
        """
        zoom_in_only = False
        # if np.min(boxes[:, 2:] - boxes[:, :2]) < 45: zoom_in_only = True #35 88.2
        image, boxes, label, src_img_rect = self.expand(image, boxes, labels, zoom_in_only)
        height, width, _ = image.shape
        call_cnt = 0
        while True:
            mode = random.choice(self.sample_options)
            if mode is None:
                distort_img, max_xy, min_xy = self.crop_distort(image, src_img_rect, None)
                image[int(min_xy[1]):int(max_xy[1]), int(min_xy[0]):int(max_xy[0])] = distort_img
                return image, boxes, labels
            else:
                min_iou, max_iou, min_scale, max_scale = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')
            
            # crop self.size area first failed, but not called twice
            # if call_cnt > 0 and min_scale is None: return image, boxes, labels
            # max trails (50)
            for _ in range(50):
                current_image = image
                ratio = random.uniform(min_scale, max_scale)    #same ratio 
                h, w = height*ratio, width*ratio

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)
                # is min and max overlap constraint satisfied? if not try again
                if overlap.size == 0 or (overlap.min() < min_iou or overlap.max() > max_iou):
                    continue
                
                # make sure bboxes's center in the crop
                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                # mask in that both m1 and m2 are true
                mask = m1 * m2
                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()
                # take only matching gt labels
                current_labels = labels[mask]
                
                #adjust bboxes accordiing to crop
                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]
                
                if self.distort:
                    distort_img, max_xy, min_xy = self.crop_distort(current_image, src_img_rect, rect)
                    current_image[int(min_xy[1]):int(max_xy[1]), int(min_xy[0]):int(max_xy[0])] = distort_img
                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]
                return current_image, current_boxes, current_labels
    
    def crop_distort(self, current_image, src_img_rect, rect):
        """
        crop范围内 颜色变换
        """
        if (src_img_rect is None and rect is None): #return src_image without expand and crop
            h, w = current_image.shape[:2]
            rect = np.array([0, 0, int(w), int(h)])
            max_xy, min_xy = rect[2:], rect[:2]
            # print(0)
        elif src_img_rect is None:
            max_xy, min_xy = rect[2:], rect[:2]
            # print(1)
        elif rect is None:
            max_xy, min_xy = src_img_rect[2:], src_img_rect[:2]
            # print(2)
        else:
            max_xy = np.minimum(src_img_rect[2:], rect[2:])
            min_xy = np.maximum(src_img_rect[:2], rect[:2])
            # print(3)
        distort_img = self.distort(current_image[int(min_xy[1]):int(max_xy[1]), int(min_xy[0]):int(max_xy[0]), :], None, None)[0]
        # print('hsshsss', src_img_rect, rect, distort_img.shape)
        return distort_img, max_xy, min_xy

    def expand(self, image, boxes, labels, zoom_in_only=False):
        if random.randint(2) or zoom_in_only: return image, boxes, labels, None
        height, width, depth = image.shape
        ratio = random.uniform(1, 1.3) #2.1#2.5 #2.2 #2 #1.6 #1.8 #1.3

        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)
        image_rec = np.array([int(left), int(top), int(left+width), int(top+height)])

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)
        expand_image[:, :] = np.array([random.randint(0, 255) for _ in range(3)], dtype=image.dtype) #self.mean
        expand_image[int(top):int(top + height), int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))
        
        return image, boxes, labels, image_rec


class CutOut(object):
    def __init__(self, cfg):
        self.cfg = cfg
        pass

    def __call__(self, image, boxes, labels):
        if random.randint(2): return image, boxes, labels
        h, w = image.shape[:2]
        # create random masks, 10*10, 50*50, 100, 200... #+ [0.0167] * 8 
        # scales = [0.4167]*1 + [0.2083] * 2 + [0.1042] * 4 + [0.0208] * 8 # image size fraction
        scales = [0.2083] + [0.1042] * 2  #[0.3167] + 
        for s in scales:
            color = [random.randint(0, 255) for _ in range(3)]
            cnt = 0
            while cnt < 10:
                # ratio = random.uniform(0.0167, s)
                mask_h = random.randint(1, int(h * s))
                mask_w = random.randint(1, int(w * s))
                # mask_box
                xmin = max(0, random.randint(0, w) - mask_w // 2)
                ymin = max(0, random.randint(0, h) - mask_h // 2)
                xmax = min(w, xmin + mask_w)
                ymax = min(h, ymin + mask_h)
                # return unobscured labels
                if labels is not None and len(labels):
                    mask_box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                    ioa = jaccard_numpy(boxes, mask_box, return_zero=True)  # intersection over area
                    mask = ioa <= 0.6  # remove >50% obscured labels
                    if not mask.any(): # all boxes occlused > 50%.
                        cnt += 1
                        if cnt == 10:
                            boxes, labels = None, None
                            break
                    else: # part boses occlused < 50%
                        mask_bg = ~mask
                        # print("mask_bg....", mask_bg)
                        for box_ in boxes[mask_bg, :]:
                            image[int(box_[1]):int(box_[3]), int(box_[0]):int(box_[2])] = np.array(color, dtype=image.dtype)
                        boxes = boxes[mask, :]
                        labels = labels[mask]
                        break
                else:
                    break
            # apply random color mask   64, 191
            image[ymin:ymax, xmin:xmax] = np.array(color, dtype=image.dtype)
        return image, boxes, labels

class Expand2Canvas(object):
    def __init__(self, cfg, size, mean, use_base=False):
        self.mean = mean
        self.size = size
        self.use_base = use_base

    def __call__(self, image, boxes, labels):
        ch, cw= self.size
        height, width, depth = image.shape
        expand_image = np.zeros((ch, cw, depth), dtype=image.dtype)

        ratio = 1.
        if height > ch or width > cw:
            ratio = np.min(np.array([ch, cw]).astype(np.float) / np.array([height, width]))
        height, width = int(height * ratio), int(width * ratio)

        left = top = 0
        interpolation = cv2.INTER_LINEAR #cv2.INTER_AREA #  
        if not self.use_base:
            left = random.uniform(0, cw - width)
            top = random.uniform(0, ch - height)
            interpolation = random.choice([cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_AREA])
            #random color  # 64, 191
            expand_image[:, :, :] = np.array([random.randint(0, 255) for _ in range(3)], dtype=image.dtype) #self.mean

        image = cv2.resize(image, (width, height), interpolation=interpolation)
        expand_image[int(top):int(top + height), int(left):int(left + width)] = image
        image = expand_image
       
        if boxes is not None: 
            # boxes = boxes.copy()
            boxes *= ratio
            if not self.use_base:
                boxes[:, :2] += np.array((int(left), int(top)), dtype=np.float32)
                boxes[:, 2:] += np.array((int(left), int(top)), dtype=np.float32)
                #remove small face
                boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                #20*20 #18*18
                small_area = 36 #44.4444 #64 #64 #47.61 #36 #20.25 #72 #90. #64. #81.

                mask = boxes_area < small_area
                if mask.any():
                    fill_color = [random.randint(0, 255) for _ in range(3)]
                    image = mask_img(image, boxes[mask, :], fill_color=fill_color)
                
                mask = boxes_area >= small_area
                if not mask.any():
                    boxes, labels = None, None
                else:
                    boxes = boxes[mask, :]
                    labels = labels[mask]

        return image, boxes, labels

class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            if boxes is not None:
                boxes = boxes.copy()
                boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes

class ToGray(object):
    def __call__(self, image, boxes, classes):
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        return image, boxes, classes

class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
            # RandomHSV()
        ]
        self.pd_gray = [
            RandomContrast(),
            ConvertColor(current='BGR', transform='GRAY'),
            ConvertColor(current='GRAY', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    #TODO rand_num=-1 for money
    def __call__(self, image, boxes, labels, rand_num=-1):
        im, boxes, labels = self.rand_brightness(image, boxes, labels)
        if rand_num < 0: distort = Compose(self.pd[:1])
        if rand_num > 0: #for bgr img
            rand_idx = random.randint(rand_num)
            #rand_num=2: for bgr img
            if rand_idx == 0:
                distort = Compose(self.pd[:-1])
            elif rand_idx == 1:
                distort = Compose(self.pd[1:])
            #rand_num=4: for bgr2gray img
            elif rand_idx == 2:
                distort = Compose(self.pd_gray[:-1])
            elif rand_idx == 3:
                distort = Compose(self.pd_gray[1:]) 
            image, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(image, boxes, labels)
        #return im, boxes, labels


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None, tb_writer=None):
        # if tb_writer is not None:
        #     tb_writer.cfg['aug_name'] = '0_input'
        #     target = np.hstack((boxes, np.expand_dims(labels+1, axis=1)))
        #     vis_img_box(img, target, None, tb_writer)
        for i, t in enumerate(self.transforms):
            img, boxes, labels = t(img, boxes, labels)
            if boxes is None: continue
            # target = np.hstack((boxes, np.expand_dims(labels+1, axis=1)))
            # if tb_writer is not None and i + 1 in tb_writer.cfg['aug_vis_list']:
            #     tb_writer.cfg['aug_name'] = '{}_{}'.format(i + 1, type(t).__name__)
            #     vis_img_box(img, target, None, tb_writer)
        return img, boxes, labels

class SSDAugmentation(object):
    def __init__(self, cfg, size=(300, 300), mean=(104, 117, 123),
                 use_base=False, writer=None):
        self.use_base = use_base  # for eval mode
        self.mean = mean
        self.size = size
        self.writer = writer
        self.epoch = 0
        # self.mean = 128
        self.augment = Compose([
            ConvertFromInts(),
            #Resize(self.size),
            ToAbsoluteCoords(),
            RandomSampleCrop(cfg, self.size, self.mean, use_distort=True),
            # ConvertFromInts(),
            Expand2Canvas(cfg, self.size, self.mean, self.use_base),
            RandomBlur(),
            CutOut(cfg),
            # RandomRotate(),
            RandomMirror(),
            ToPercentCoords(),
            # Resize(self.size),
            SubtractMeans(self.mean)
        ])
        self.base_transform = Compose([
            ConvertFromInts(),
            Expand2Canvas(cfg, self.size, self.mean, self.use_base),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels, tb_writer=None):
        if self.use_base or boxes is None:
            return self.base_transform(img, boxes, labels, tb_writer=tb_writer)
        else:
            return self.augment(img, boxes, labels, tb_writer=tb_writer)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img_path = '/home/maolei/data/face_det/sndg/JPEGImages/2.jpg'
    
    # [[xmin, ymin, xmax, ymax, label_ind], ... ]
    img = cv2.imread(img_path)
    h, w, c = img.shape
    boxes = np.array([[426./w, 181./h, 524./w, 279./h, 1], ])   #814, 156, 909, 251

    ssd_aug = SSDAugmentation(None, size=(360, 640))

    transform_img, box, label = ssd_aug(img, boxes[:, :4], boxes[:, 4], None)
    # print(box)
    transform_img += ssd_aug.mean
    cv2.imshow('sss', transform_img.astype(np.uint8))
    cv2.waitKey(0)
    cv2.waitKey(0)
    cv2.destroyWindow('sss')
