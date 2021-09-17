# -*- coding: UTF-8 -*-
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import xmltodict
import time, random
import cv2
from math import sqrt 

file_list = 'test.txt'
data_name = 'tempLicence_20180803'
data_root = '/home/maolei/data/temp_LP/'


file_list = 'dms_det_ditie0205/face_det_mask200128_train.lst.lst' #'dms_20181226.lst' #'DMS_0614_train.lst' #'DMS_0708/DMS_phone_0708.lst' #'DMS_face_attribute_0429_train.txt' #
data_name = 'dms_det_mask'
data_root = '/home/maolei/data/dms_det/'


# file_list = 'image_0101_yinyang.lst' #'DMS_0711/nocoverface.lst' #sndgcar_train.lst'
# data_name = ''
# data_root = '/home/maolei/data/face_attr/yinyang_face/' #'/home/maolei/data/face_det/'

file_list = 'maolei/save3txt_0501_high_all.txt' #'sndg_train.lst' #'face_side1.8.lst' #'face_det_mask/face_det_mask200128/face_det_mask200128_train.lst'
data_name = ''
data_root = '/mnt/data2/reid_down/infraredface/'

file_list = 'facedet_badcase_sideface/gucci_xj_zunyi.lst' #'sndg_train.lst' #'face_side1.8.lst' #
data_name = ''
data_root = '/home/maolei/data/face_det/'

# file_list = 'xinjiang_c.lst' #'furg_fire.lst'
# data_name = ''
# data_root = '/home/maolei/data/tmp/face_det_testbug/'

data_path = '{}/{}'.format(data_root, data_name)

def save_list2txt(img_gt_list, file_path):

    if img_gt_list is None or len(img_gt_list) == 0:
        print('save failed', file_path)
        return
    fw = open(file_path, 'w')
    fw.writelines(img_gt_list)
    fw.close()

class_dict = {'poi_water':0,'poi_phone':1,'poi_palm':2,'poi_face':3}
def get_gt(file_path, anno_key=None, resize_hw=None):
    """ get gt and hw from xml path. do not need to know the category in advance
    Args:
        file_path (str): xml file path
        resize_hw (tuple | None): Resized img size
    return:
        gt (np.array()): each contains [int(xmin), int(ymin), int(xmax), int(ymax), class_dict[name], -1]
        hw (list(int|folat)): square | h | w
    """
    global class_dict
    gt = []
    hw = []
    with open(file_path, 'r') as f:
        d = xmltodict.parse(f.read())
        anno = d['annotation']
        # folder = anno['folder']
        filename = anno['filename']
        width = int(anno['size']['width'])
        height = int(anno['size']['height'])
        depth = anno['size']['depth']
        if not 'object' in anno.keys():
            return np.array(gt), hw
        
        objs = anno['object']
        if not isinstance(objs, list):   #if len(objs) is one, objs will not be list
            objs = [objs]
        
        for obj in objs:
            name = obj['name'].lower().strip()
            # if 'fjs_' in name: continue
            if anno_key and name not in anno_key: continue
            xmin = int(float(obj['bndbox']['xmin']))
            ymin = int(float(obj['bndbox']['ymin']))
            xmax = int(float(obj['bndbox']['xmax']))
            ymax = int(float(obj['bndbox']['ymax']))

            if resize_hw is not None:
                ratio_h = resize_hw[0] / float(height)
                ratio_w = resize_hw[1] /  float(width)
                xmin = int(ratio_w * xmin)
                ymin = int(ratio_h * ymin)
                xmax = int(ratio_w * xmax)
                ymax = int(ratio_h * ymax)
                
            HW = [(ymax - ymin), (xmax - xmin)]
            if min(HW) < 2: continue
            # area = HW[0] * HW[1]
            # HW = [int(sqrt(area))]
            HW = [HW[1]] #[HW[0]] #[HW[1]*1.0 / HW[0]] #
            hw += HW
            if name not in class_dict.keys():
                class_dict[name] = len(class_dict)
            
            gt.append([int(xmin), int(ymin), int(xmax), int(ymax), class_dict[name], -1])
    return np.array(gt), hw

def get_onelinepred_results(pred_file, thred=0.1):
    """"from pred_file parse pred_results
    Args:
        # TODO save format of  pred_file still unknown
        pred_file (str): pred_file path 
        thred: pred_box's score less than it could be ignored
    Return:
        pred_dict (dict(list)) : output predict result. The outer dict means different images
                                , inner list contains  xywh class_id(1) score
    
    """
    if pred_file is None: return None
    
    pred_dict = {}
    lines = open(pred_file, 'r').readlines()
    for line in lines:
        split_item = line.strip().split()
        if len(split_item) < 5: continue
        image_path = split_item[0]

         #image key first occur
        if not image_path in pred_dict.keys():
            pred_dict[image_path] = list()

        pred_box = np.array(split_item[1:]).reshape((-1, 9)).astype(np.float)
        #if int(pred_cls) < 2: pred_cls = '0'
        for box in pred_box:
            cls_id = 1 #int(box[0]) - 1  #skip background
            score = box[0]
            # if not (abs(box[8]) < 35 and abs(box[7]) < 35 and abs(box[6]) < 35): continue
            # if score < thred or box[5] < 0.5: continue
            pred_dict[image_path].append(box[1:5].tolist()+[cls_id, score])  #box+cls
    return pred_dict

def get_pred_results(pred_file, thred=0.1):
    if pred_file is None: return None
    global class_dict
    class_dict = {'-1':-1, '0':0}
    pred_dict = {}
    lines = open(pred_file, 'r').readlines()
    bbox_idx = 1
    for line in lines:
        split_item = line.strip().split()
        if len(split_item) < 4:
            # pred_dict[image_path].append([20,20,50,50,-1,0])
            continue
        # print(split_item)
        image_path = split_item[0].split('-_-')[-1] if '-_-' in split_item[0] else split_item[0].split('/')[-1]
        # image_path = os.path.join("zunyi_guizhou_badcase", split_item[0].split('-_-')[-3], image_path)
        #image key first occur
        if image_path not in pred_dict.keys():
            pred_dict[image_path] = list()
        
        score = float(split_item[bbox_idx])
        if score < thred: continue
        if float(split_item[bbox_idx+5]) < 0.5: continue

        box = list(map(float, split_item[bbox_idx+1:bbox_idx+5]))
        # print('s', score, box)
        box.append(0.)
        box.append(float(split_item[bbox_idx+5]))
        pred_dict[image_path].append(box) #list(map(int, box))
    return pred_dict


def show_results(image, targets=None, preds=None, resize_hw=None):
    """plot gt and pred bbox on input image.The image could be resized, 
        and gt will be accordingly resized
    Args:
        image: image
        targets (tensor): gt .  the outer axis indicates different gt,
                         inner axis includes xyxy label
        preds (tensor): predictions.  the outer axis indicates different pred box, 
                        inner axis includes xyxy label score
    """
    h, w, c = image.shape
    scale = 1
    if not resize_hw:
        scale = min(640. / max(h, w), 1)
        # scale = 0.5
        resize_hw = [h*scale, w*scale]
    
    try:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60] #default 95
        result, encimg = cv2.imencode('.jpg', image, encode_param)
        if result:
            image = cv2.imdecode(encimg, 1).astype(image.dtype)
    except: print('encode img failed and show src img ...')
    
    image = cv2.resize(image, (int(resize_hw[1]), int(resize_hw[0])))
    cv2.putText(image,'{:.3f}'.format(scale), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    if not (targets.shape[0] == 0 or targets is None):
        gt = targets[:, :4]
        label = targets[:, 4]
        ###Plot the boxes
        cls = len(class_dict)
        for i in range(len(gt)):
            xmin = int(round(gt[i][0]) / w * resize_hw[1])
            ymin = int(round(gt[i][1]) / h * resize_hw[0])
            xmax = int(round(gt[i][2]) / w * resize_hw[1])
            ymax = int(round(gt[i][3]) / h * resize_hw[0])
            
            coords = (xmin, ymin), xmax-xmin, ymax-ymin
            color = (0, 255, 0) #colors[label[i]]
            # print(label[i], 'label', class_dict)
            display_txt = 'gt_{}'.format(int(label[i]))
            # cv2.rectangle(image, (xmin-5, ymin-30), (xmin+190, ymin), (46, 184, 255), thickness=-1)#xmin＋115#xmax+5
            cv2.putText(image,'{}'.format(display_txt), (xmin-10, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness=1)

    if not (preds.shape[0] == 0 or preds is None):
        gt = preds[:, :4]
        label = preds[:, 4]
        score = preds[:, 5]
        ###Plot the boxes
        colors = [(0,0,255),(255,0,0),(0,255,255),(128,0,128),(128,128,0),(255,165,0),(192,14,235),]
        for i in range(len(gt)):
            xmin = int(round(gt[i][0]) / w * resize_hw[1])
            ymin = int(round(gt[i][1]) / h * resize_hw[0])
            xmax = int(round(gt[i][2]) / w * resize_hw[1])
            ymax = int(round(gt[i][3]) / h * resize_hw[0])
            
            coords = (xmin, ymin), xmax-xmin, ymax-ymin
            color = colors[int(label[i])]
            # print(label[i], 'label', class_dict)
            display_txt = [key for (key, value) in class_dict.items() if value == label[i]][0].replace('_', '')+'_{:.3f}'.format(score[i])
            # cv2.rectangle(image, (xmin-5, ymin-30), (xmin+190, ymin), (46, 184, 255), thickness=-1)#xmin＋115#xmax+5
            cv2.putText(image,'{}'.format(display_txt), (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (248,21,196), 1)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness=1)
    return image

def find_maxiou(BBGT, bb):
    ixmin = np.maximum(BBGT[:, 0], bb[0])
    iymin = np.maximum(BBGT[:, 1], bb[1])
    ixmax = np.minimum(BBGT[:, 2], bb[2])
    iymax = np.minimum(BBGT[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin, 0.)
    ih = np.maximum(iymax - iymin, 0.)
    inters = iw * ih
    uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
            (BBGT[:, 2] - BBGT[:, 0]) *
            (BBGT[:, 3] - BBGT[:, 1]) - inters)
    overlaps = inters / uni
    ovmax = np.max(overlaps)
    jmax = np.argmax(overlaps)
    return ovmax, int(jmax)


def crop_imgs(image, gt, save_path, img_path, pred=None):
    save_list = []
    scale = 0.2
    # h, w, c = image.shape

    # img_name = os.path.basename(img_path)
    split_items = img_path.split('/')
    img_name = split_items[-1]
    # img_name = '_'.join([device_id]+split_items[-2:])
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    # if pred is None: return 'no pred'
    str_w = ''

    det = np.array(pred) #pred
    R = [-1] * len(det)
    for roi_idx, box in enumerate(pred):
        xmin, ymin, xmax, ymax, label, _ = box
        h_, w_ = ymax - ymin, xmax - xmin
        # if max(h_, w_) < 80: continue

        ovmax, jmax = find_maxiou(det, box[:4])
        if ovmax > 0 and R[jmax] == 1: return 'pred bug'
        if ovmax > 0.3:
            R[jmax] = 1
            if str_w != '': str_w += ' '
            str_w += '{:.3f} {:.3f} {:.3f} {:.3f}'.format(det[jmax][0], det[jmax][1], det[jmax][2], det[jmax][3])
        else: return 'pred bug2'

    if str_w == '': return 'gt bug'
    return str_w

        # scale = 1
        # xmin1 = max(0, xmin - w_ * scale)
        # ymin1 = max(0, ymin - h_ * scale)
        # xmax1 = min(w, xmax + w_ * scale)
        # ymax1 = min(h, ymax + h_ * scale)
        # roi = image[int(ymin1):int(ymax1), int(xmin1):int(xmax1)]
        # if min(roi.shape) == 0: continue
        # # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # # roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        # coor_str = ' {:.3f} {:.3f} {:.3f} {:.3f}'.format(xmin-xmin1, ymin-ymin1, xmax-xmin1, ymax-ymin1)
        # # new_name = '%05d_%d_%s_%s' % (idx, roi_idx, split_item[0], split_item[-1])
        # # shutil.copy(os.path.join(data_root, img_path), os.path.join(save_path, img_name))
        # new_name = str(roi_idx) + '_' + img_name
        # cv2.imwrite(save_path + '/' + new_name, roi, encode_param)
        # save_list.append(new_name + coor_str + '\n')
    
    '''
    get max box
    '''
    # max_box = []
    # max_score = 0
    # if len(gt) == 0: cnt[0] += 1
    # for roi_idx, box in enumerate(gt):
    #     xmin, ymin, xmax, ymax, label, score = box
    #     if max_score < score:
    #         max_box = [str(xmin), str(ymin), str(xmax), str(ymax)]
    #         max_score = score
    # if max_score > 0.3:
    #     save_list.append(img_path+' '+' '.join(max_box)+'\n')
    # else:
    #     print(img_path, max_box, max_score)
    #     cnt[1] += 1

    return save_list


def show_img_gt(img_list_file, pred_file=None, index=0, resize_hw=None, key=None, vis=False, crop=False, shuffle=False):
    #load data list
    img_list_lines = open(img_list_file).readlines()
    pred_dict = None
    if pred_file and 'oneline' in pred_file:
        pred_dict = get_onelinepred_results(pred_file, thred=0.15)
    elif pred_file:
        pred_dict = get_pred_results(pred_file, thred=0.3) #
    if shuffle:
        import random
        random.shuffle(img_list_lines)
    if vis or crop:
        resize_hw = None
        save_name = os.path.splitext(os.path.basename(img_list_file))[0] + '.coor'
        save_path = os.path.splitext(os.path.basename(img_list_file))[0] + '_viscrop'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
    total_imgs = len(img_list_lines)
    idx = index
    save_list = []
    cache_idx = []
    while True:
        line = img_list_lines[idx]
        split_items = line.strip().split()
        if len(split_items) == 2 and split_items[1][-4:] == '.xml':
            img_path, gt_xml = split_items
        else:
            img_path, gt_xml = split_items[0], None
        if key:
            idx += 1
            # print(idx, img_path)
            if key in img_path: key = None
            if idx == total_imgs: break
            if key: continue

        image = cv2.imread('{}/{}'.format(data_root, img_path))
        if image is None:
            print('Unable read image:', '{}/{}'.format(data_root, img_path))
            idx += 1
            continue
        # image = None

        gt, pred = np.array([]), np.array([])

        if gt_xml is not None:
            gt_xml = '{}/{}'.format(data_root, gt_xml)
            gt, hw = get_gt(gt_xml, anno_key=None) #, 'nohelmet', ['all_cover', 'part_cover', 'lp']
        
        if pred_dict is not None:
            img_key = img_path #img_path.split('/')[-1] #
            # print(img_key)
            if img_key in pred_dict.keys():
                pred = np.array(pred_dict[img_key])
        
        img_name = img_path #img_path.split('/')[-1]
        if crop:
            idx += 1
            if idx % 5000 == 0: print('crop imgs:', idx)
            if idx == total_imgs: break

            # print(gt, pred)
            # if len(gt) == 0 or len(pred)==0: continue
            # if len(gt) != 1: continue
            # if max(gt[0][2] - gt[0][0], gt[0][3] - gt[0][1]) < 240: continue

            str_w = crop_imgs(image, gt, save_path, img_path, pred)
            save_list.append(img_name+' '+str_w+'\n')
            # if idx > 500: break
            continue

        h, w, c = image.shape
        img = show_results(image, gt, pred, resize_hw=resize_hw)

        if vis:
            # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
            cv2.imwrite(os.path.join(save_path, img_name), img)
            idx += 1
            # if idx > 20: break
            if idx % 5000 == 0: print('vis imgs:', idx)
            if idx == total_imgs: break
            continue
        
        #default is show results
        cv2.imshow('{}_{}'.format(idx, img_name), img)
        print('Space to next, p to preview, e to exit: ')
        str_ = chr(cv2.waitKey(0)).lower()
        while str_ not in [' ', 'p', 'e', 's']:
            print('Space to next, p to preview, e to exit: ')
            str_ = chr(cv2.waitKey(0)).lower()
        cv2.destroyWindow('{}_{}'.format(idx, img_name))
        if str_ == ' ':
            cache_idx.append(idx)
            idx += 1
        if str_ == 'p':
            idx = cache_idx.pop() if len(cache_idx) > 0 else 0
        if str_ == 's': #bug  bug
            save_list.append(img_path + ' ' + gt_xml + '\n')
            idx += 1
        if str_ == 'e':
            break
        idx %= total_imgs
    if vis or crop:
        save_list2txt(save_list, save_name)


def show_gt_distribute(img_list_file, resize_hw=None):
    #load data list
    lines = open(img_list_file).readlines()

    save_list = []
    box_size = dict()
    for idx, line in enumerate(lines):
        img_path, gt_xml = line.strip().split()
        img_name = img_path.split('/')[-1]

        gt_xml = '{}/{}'.format(data_root, gt_xml)
        gt, hw = get_gt(gt_xml, resize_hw=resize_hw)
        
        if len(gt) > 0:
            wh = gt[:, 2:4] - gt[:, :2]
            if min(np.min(wh, axis=0)) > 30:
                save_list.append(line.strip() + "\n")
    # save_list2txt(save_list, 'tmp30.txt')
        #print(hw)
        # if hw is not []:
        #     d = {x:hw.count(x) for x in hw}
            
        #     for k in d.keys():
        #         box_size[k] = box_size.get(k,0) + d[k]
        
    # lists = sorted(box_size.items()) # sorted by key, return a list of tuples
    # x, y = zip(*lists) # unpack a list of pairs into two tuples
    # plt.plot(x, y)
    # plt.show()


if __name__ == '__main__':
    img_list_file = '{}/{}'.format(data_path, file_list)
    pred_file = None #"save2txt_xinjiang_c.lst" #
    key = None #"139607" #"136548" #'137521.jpg'  #bigface: 137715
    show_img_gt(img_list_file, pred_file=pred_file, index=0, resize_hw=None, key=key, vis=False, crop=False, shuffle=True)
    # show_gt_distribute(img_list_file, resize_hw=None)
