# -*- coding: utf-8 -*-
import math
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os.path as osp
from PIL import Image
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签  
plt.rcParams['font.family']='sans-serif' 
plt.rcParams['figure.figsize'] = (20.0, 20.0)
 
                
dataset=dict(
            ann_file=(
                    ('', 'has_people_phone_rand3w_train2.lst'),
                    ('', 'jianhang_0412_RMB_rand3w_train2.lst'), 
                    ('', 'money_phone_20210710_rand2w_2w_train.lst'),
                    ('','colloect_phone_money_20210708_train.lst'),
                    ),
            img_prefix='/mnt/datadisk0/jingzhudata/phone_money/',
            classes= ('phone', 'money')
            )
ch, cw = 576, 960

# 读取数据
class PlotRatio(object):
    
    def __init__(self, **kwargs):
        super(PlotRatio, self).__init__()
        self.img_prefix = dataset['img_prefix']
        
    def plot_ratio(self, ann_file, classes):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        # print(ann_file, "....debug")
        assert isinstance(ann_file, (list, tuple)), "ann_file must be list or tuple in DGVOCDataset"

        count_wh = [0]*10
        count_squares = [0]*11
        num_class = [0]*2

        for (year, name) in ann_file:
            rootpath = osp.join(self.img_prefix, year)
            for img_id, line in enumerate(open(osp.join(rootpath, name))):
                if ';' not in line:
                    split_item = line.strip().split()
                else:
                    split_item = line.strip().split(';')
                if len(split_item) != 2:
                    img_path = split_item[0]
                    xml_path = None
                else:
                    img_path, xml_path = split_item
                    if '.xml' != xml_path[-4:]: xml_path = None
                if xml_path is None: continue
                img_path_com = osp.join(rootpath, img_path)
                xml_path_com = osp.join(rootpath, xml_path)
                img = Image.open(img_path_com)
                width, height = img.size # 原图宽高, 标签有时有问题
                tree = ET.parse(xml_path_com)
                root = tree.getroot()
                # size = root.find('size')
                # width = size.find('width')
                # height = size.find('height')
                for obj in root.findall('object'):
                    name = obj.find('name').text.lower().strip()
                    # if 'fjs_' in name:
                    #     name = name.replace('fjs_', '')
                    if name not in classes:
                        continue
                    else :
                        idx = classes.index(name)
                        num_class[idx] += 1

                        bndbox = obj.find('bndbox')
                        xmin = bndbox.find('xmin').text
                        ymin = bndbox.find('ymin').text
                        xmax = bndbox.find('xmax').text
                        ymax = bndbox.find('ymax').text

                        #NOTE filter mislabeling gt
                        w_box = float(xmax) - float(xmin)
                        h_box = float(ymax) - float(ymin)
                        if w_box * h_box <= 0 or min(w_box, h_box) < 4 or max(w_box, h_box) < 4 or max(w_box, h_box) > 360:
                            continue

                        ratio2 = 1.
                        if height > ch or width > cw:
                            ratio2 = np.min(np.array([ch, cw]).astype(np.float64) / np.array([height, width]))
                        
                        w = (w_box) * ratio2
                        h = (h_box) * ratio2

                        if w==0 or h==0:
                            continue

                        ratio = round(w/h, 1)
                        scale = round(w*h, 1)
                        square = math.sqrt(scale)

                        if ratio < 0.25:
                            count_wh[0] += 1
                        elif 0.25 <= ratio < 1/3:
                            count_wh[1] += 1
                        elif 1/3 <= ratio < 1/2:
                            count_wh[2] += 1
                        elif 1/2 <= ratio < 1:
                            count_wh[3] += 1
                        elif 1 <= ratio < 1.5:
                            count_wh[4] += 1
                        elif 1.5 <= ratio < 2:
                            count_wh[5] += 1
                        elif 2 <= ratio < 2.5:
                            count_wh[6] += 1
                        elif 2.5 <= ratio < 3:
                            count_wh[7] += 1
                        elif 3 <= ratio < 4:
                            count_wh[8] += 1
                        else:
                            count_wh[9] += 1

                        if square < 8:
                            count_squares[0] += 1
                        elif 8 <= square < 16:
                            count_squares[1] += 1
                        elif 16 <= square < 21:
                            count_squares[2] += 1
                        elif 21 <= square < 32:
                            count_squares[3] += 1
                        elif 32 <= square < 64:
                            count_squares[4] += 1
                        elif 64 <= square < 128:
                            count_squares[5] += 1
                        elif 128 <= square < 256:
                            count_squares[6] += 1
                        elif 256 <= square < 512:
                            count_squares[7] += 1
                        elif 512 <= square < 1024:
                            count_squares[8] += 1
                        elif 1024 <= square < 2048:
                            count_squares[9] += 1
                        elif 2048 <= square < 4096:
                            count_squares[10] += 1

        # 绘图
        wh_df = pd.DataFrame(count_wh, index=['0-0.25','0.25-0.33','0.33-0.5','0.5-1','1-1.5','1.5-2','2-2.5',\
                                              '2.5-3','3-4', '>4'], columns=['宽高比'])
        wh_df.plot(kind='bar', color ='#55aacc')
        plt.savefig('./phone_wallet_ratios.jpg')
        #plt.savefig('./dms_ratios_face.jpg')

        squares_df = pd.DataFrame(count_squares, index=['0-8','8-16','16-21', '21-32','32-64','64-128',\
                                                    '128-256','256-512','512-1024','1024-2048','2048-4096'], columns=['边长范围'])
        squares_df.plot(kind='bar', color ='#55aacc')
        plt.savefig('./phone_wallet_squares.jpg')
        #plt.savefig('./dms_squares_face.jpg')

        num_class_df = pd.DataFrame(num_class,index=['phone', 'money'], columns=['类别数'])
        num_class_df.plot(kind='bar')
        plt.savefig('./rmp.jpg')
        

pr = PlotRatio()
pr.plot_ratio(ann_file = dataset['ann_file'], classes=dataset['classes'])
#pr.plot_ratio(ann_file = dataset['ann_file'], classes=dataset['classes'][3])