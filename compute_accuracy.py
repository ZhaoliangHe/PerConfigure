<<<<<<< HEAD
import io
import os
import scipy.misc
import numpy as np
import six
import time
import pickle
from six import BytesIO

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils

def compute_iou(box1, box2, X, Y):
    '''
    ymin, xmin, ymax, xmax = box
    把纵坐标当成X，横坐标当成Y，懒得改了...
    '''
    Xmin1, Ymin1, Xmax1, Ymax1 = box1
    Xmin2, Ymin2, Xmax2, Ymax2 = box2
    Xmin1 = X*Xmin1; Xmax1 = X*Xmax1; Xmin2 = X*Xmin2; Xmax2 = X*Xmax2
    Ymin1 = Y*Ymin1; Ymax1 = Y*Ymax1; Ymin2 = Y*Ymin2; Ymax2 = Y*Ymax2
    # 获得相交区域的左上角坐标和右下角坐标 min = max(min) max = min(max)
    inter_Xmin = max(Xmin1, Xmin2)
    inter_Ymin = max(Ymin1, Ymin2)
    inter_Xmax = min(Xmax1, Xmax2)
    inter_Ymax = min(Ymax1, Ymax2)

    # 以免不相交
    W = max(0, inter_Xmax - inter_Xmin)
    H = max(0, inter_Ymax - inter_Ymin)

    # 计算相交区域面积
    inter_area = W * H

    # 计算并集面积
    merge_area = (Xmax1 - Xmin1) * (Ymax1 - Ymin1) + (Xmax2 - Xmin2) * (Ymax2 - Ymin2)

    # 计算IOU
    IOU = inter_area / (merge_area - inter_area + 1e-6)

    return IOU

def compute_box_f1(test_result,real_result):
    test_box_number = test_result['boxes'].shape[0]
    real_box_number = real_result['boxes'].shape[0]
    true_positive = 0
    for i in range(test_box_number):
        max_iou = 0;index_iou = 0
        box = test_result['boxes'][i]
        for j in range(real_box_number):
            iou = compute_iou(box,real_result['boxes'][j],1024,1024)
            if(iou>max_iou):
                max_iou = iou
                index_iou = j
        if (test_result['classes'][i] == real_result['classes'][index_iou]):
            true_positive = true_positive + 1
        elif ((test_result['classes'][i] + real_result['classes'][index_iou]) == 11 ):
            true_positive = true_positive + 1
    acc = true_positive/test_box_number
    rec = true_positive/real_box_number
    if rec>1:
        rec =1
    f1 = (2*acc*rec) / (acc+rec)
    return f1

def namestr(obj, namespace):
 return [name for name in namespace if namespace[name] is obj]

def compute_model_f1(test_model_result,ground_truths):
    f1s = []
    for i in range(len(ground_truths)):
        f1 = compute_box_f1(test_model_result[i],ground_truths[i])
        f1s.append(f1)
    print(namestr(test_model_result, globals())[0],'avg f1',np.mean(f1s))
    # return np.mean(f1s)
    return f1s

my_path = "/home/hezhaoliang/" #instead of yours

result_path = my_path+"PerConfigure/results/"+"ground_truths"#"detections_results"

with open(result_path+'.defaultdict', 'rb') as f:
    ground_truths = pickle.load(f)
with open(my_path+"PerConfigure/results/rcnn_in_1024.defaultdict", 'rb') as f:
    rcnn_in_1024 = pickle.load(f)
with open(my_path+"PerConfigure/results/rcnn_in_640.defaultdict", 'rb') as f:
    rcnn_in_640 = pickle.load(f)
with open(my_path+"PerConfigure/results/rcnn_res_1024.defaultdict", 'rb') as f:
    rcnn_res_1024 = pickle.load(f)
with open(my_path+"PerConfigure/results/rcnn_res_640.defaultdict", 'rb') as f:
    rcnn_res_640 = pickle.load(f)
with open(my_path+"PerConfigure/results/ssd_res_1024.defaultdict", 'rb') as f:
    ssd_res_1024 = pickle.load(f)
with open(my_path+"PerConfigure/results/ssd_res_640.defaultdict", 'rb') as f:
    ssd_res_640 = pickle.load(f)
with open(my_path+"PerConfigure/results/ssd_mo_640.defaultdict", 'rb') as f:
    ssd_mo_640 = pickle.load(f)
# with open(my_path+"PerConfigure/results/ssd_mo_320.defaultdict", 'rb') as f:
#     ssd_mo_320 = pickle.load(f)

rcnn_res_1024_f1 = compute_model_f1(rcnn_in_1024,ground_truths)
rcnn_in_640_f1 = compute_model_f1(rcnn_in_640,ground_truths)
rcnn_res_1024_f1 = compute_model_f1(rcnn_res_1024,ground_truths)
rcnn_res_640_f1 = compute_model_f1(rcnn_res_640,ground_truths)
ssd_res_1024_f1 = compute_model_f1(ssd_res_1024,ground_truths)
ssd_res_640_f1 = compute_model_f1(ssd_res_640,ground_truths)
ssd_mo_640_f1 = compute_model_f1(ssd_mo_640,ground_truths)
# ssd_mo_320_f1 = compute_model_f1(ssd_mo_320,ground_truths)

=======
import io
import os
import scipy.misc
import numpy as np
import six
import time
import pickle
import config as conf
import ffmpeg_code as fc
from six import BytesIO

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import ffmpeg_code

def compute_iou(box1, box2, X, Y):
    '''
    ymin, xmin, ymax, xmax = box
    把纵坐标当成X，横坐标当成Y，懒得改了...
    '''
    Xmin1, Ymin1, Xmax1, Ymax1 = box1
    Xmin2, Ymin2, Xmax2, Ymax2 = box2
    Xmin1 = X*Xmin1; Xmax1 = X*Xmax1; Xmin2 = X*Xmin2; Xmax2 = X*Xmax2
    Ymin1 = Y*Ymin1; Ymax1 = Y*Ymax1; Ymin2 = Y*Ymin2; Ymax2 = Y*Ymax2
    # 获得相交区域的左上角坐标和右下角坐标 min = max(min) max = min(max)
    inter_Xmin = max(Xmin1, Xmin2)
    inter_Ymin = max(Ymin1, Ymin2)
    inter_Xmax = min(Xmax1, Xmax2)
    inter_Ymax = min(Ymax1, Ymax2)

    # 以免不相交
    W = max(0, inter_Xmax - inter_Xmin)
    H = max(0, inter_Ymax - inter_Ymin)

    # 计算相交区域面积
    inter_area = W * H

    # 计算并集面积
    merge_area = (Xmax1 - Xmin1) * (Ymax1 - Ymin1) + (Xmax2 - Xmin2) * (Ymax2 - Ymin2)

    # 计算IOU
    IOU = inter_area / (merge_area - inter_area + 1e-6)

    return IOU


def compute_box_f1(test_result,real_result):
    test_box_number = test_result['boxes'].shape[0]
    real_box_number = real_result['boxes'].shape[0]
    true_positive = 0
    for i in range(test_box_number):
        max_iou = 0;index_iou = 0
        box = test_result['boxes'][i]
        for j in range(real_box_number):
            iou = compute_iou(box,real_result['boxes'][j],1024,1024)
            if(iou>max_iou):
                max_iou = iou
                index_iou = j
        if (test_result['classes'][i] == real_result['classes'][index_iou]):
            true_positive = true_positive + 1
        elif ((test_result['classes'][i] + real_result['classes'][index_iou]) == 11 ):
            true_positive = true_positive + 1
    acc = true_positive/test_box_number
    rec = true_positive/real_box_number
    if rec>1:
        rec =1
    f1 = (2*acc*rec) / (acc+rec)
    return f1


def namestr(obj):
    ns = globals()
    return [name for name in ns if ns[name] is obj]


def get_r_frames(t_fps):
    # l = model_key.split("_")
    # last_i = len(l) - 1
    # l.append(l[last_i] + "x" + l[last_i])
    # l[last_i] = str(t_fps)
    # k = "_".join(l)
    # nr_frames = len(os.listdir(video_path)) - 1
    # video_path = conf.path['user_path'] + conf.path['project_path'] + conf.path['dataset_path'] + "_" + k
    video_path = conf.path['user_path'] + conf.path['project_path'] + conf.path['dataset_path']
    video_info = fc.get_video_info(video_path)
    r_frames = fc.cal_r_frames(int(t_fps), video_info['fps'], video_info['frames'])
    return r_frames


def preprocess_test_result(test_model_result, t_fps, nr_ground_truths):
    r_frames = get_r_frames(t_fps)
    j = 0; ret = []
    next_frame = r_frames[j] - 1
    if len(r_frames) + len(test_model_result) - 1 == nr_ground_truths:
        r_frames.pop()
    for i in range(nr_ground_truths):
        if i == next_frame:
            if i == 0:
                ret.append(test_model_result[i])
            else:
                ret.append(ret[i-1])
            j += 1
            if j != len(r_frames):
                next_frame = r_frames[j] - 1
            continue

        ret.append(test_model_result[i-j])
    return ret


def compute_model_f1(test_model_result, ground_truths, t_fps, model_key):
    f1s = []
    nr_ground_truths = len(ground_truths)
    if len(test_model_result) != nr_ground_truths:
        test_model_result = preprocess_test_result(test_model_result, t_fps, nr_ground_truths)

    for i in range(nr_ground_truths):
        f1 = compute_box_f1(test_model_result[i],ground_truths[i])
        f1s.append(f1)
    print(model_key + "_" + str(t_fps),'avg f1',np.mean(f1s))
    # return np.mean(f1s)
    return f1s


def main():
    result_path = conf.path['user_path'] + conf.path['project_path'] + "results/"
    ground_truths_path = result_path + "ground_truths"
    fps_list = conf.fps
    for i in range(len(fps_list)):
        t_fps = str(fps_list[i])
        with open(ground_truths_path + '.defaultdict', 'rb') as f:
            ground_truths = pickle.load(f)
        with open(result_path + "rcnn_in_1024" + "_" + t_fps + ".defaultdict", 'rb') as f:
            rcnn_in_1024 = pickle.load(f)
        with open(result_path + "rcnn_in_640" + "_" + t_fps + ".defaultdict", 'rb') as f:
            rcnn_in_640 = pickle.load(f)
        with open(result_path + "rcnn_res_1024" + "_" + t_fps + ".defaultdict", 'rb') as f:
            rcnn_res_1024 = pickle.load(f)
        with open(result_path + "rcnn_res_640" + "_" + t_fps + ".defaultdict", 'rb') as f:
            rcnn_res_640 = pickle.load(f)
        with open(result_path + "ssd_res_1024" + "_" + t_fps + ".defaultdict", 'rb') as f:
            ssd_res_1024 = pickle.load(f)
        with open(result_path + "ssd_res_640" + "_" + t_fps + ".defaultdict", 'rb') as f:
            ssd_res_640 = pickle.load(f)
        with open(result_path + "ssd_mo_640" + "_" + t_fps + ".defaultdict", 'rb') as f:
            ssd_mo_640 = pickle.load(f)
        # with open(my_path+"PerConfigure/results/ssd_mo_320.defaultdict", 'rb') as f:
        #     ssd_mo_320 = pickle.load(f)

        rcnn_res_1024_f1 = compute_model_f1(rcnn_in_1024, ground_truths, t_fps, "rcnn_in_1024")
        rcnn_in_640_f1 = compute_model_f1(rcnn_in_640, ground_truths, t_fps, "rcnn_in_640")
        rcnn_res_1024_f1 = compute_model_f1(rcnn_res_1024, ground_truths, t_fps, "rcnn_res_1024")
        rcnn_res_640_f1 = compute_model_f1(rcnn_res_640, ground_truths, t_fps, "rcnn_res_640")
        ssd_res_1024_f1 = compute_model_f1(ssd_res_1024, ground_truths, t_fps, "ssd_res_1024")
        ssd_res_640_f1 = compute_model_f1(ssd_res_640, ground_truths, t_fps, "ssd_res_640")
        ssd_mo_640_f1 = compute_model_f1(ssd_mo_640, ground_truths, t_fps, "ssd_mo_640")
        # ssd_mo_320_f1 = compute_model_f1(ssd_mo_320,ground_truths)

main()
>>>>>>> chentang
