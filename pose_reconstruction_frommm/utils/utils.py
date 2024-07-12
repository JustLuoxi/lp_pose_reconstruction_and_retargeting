# Author: Xi Luo
# Email: sunshine.just@outlook.com
# H2tc data processing utils

import pandas as pd
import numpy as np
import json
from addict import Dict
from scipy.spatial.transform import Rotation
import os.path as osp

def read_csv(filepath):
    """
    Read a csv file
    INPUT:
        filepath: file path of the csv file to be read
    OUTPUT:
        df: pandas dataframe of the csv data
    """

    df = pd.read_csv(filepath, delimiter=" ")
    return df


def load_timestamp(path):
    """
    Load timestamp file

    Returns:
        list: of int timestamps

    Args:
        path (string): path to the timestamp file

    """

    with open(path, "r") as f:
        # ignore the first row since it is the header
        return [int(ts) for ts in f.readlines()[1:]]

def load_openpose_timestamp(cvs_path, roi_name, aligned, s_frame, e_frame):
    opti = {}
    with open(cvs_path, "r") as f:
        opti[roi_name] = {}  # new sub dict for a object ID
        for line in f.readlines()[1:]:  # iterate every row from 2nd
            line = line.split(",")  # a list of separated data
            # timestamp as key, the rest data as value
            tmp = [float(x) for x in line[1:]]
            opti[roi_name][int(line[0])] = tmp

    # extract the frames' info.
    frame_num = e_frame - s_frame + 1
    data = np.zeros(shape=(frame_num, dim_feat))  # position only
    dim_f = data.shape[1]
    for i in range(s_frame, e_frame + 1, 1):
        id = str(i)
        try:
            tmp = opti[roi_name][aligned[id][roi_name]]
            data[i - s_frame] = tmp[:dim_f]
            # data = data / np.array([2.0, 2.0, 2.0])
        except:
            print("%s get error in frame %d" % (roi_name, i))
            continue
    return data

def load_optic_timestamp(cvs_path, roi_name, aligned, s_frame, e_frame, dim_feat=3):
    opti = {}
    with open(cvs_path, "r") as f:
        opti[roi_name] = {}  # new sub dict for a object ID
        for line in f.readlines()[1:]:  # iterate every row from 2nd
            line = line.split(",")  # a list of separated data
            # timestamp as key, the rest data as value
            tmp = [float(x) for x in line[1:]]
            opti[roi_name][int(line[0])] = tmp

    # extract the frames' info.
    frame_num = e_frame - s_frame + 1
    data = np.zeros(shape=(frame_num, dim_feat))  # position only
    dim_f = data.shape[1]
    for i in range(s_frame, e_frame + 1, 1):
        id = str(i)
        try:
            tmp = opti[roi_name][aligned[id][roi_name]]
            data[i - s_frame] = tmp[:dim_f]
            # data = data / np.array([2.0, 2.0, 2.0])
        except:
            print("%s get error in frame %d" % (roi_name, i))
            continue
    return data


def load_handpose_timestamp(cvs_path, roi_name, aligned, s_frame, e_frame):
    opti = {}
    with open(cvs_path, "r") as f:
        opti[roi_name] = {}  # new sub dict for a object ID
        for line in f.readlines()[1:]:  # iterate every row from 2nd
            line = line.split(",")  # a list of separated data
            # timestamp as key, the rest data as value
            tmp = [float(x) for x in line[1:]]
            opti[roi_name][int(line[0])] = tmp

    # extract the frames' info.
    frame_num = e_frame - s_frame + 1
    data = np.zeros(shape=(frame_num, 60))  # [seq_len, 60]
    dim_f = data.shape[1]
    for i in range(s_frame, e_frame + 1, 1):
        id = str(i)
        try:
            tmp = opti[roi_name][aligned[id][roi_name]]
            tmp = np.radians(tmp[:dim_f])
            data[i - s_frame] = tmp  # degree to radians
        except:
            print("%s get error in frame %d" % (roi_name, i))
            continue
    return data

def loadseqseg(data_folder, take_id, stage="throw"):
    
    processed_folder = os.path.join(data_folder,take_id,"processed")
    # load alignment file
    align_path = os.path.join(processed_folder, "alignment.json")
    with open(align_path, "r") as f:
        aligned = json.loads(f.read())

    # load annotation file 
    anno_file = os.path.join(data_folder, take_id, "%s.json" % take_id)
    with open(anno_file, "r") as f:
        jsons = json.loads(f.read())
        anno = Dict(jsons)
    action = anno[stage]
    
    # get the start and end frame
    s_frame = 0
    e_frame = 0
    e_frame = anno[stage].time_point["sub1_head_motion"]["frame"]
    for i in range(e_frame - s_frame + 1):
        ts = aligned[str(i)]["right_hand_pose"]
        ts_L = aligned[str(i)]["left_hand_pose"]
        if ts != None and ts_L!=None:
            s_frame = i
            break
    return s_frame, e_frame, anno, aligned

def loadhands(data_folder, take_id):
    processed_folder = os.path.join(data_folder,take_id,"processed")
    
    # load alignment file
    align_path = os.path.join(processed_folder, "alignment.json")
    with open(align_path, "r") as f:
        aligned = json.loads(f.read())

    # load annotation file 
    anno_file = os.path.join(data_folder, take_id, "%s.json" % take_id)
    with open(anno_file, "r") as f:
        jsons = json.loads(f.read())
        anno = Dict(jsons)
    action = anno["throw"]
    
    # get the start and end frame
    s_frame = 0
    e_frame = 0
    e_frame = anno["throw"].time_point["sub1_head_motion"]["frame"]
    for i in range(e_frame - s_frame + 1):
        ts = aligned[str(i)]["right_hand_pose"]
        ts_L = aligned[str(i)]["left_hand_pose"]
        if ts != None and ts_L!=None:
            s_frame = i
            break
    
    ## load hands poses
    file = os.path.join(processed_folder, "right_hand_pose.csv")
    file_L = os.path.join(processed_folder, "left_hand_pose.csv")
    # dim: [seq_len, 60]
    rhand_pose = load_handpose_timestamp(file, "right_hand_pose", aligned, s_frame, e_frame)
    lhand_pose = load_handpose_timestamp(file_L, "left_hand_pose", aligned, s_frame, e_frame)
    
    ## load right hand motion
    _HANDS = "sub1_right_hand_motion"
    rhand_file = os.path.join(processed_folder, _HANDS + ".csv")
    rhand_RT = load_optic_timestamp(rhand_file, _HANDS, aligned, s_frame, e_frame, dim_feat=7)
    rhand_T = rhand_RT[:,:3]/2.0 # normalize to [0,1]
    # rhand_R = Rotation.from_quat(rhand_RT[:,3:]).as_euler('xyz', degrees=True)
    rhand_R = rhand_RT[:,3:]
    
    ## load left hand motion
    _HANDS = "sub1_left_hand_motion"
    lhand_file = os.path.join(processed_folder, _HANDS + ".csv")
    lhand_RT = load_optic_timestamp(lhand_file, _HANDS, aligned, s_frame, e_frame, dim_feat=7)
    lhand_T = lhand_RT[:,:3]/2.0 # normalize to [0,1]
    # lhand_R = Rotation.from_quat(lhand_RT[:,3:]).as_euler('xyz', degrees=True)
    lhand_R = lhand_RT[:,3:]
    
    # [seq_len, 3+4+3+4]
    hands_RT = np.concatenate((rhand_T, rhand_R,lhand_T, lhand_R), axis=1)
    # [ seq_len, 60+60 ]
    hands_pose = np.concatenate((rhand_pose, lhand_pose), axis=1)
    return hands_RT, hands_pose, action.hand, anno



def extract_opticdata(processed_folder,roi_name,s_frame,e_frame,dim_feat):
    align_path = os.path.join(processed_folder, "alignment.json")
    with open(align_path, "r") as f:
        aligned = json.loads(f.read())
    sub1_head_file = os.path.join(processed_folder, roi_name+".csv")
    data = load_optic_timestamp(
                sub1_head_file, roi_name, aligned, s_frame, e_frame, dim_feat
            )
    return data


from torch.optim import lr_scheduler
def get_scheduler(optimizer, policy, nepoch_fix=None, nepoch=None, decay_step=None):
    if policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - nepoch_fix) / float(nepoch - nepoch_fix + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=decay_step, gamma=0.1)
    elif policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', policy)
    return scheduler

import torch
def get_dct_matrix(N, is_torch=True,device=None,dtype=None):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    if is_torch:
        dct_m = torch.from_numpy(dct_m)
        idct_m = torch.from_numpy(idct_m)
        if dtype is not None:
            dct_m = dct_m.to(dtype=dtype)
            idct_m = idct_m.to(dtype=dtype)
        if device is not None:
            dct_m = dct_m.to(device)
            idct_m = idct_m.to(device)
    return dct_m, idct_m

# ==================== logger ============================================
import logging
import os
def create_logger(filename, file_handle=True):
    # create logger
    logger = logging.getLogger(filename)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(stream_formatter)
    logger.addHandler(ch)

    if file_handle:
        # create file handler which logs even debug messages
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fh = logging.FileHandler(filename, mode='a')
        fh.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('[%(asctime)s] %(message)s')
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

    return logger

def batch_to(dst, *args):
    return [x.to(dst) if x is not None else None for x in args]

# ================== PCA ======================================
# X -> X_PCA
def data2pca(data, components, mean):
    tmp = data - mean
    data_pca = np.dot(tmp, np.linalg.pinv(components))
    return data_pca

# X -> X_PCA
def data2pca_M(data, components, mean, M):
    components_M = components[:M]
    tmp = data - mean
    data_pca = np.dot(tmp, np.linalg.pinv(components_M))
    return data_pca

# PCA -> X
def pca2data(data_pca, components, mean):
    data_recover = np.matmul(data_pca, components) + mean
    return data_recover

# PCA -> X
def pca2data_M(data_pca, components, mean, M):
    components_M = components[:M]
    data_recover = np.matmul(data_pca, components_M) + mean
    return data_recover

# ==================================== data loading =============================
def load_align_anno(take_folder):
    
    # load alignment file
    processed_folder = os.path.join(take_folder,"processed")
    align_path = os.path.join(processed_folder, "alignment.json")
    with open(align_path, "r") as f:
        aligned = json.loads(f.read())

    # load annotation file 
    take_folder = take_folder.replace("\\","/")
    take_id = take_folder.split("/")[-1]
    anno_file = os.path.join(take_folder, "%s.json" % take_id)
    with open(anno_file, "r") as f:
        jsons = json.loads(f.read())
        anno = Dict(jsons)  
    return aligned, anno



    
