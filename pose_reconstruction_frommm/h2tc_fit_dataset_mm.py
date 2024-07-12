# Author: Xi Luo
# Email: sunshine.just@outlook.com
# H2tc dataset-related functions for multi-modal based human pose optimization

import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import os.path as osp
import json

from torch.utils.data import Dataset
from pose_fitting.fitting_utils import matrix_to_axis_angle

import numpy as np
import torch
import cv2

from addict import Dict
from pathlib import Path
from scipy.spatial.transform import Rotation

DEFAULT_GROUND = [0.0, 1.0, 0.0, -0.5] # ground plane, y-axis is up

# ============== h2tc pose 2 mano pose =======
# data_RH: [seq_len, 20,3]
def pose2manopose(data_RH, is_rhand=True):
    # map index
    mano_idx = [
        # 0,  # wrist
        2,3,4,  # index
        6,7,8,  # middle
        10,11,12,  # pinky
        14,15,16,  # ring
        17,18,19,  # thumb
    ]
    data_RH_mano = data_RH[:, mano_idx, :]
    N_len = data_RH.shape[0]
    data_RH_mano = data_RH_mano.reshape(N_len*15, 3)
    if is_rhand==False:
        data_RH_mano[:,2] = -data_RH_mano[:,2]
        data_RH_mano[:,1] = -data_RH_mano[:,1]

    # from rotation angles to axis angle(mano's manner)
    data_RH_mano_rot = Rotation.from_euler("xyz", data_RH_mano, degrees=False).as_matrix()
    data_RH_mano_rot = torch.from_numpy(data_RH_mano_rot).view(N_len,15,3,3)
    data_RH_mano_axisangle = matrix_to_axis_angle(data_RH_mano_rot).view(N_len,45)

    return data_RH_mano_axisangle

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

def loadhandspose(data_folder, take_id, s_frame=0, e_frame=None):
    processed_folder = os.path.join(data_folder,take_id,"processed")
    
    # load alignment file
    align_path = os.path.join(processed_folder, "alignment.json")
    with open(align_path, "r") as f:
        aligned = json.loads(f.read())

    # get the start and end frame
    if e_frame == None:
        # load annotation file 
        anno_file = os.path.join(data_folder, take_id, "%s.json" % take_id)
        with open(anno_file, "r") as f:
            jsons = json.loads(f.read())
            anno = Dict(jsons)
        action = anno["throw"]
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
    r_mano_pose = pose2manopose(rhand_pose.reshape(-1,20,3))
    l_mano_pose = pose2manopose(lhand_pose.reshape(-1,20,3), is_rhand=False)
    
    return l_mano_pose, r_mano_pose

class H2TCFitDataset(Dataset):

    def __init__(self, joints2d_path,
                       cam_mat,
                       seq_len=None,
                       img_path=None,
                       load_img=False,
                       video_name='rgb_video',
                       args = None
                 ):
        '''
        Creates a dataset based on a single RGB video.

        - joints2d_path : path to saved OpenPose keypoints for the video
        - cam_mat : 3x3 camera intrinsics
        - seq_len : If not none, the maximum number of frames in a subsequence, will split the video into subsequences based on this. If none, the dataset contains a single sequence of the whole video.
        - img_path : path to directory of video frames
        - load_img : if True, will load and return the video frames as part of the data.
        '''
        super(H2TCFitDataset, self).__init__()

        self.joints2d_path = joints2d_path
        self.cam_mat = cam_mat
        self.seq_len = seq_len
        self.img_path = img_path
        self.load_img = load_img
        self.video_name = video_name
        self.load_zedcam()
        self.args = args
        self.DEFAULT_FPS = 60

        # catch or throw
        data_folder =osp.dirname( osp.dirname( osp.dirname(self.img_path)))
        take_id = self.img_path.split('/')[-3]
        anno_file = os.path.join(data_folder, take_id, "%s.json" % take_id) # load annotation file 
        with open(anno_file, "r") as f:
            jsons = json.loads(f.read())
            anno = Dict(jsons)
        self.args.catch_throw = anno['sub1_cmd']['action']

        # load data  
        self.data_dict, self.seq_intervals = self.load_data()
        self.data_len = len(self.data_dict['img_paths'])
        print('RGB dataset contains %d sub-sequences...' % (self.data_len))

    def load_zedcam(self):
        # zed camera intrisic paramter
        # K = np.array([[693.91839599609375, 0.0, 665.73150634765625],
        #                             [0.0, 693.91839599609375, 376.775787353515625],
        #                             [0.0, 0.0, 1.0]])

        # the camera intrinsics of the 'rgbd0' images
        # access it from http://calib.stereolabs.com/?SN=17471  [LEFT_CAM_HD]
        K = np.array([[699.78, 0.0, 660.19],
                                    [0.0, 699.78, 365.3615],
                                    [0.0, 0.0, 1.0]])
        self.cam_mat = K

    def check_person_id(self,mm_para):
        # person boundingbox 
        mm_bbx = mm_para['bboxes_xyxy']
        mm_person_id = mm_para['person_id']
        
        # person head 3d 
        processed_folder = osp.dirname(self.img_path)
        motion_sub1_head = self.load_optic_timestamp(processed_folder, "sub1_head_motion")
        
        # project to screen
        data_folder = str(Path(self.args.data_path).parent.parent)
        extri_file = osp.join(data_folder,'CamExtr.txt')
        cam_RT = torch.from_numpy(np.loadtxt(extri_file)).float()
        cam_R = cam_RT[:3,:3]
        cam_T = cam_RT[:3,3:4]
        sub1_head = torch.from_numpy(motion_sub1_head[0,:3]).float() # [,3] get the first frame head motion 
        cam_head = torch.mm(cam_R,sub1_head.unsqueeze(1)) + cam_T
        uv_head = torch.mm(torch.from_numpy(self.cam_mat).float(), cam_head)
        uv_head = torch.div(uv_head, uv_head[2])

        # find the closest
        n_person = int(mm_bbx.shape[0]/300)
        dis = 1e10
        person_id = -1
        for id in range(n_person):
            bbx = mm_bbx[mm_person_id==id][0] # [1,2]
            bbx_mid = (bbx[0] + bbx[2])/2.0
            t_dis = (uv_head[0] - bbx_mid)**2
            if t_dis<dis:
                dis = t_dis
                person_id = id
        
        # return id
        return person_id

    def load_data(self):
        # only need the necessary range
        data_folder =osp.dirname( osp.dirname( osp.dirname(self.img_path)))
        take_id = self.img_path.split('/')[-3]
        s_frame, e_frame, _, _ = self.loadseqseg(data_folder, take_id, stage=self.args.catch_throw)
        if self.args.num_frame is not None:
            e_frame = s_frame + self.args.num_frame -1
        
        # path to image frames
        img_paths = None
        if self.img_path is not None:
            img_paths = [osp.join(self.img_path, img_fn)
                            for img_fn in os.listdir(self.img_path)
                            if img_fn.endswith('.png') or
                            img_fn.endswith('.jpg') and
                            not img_fn.startswith('.')]
            img_paths = sorted(img_paths)

        frame_names = ['_'.join(f.split('/')[-1].split('.')[:-1]) for f in img_paths]
        num_frames = len(frame_names)
        print('Found video with %d frames...' % (num_frames))
        print('Stage: %s' % self.args.catch_throw)
        print('Used frames: %d - %d' % (s_frame, e_frame))

        # floor plane
        floor_plane = np.array(DEFAULT_GROUND)
        
        # load head and hands optitrack info. 
        processed_folder = osp.dirname(self.img_path)
        motion_sub1_head = self.load_optic_timestamp(processed_folder, "sub1_head_motion")
        motion_sub2_head = self.load_optic_timestamp(processed_folder, "sub2_head_motion")
        # motion_sub1_head[:,1] -= 0.04 # head offset -- 4cm
        motion_rhand = self.load_optic_timestamp(processed_folder, "sub1_right_hand_motion")
        motion_lhand = self.load_optic_timestamp(processed_folder, "sub1_left_hand_motion")
        
        # load mm pose
        mm_pose = None
        mm_file = self.args.mmhuman
        mm_para = np.load(mm_file, allow_pickle=True)
        mm_smpl = mm_para['smpl']
        mm_person_id = mm_para['person_id']
        fl = mm_smpl.flat
        for value in fl:
            # print(key)
            mm_pose= value['body_pose']

        seq_intervals = []
        num_seqs = 1
        self.seq_len = num_frames
        seq_intervals = [(0, self.seq_len)]

        # intrinsics
        cam_mat = self.cam_mat

        # get data 
        data_out = {
            'img_paths' : [],
            'mask_paths' : [],
            'cam_matx' : [],
            'joints2d' : [],
            'floor_plane' : [],
            'names' : [],
            'sub1_head' :[],
            'sub2_head' :[],
            'rhand' : [],
            'lhand': [],
            'rhand_pose' : [],
            'lhand_pose' : [],
            'mmhuman' : [],
        }

        data_out['cam_matx']= cam_mat

        rgbd_id = img_paths[0].split('/')[-2]
        data_out['floor_plane']=floor_plane
        data_out['names']=self.video_name

        if img_paths is not None:
            data_out['img_paths']=img_paths[s_frame:e_frame+1] 
            
        data_out['sub1_head']=motion_sub1_head[s_frame:e_frame+1,:] 
        data_out['sub2_head']=motion_sub2_head[s_frame:e_frame+1,:] 
        data_out['rhand']=motion_rhand[s_frame:e_frame+1,:] 
        data_out['lhand']=motion_lhand[s_frame:e_frame+1,:] 
        n = data_out['lhand'].shape[0]
        
        # find the mmhuman pose for the sub1 person(the left one in h2tc data)
        person_id = self.check_person_id(mm_para)
        data_out['mmhuman']=mm_pose[mm_person_id == person_id][:n,:,:].reshape((n,69)) 
        data_out['mmhuman']=data_out['mmhuman'][:,0:63]

        data_path = self.img_path
        data_path = Path(data_path)
        data_folder = str(data_path.parent.parent)
        take_id = data_folder.split('/')[-1]
        data_folder = str(data_path.parent.parent.parent)
        lhand_pose, rhand_pose = loadhandspose(data_folder=data_folder, take_id=take_id, s_frame=s_frame,e_frame=e_frame)
        data_out['rhand_pose'] = rhand_pose.float()
        data_out['lhand_pose'] = lhand_pose.float()

        if self.args.fps != self.DEFAULT_FPS:
            # supersample
            if self.args.fps > self.DEFAULT_FPS: 
                print('Cannot supersample data, set the fps as the default fps 60!')
                self.args.fps = self.DEFAULT_FPS
            # downsample
            else: 
                fps_ratio = float(self.args.fps) / self.DEFAULT_FPS
                img_length = len(data_out['img_paths'])
                print('Downsamp ratio: %f' % (fps_ratio))
                new_num_frames = int(fps_ratio*img_length)
                print('Downsamp num frames: %d' % (new_num_frames))
                # print(cur_num_frames)
                # print(new_num_frames)
                downsamp_inds = np.linspace(0, img_length-1, num=new_num_frames, dtype=int)
                self.seq_len = img_length
                data_out['img_paths'] = np.asarray(data_out['img_paths'])[downsamp_inds].tolist()
                data_out['sub1_head'] = data_out['sub1_head'][downsamp_inds,:]
                data_out['sub2_head'] = data_out['sub2_head'][downsamp_inds,:]
                data_out['rhand'] = data_out['rhand'][downsamp_inds,:]
                data_out['lhand'] = data_out['lhand'][downsamp_inds,:]
                data_out['rhand_pose'] = data_out['rhand_pose'][downsamp_inds,:]
                data_out['lhand_pose'] = data_out['lhand_pose'][downsamp_inds,:]
                data_out['mmhuman'] = data_out['mmhuman'][downsamp_inds,:]

        return data_out, seq_intervals

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        obs_data = dict()
        gt_data = dict()

        # floor plane
        obs_data['floor_plane'] = self.data_dict['floor_plane']
        # intrinsics
        gt_data['cam_matx'] = torch.Tensor(self.data_dict['cam_matx'])
        # meta-data
        gt_data['name'] = self.data_dict['names']

        # the frames used in this subsequence
        obs_data['seq_interval'] = torch.Tensor(list(self.seq_intervals)).to(torch.int)
        # motion of sub1_head, rhand, lhand
        obs_data['sub1_head'] = torch.Tensor(self.data_dict['sub1_head'])
        obs_data['sub2_head'] = torch.Tensor(self.data_dict['sub2_head'])
        obs_data['rhand'] = torch.Tensor(self.data_dict['rhand'])
        obs_data['lhand'] = torch.Tensor(self.data_dict['lhand'])
        obs_data['rhand_pose'] = torch.Tensor(self.data_dict['rhand_pose'])
        obs_data['lhand_pose'] = torch.Tensor(self.data_dict['lhand_pose'])
        obs_data['mmhuman'] = torch.Tensor(self.data_dict['mmhuman'])
        
        return obs_data, gt_data
    
    # load optic-track data
    def load_optic_timestamp(self, processed_folder, roi_name, dim_feat=7):
        opti = {}
        cvs_path = os.path.join(processed_folder, roi_name + ".csv")
        with open(cvs_path, "r") as f:
            opti[roi_name] = {}  # new sub dict for a object ID
            for line in f.readlines()[1:]:  # iterate every row from 2nd
                line = line.split(",")  # a list of separated data
                # timestamp as key, the rest data as value
                tmp = [float(x) for x in line[1:]]
                opti[roi_name][int(line[0])] = tmp

        # get aligned info
        align_path = os.path.join(processed_folder, "alignment.json")
        with open(align_path, "r") as f:
            aligned = json.loads(f.read())
            
        # extract the frames' info.
        frame_num = len(aligned)
        data = np.zeros(shape=(frame_num, dim_feat))  # position only
        data[:,-1]=1
        dim_f = data.shape[1]
        for i in range(0, frame_num, 1):
            id = str(i)
            try:
                tmp = opti[roi_name][aligned[id][roi_name]]
                data[i] = tmp[:dim_f]
                # data = data / np.array([2.0, 2.0, 2.0])
            except:
                print("%s get error in frame %d" % (roi_name, i))
                continue
        return data
    

    def loadseqseg(self, data_folder, take_id, stage="throw"):
    
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
        # action = anno[stage]
        
        # get the start and end frame
        s_frame = 0
        e_frame = 0
        if stage == "throw":
            e_frame = anno[stage].time_point["sub1_head_motion"]["frame"]
        else:
            e_frame = anno[stage].time_point_stable["sub1_head_motion"]["frame"]
        for i in range(e_frame - s_frame + 1):
            ts = aligned[str(i)]["right_hand_pose"]
            ts_L = aligned[str(i)]["left_hand_pose"]
            if ts != None and ts_L!=None:
                s_frame = i
                break
        return s_frame, e_frame, anno, aligned    
