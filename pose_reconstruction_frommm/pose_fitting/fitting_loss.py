import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logging import Logger

from pose_fitting.fitting_utils import OP_NUM_JOINTS, perspective_projection, apply_robust_weighting, gmof

from human_tools.body_model import SMPL_PARENTS

CONTACT_HEIGHT_THRESH = 0.08

class FittingLoss(nn.Module):
    '''
    Functions to compute all needed losses for fitting.
    '''

    def __init__(self, loss_weights,
                       smpl2op_map=None,
                       ignore_op_joints=None,
                       cam_f=None, 
                       cam_cent=None, 
                       joints2d_sigma=100,
                       args=None):
        super(FittingLoss, self).__init__()
        self.all_stage_loss_weights = loss_weights
        self.cur_stage_idx = 0
        self.loss_weights = self.all_stage_loss_weights[self.cur_stage_idx]
        self.smpl2op_map = smpl2op_map
        self.ignore_op_joints = ignore_op_joints
        self.cam_f = cam_f
        self.cam_cent = cam_cent
        self.joints2d_sigma = joints2d_sigma
        self.args=args

        self.can_reproj = self.smpl2op_map is not None and \
                          self.cam_f is not None and \
                          self.cam_cent is not None
        if self.can_reproj:
            self.cam_f = self.cam_f.reshape((-1, 1, 2))
            self.cam_cent = self.cam_cent.reshape((-1, 1, 2))

        sum_loss_weights = {k : 0.0 for k in self.loss_weights.keys()}
        for stage_idx in range(len(self.all_stage_loss_weights)):
            for k in sum_loss_weights.keys():
                sum_loss_weights[k] += self.all_stage_loss_weights[stage_idx][k]

        self.l2_loss = nn.MSELoss(reduction='none')

        self.cur_optim_step = 0
        
        self.cam_t = torch.zeros((1,3))
        self.cam_R = torch.eye(3)

    def set_stage(self, idx):
        ''' Sets the current stage index. Determines which loss weights are used '''
        self.cur_stage_idx = idx
        self.loss_weights = self.all_stage_loss_weights[self.cur_stage_idx]
        # Logger.log('Stage %d loss weights set to:' % (idx+1))
        # Logger.log(self.loss_weights)

    def forward(self):
        pass

    def root_fit(self, observed_data, pred_data):
        '''
        For fitting just global root trans/orientation. Only computes joint/point/vert losses, i.e. no priors.
        '''
        stats_dict = dict()
        loss = 0.0

        # 2D re-projection loss
        use_reproj_loss = 'joints2d' in observed_data and \
                          'joints3d' in pred_data and \
                          'joints3d_extra' in pred_data and \
                          self.loss_weights['joints2d'] > 0.0
        if use_reproj_loss:
            if not self.can_reproj:
                Logger.log('Must provide camera intrinsics and SMPL to OpenPose joint map to use re-projection loss!')
                exit()
            cur_loss = self.joints2d_loss(observed_data['joints2d'],
                                          pred_data['joints3d'],
                                          pred_data['joints3d_extra'],
                                          debug_img=None)
            loss += self.loss_weights['joints2d']*cur_loss
            stats_dict['joints2d'] = cur_loss

        return loss, stats_dict
        
    def smpl_fit(self, observed_data, pred_data, nsteps):
        '''
        For fitting full shape and pose of SMPL.

        nsteps used to scale single-step losses
        '''
        # first all observation losses
        loss, stats_dict = self.root_fit(observed_data, pred_data)

        # prior to keep latent pose likely
        if 'latent_pose' in pred_data and self.loss_weights['pose_prior'] > 0.0:
            cur_loss = self.pose_prior_loss(pred_data['latent_pose'])
            loss += self.loss_weights['pose_prior']*cur_loss
            stats_dict['pose_prior'] = cur_loss

        # prior to keep PCA shape likely
        if 'betas' in pred_data and self.loss_weights['shape_prior'] > 0.0:
            cur_loss = self.shape_prior_loss(pred_data['betas'])
            loss += self.loss_weights['shape_prior']*nsteps*cur_loss
            stats_dict['shape_prior'] = cur_loss

        # smooth 3d joint motion
        if self.loss_weights['joints3d_smooth'] > 0.0:
            cur_loss = self.joints3d_smooth_loss(pred_data['joints3d'])
            loss += self.loss_weights['joints3d_smooth']*cur_loss
            stats_dict['joints3d_smooth'] = cur_loss

        return loss, stats_dict

    # fit to the optitrack data 
    def optitrack_fit(self, observed_data, pred_data, flag_rhand, flag_lhand):

        # opti-track 3d points
        gt_rhand = observed_data['rhand'][0,:,:3]
        gt_lhand = observed_data['lhand'][0,:,:3]


        # predict smpl corresponding points
        smpl_rhand = pred_data['points3d'][0, :, 5459, :] # rhand id is 5459  [seq_num, 3]
        smpl_lhand = pred_data['points3d'][0, :, 2213, :] # lhand id is 2213 [seq_num, 3]

        # compute loss 
        loss_rhand = torch.sum(((gt_rhand - smpl_rhand)**2)[flag_rhand]) 
        loss_lhand = torch.sum(((gt_lhand - smpl_lhand)**2)[flag_lhand]) 
            
        loss = loss_rhand + loss_lhand  # sub1 person, the left one, z>0 
        
        return loss
        
    def get_visible_mask(self, obs_data):
        '''
        Given observed data gets the mask of visible data (that actually contributes to the loss).
        '''
        return torch.logical_not(torch.isinf(obs_data))

    def joints2d_loss(self, joints2d_obs, joints3d_pred, joints3d_extra_pred, cam_t=None, cam_R=None, debug_img=None):
        '''
        Cam extrinsics are assumed the same for entire sequence
        - cam_t : (B, 3)
        - cam_R : (B, 3, 3)
        '''
        B, T, _, _ = joints2d_obs.size()
        # need extra joints that correspond to openpose
        joints3d_full = torch.cat([joints3d_pred, joints3d_extra_pred], dim=2)
        joints3d_op = joints3d_full[:,:,self.smpl2op_map,:]
        joints3d_op = joints3d_op.reshape((B*T, OP_NUM_JOINTS, 3))

        # either use identity cam params or expand the ones given to full sequence
        if cam_t is None:
            # cam_t = torch.zeros((B*T, 3)).to(joints3d_pred)
            cam_t = self.cam_t.expand((B*T, 3)).to(joints3d_pred)
        else:
            cam_t = cam_t.unsqueeze(1).expand((B, T, 3)).reshape((B*T, 3))
        if cam_R is None:
            cam_R = self.cam_R.reshape((1, 3, 3)).expand((B*T, 3, 3)).to(joints3d_pred)
        else:
            cam_R = cam_R.unsqueeze(1).expand((B, T, 3, 3)).reshape((B*T, 3, 3))

        # project points to 2D
        cam_f = self.cam_f.expand((B, T, 2)).reshape((B*T, 2))
        cam_cent = self.cam_cent.expand((B, T, 2)).reshape((B*T, 2))
        joints2d_pred = perspective_projection(joints3d_op,
                                               cam_R,
                                               cam_t,
                                               cam_f,
                                               cam_cent)

        # compared to observations
        joints2d_pred = joints2d_pred.reshape((B, T, OP_NUM_JOINTS, 2))
        joints2d_obs_conf = joints2d_obs[:,:,:,2:3]
        if self.ignore_op_joints is not None:
            joints2d_obs_conf[:,:,self.ignore_op_joints] = 0.0 # set confidence to 0 so not weighted

        # weight errors by detection confidence
        robust_sqr_dist = gmof(joints2d_pred - joints2d_obs[:,:,:,:2], self.joints2d_sigma)
        reproj_err = (joints2d_obs_conf**2) * robust_sqr_dist
        loss = torch.sum(reproj_err)
        return loss

    def joints3d_smooth_loss(self, joints3d_pred):
        # minimize delta steps
        loss = (joints3d_pred[:,1:,:,:] - joints3d_pred[:,:-1,:,:])**2
        loss = 0.5*torch.sum(loss)
        return loss

    def pose_prior_loss(self, latent_pose_pred):
        # prior is isotropic gaussian so take L2 distance from 0
        loss = latent_pose_pred**2
        loss = torch.sum(loss)
        return loss

    def joint_consistency_loss(self, smpl_joints3d, rollout_joints3d):
        loss = (smpl_joints3d - rollout_joints3d)**2
        loss = 0.5*torch.sum(loss)
        return loss

    def shape_prior_loss(self, betas_pred):
        # prior is isotropic gaussian so take L2 distance from 0
        loss = betas_pred**2
        loss = torch.sum(loss)
        return loss
