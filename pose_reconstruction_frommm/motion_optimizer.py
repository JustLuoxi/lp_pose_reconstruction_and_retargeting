# Author: Xi Luo
# Email: sunshine.just@outlook.com
# Motion optimizer for multi-modal based human pose optimization

import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import torch
import trimesh
import numpy as np
import torch.nn as nn
from pathlib import Path

from utils.logging import Logger
from utils.transforms import rotation_matrix_to_angle_axis
from human_tools.body_model import SMPL_JOINTS, KEYPT_VERTS, smpl_to_openpose
from pose_fitting.fitting_utils import OP_IGNORE_JOINTS
from pose_fitting.fitting_loss import FittingLoss
from pose_fitting.fitting_utils import *

LINE_SEARCH = 'strong_wolfe'
J_BODY = len(SMPL_JOINTS)-1 # no root

class MotionOptimizer():
    ''' Fits SMPL shape and motion to observation sequence '''

    def __init__(self, device,
                       body_model, # SMPL model to use (its batch_size should be B*T)
                       num_betas, # beta size in SMPL model
                       batch_size, # number of sequences to optimize
                       seq_len, # length of the sequences
                       observed_modalities, # list of the kinds of observations to use
                       loss_weights, # dict of weights for each loss term
                       pose_prior, # VPoser model
                       camera_matrix=None, # camera intrinsics to use for reprojection if applicable
                       joint2d_sigma=100,
                       im_dim=(1080,1080),# image dimensions to use for visualization
                       args=None): 
        B, T = batch_size, seq_len
        self.device = device
        self.batch_size = B
        self.seq_len = T
        self.body_model = body_model
        self.num_betas = num_betas
        self.im_dim = im_dim
        self.args = args
        self.take_id = int(self.args.data_path.replace('\\', '/').split('/')[-3])

        #
        # create the optimization variables
        #

        # number of states to explicitly optimize for
        num_state_steps = T
        # latent body pose
        self.pose_prior = pose_prior
        self.latent_pose_dim = self.pose_prior.latentD
        self.latent_pose = torch.zeros((B, num_state_steps, self.latent_pose_dim)).to(device)
        # root (global) transformation
        self.body_pose = torch.zeros((B, num_state_steps, 63)).to(device)
        self.trans = torch.zeros((B, num_state_steps, 3)).to(device)
        self.root_orient = torch.zeros((B, num_state_steps, 3)).to(device) # aa parameterization
        self.root_orient[:,:,0] = np.pi
        # body shape
        self.betas = torch.zeros((B, num_betas)).to(device) # same shape for all steps
        self.left_hand_pose, self.right_hand_pose = None, None

        self.cam_f = self.cam_center = None
        if camera_matrix is None:
            Logger.log('Must have camera intrinsics (camera_matrix) to calculate losses!')
            exit()
        else:
            cam_fx = camera_matrix[:, 0, 0]
            cam_fy = camera_matrix[:, 1, 1]
            cam_cx = camera_matrix[:, 0, 2]
            cam_cy = camera_matrix[:, 1, 2]
            # focal length and center are same for all timesteps
            self.cam_f = torch.stack([cam_fx, cam_fy], dim=1)
            self.cam_center = torch.stack([cam_cx, cam_cy], dim=1)
        self.use_camera = self.cam_f is not None and self.cam_center is not None
        
        #
        # create the loss function
        #
        self.smpl2op_map = smpl_to_openpose(body_model.model_type, use_hands=False, use_face=False, use_face_contour=False, openpose_format='coco25')
        self.fitting_loss = FittingLoss(loss_weights, 
                                        self.smpl2op_map,
                                        OP_IGNORE_JOINTS,
                                        self.cam_f,
                                        self.cam_center,
                                        joints2d_sigma=joint2d_sigma,
                                        args = args).to(device)
        
    # load previous optimizing results
    def load_optresult(self, opt_path):
        smpl_para = np.load(opt_path)
        smpl_poses = smpl_para['pose_body'] # seq_len*72
        smpl_poses = torch.from_numpy(smpl_poses).unsqueeze(0).to('cuda')
        self.body_pose = smpl_poses
        self.betas[0,:] = torch.from_numpy(smpl_para['betas']) # 16
        self.trans[0,:,:] = torch.from_numpy(smpl_para['trans']) # seq_len*3
        self.root_orient[0,:,:] = torch.from_numpy(smpl_para['root_orient'])
        self.latent_pose = self.pose2latent(smpl_poses).detach()

    def run(self, observed_data,
                  data_fps=30,
                  lr=1.0,
                  num_iter=[30, 70],
                  lbfgs_max_iter=20,
                  stages_res_out=None,
                  fit_gender='neutral'):

        # load hand pose 
        self.left_hand_pose, self.right_hand_pose = observed_data['lhand_pose'][:,:self.seq_len,:], observed_data['rhand_pose'][:,:self.seq_len,:]
        
        ## load camera extrinsic
        data_folder = str(Path(self.args.data_path).parent.parent)
        extri_file = osp.join(data_folder,'CamExtr.txt')
        if osp.exists(extri_file):
            cam_RT = torch.from_numpy(np.loadtxt(extri_file)).float()
        else:
            assert osp.exists(extri_file)
            
        self.fitting_loss.cam_R = cam_RT[:3,:3]
        self.fitting_loss.cam_t = cam_RT[:3,3]
        cam_RT = cam_RT.repeat(observed_data['rhand'].shape[1],1,1).to('cuda') # [seq_num, 4, 4]
        
        if len(num_iter) < 2:
            print('Must have num iters not less than 2 stages! But %d stages were given!' % (len(num_iter)))
            exit()

        per_stage_outputs = {} # SMPLH results after each stage

        #
        # Initialize using mm pose
        #
        body_pose = observed_data['mmhuman']
        self.latent_pose = self.pose2latent(body_pose).detach()
        
        flag_rhand = (observed_data['rhand'][0,:,0]!=0).squeeze()
        flag_lhand = (observed_data['lhand'][0,:,0]!=0).squeeze()

        #
        # Stage I: global root and orientation
        #
        Logger.log('Optimizing stage 1 - global root translation and orientation for %d interations...' % (num_iter[0]))
        cur_res_out_path = os.path.join(stages_res_out[0], 'stage1_results.npz')
        if os.path.exists(cur_res_out_path):
            print('Loading stage 1 fitting results in %s' % cur_res_out_path)
            bdata = np.load(cur_res_out_path)
            self.trans = torch.from_numpy(bdata['trans']).unsqueeze(0).to(self.device)
            self.root_orient = torch.from_numpy(bdata['root_orient']).unsqueeze(0).to(self.device)
            self.betas = torch.from_numpy(bdata['betas']).unsqueeze(0).to(self.device)
            body_pose = torch.from_numpy(bdata['pose_body']).unsqueeze(0).to(self.device)
            pose_hand = torch.from_numpy(bdata['pose_hand']).unsqueeze(0).to(self.device)
            self.latent_pose = self.pose2latent(body_pose).detach()
            stage1_pred_data, _ = self.smpl_results(self.trans, self.root_orient, body_pose, self.betas)
            per_stage_outputs['stage1'] = stage1_pred_data
        else:
            self.fitting_loss.set_stage(0)
            self.trans.requires_grad = True
            self.root_orient.requires_grad = True
            self.betas.requires_grad = False
            self.latent_pose.requires_grad = False

            root_opt_params = [self.trans, self.root_orient]

            root_optim = torch.optim.LBFGS(root_opt_params,
                                            max_iter=lbfgs_max_iter,
                                            lr=lr,
                                            line_search_fn=LINE_SEARCH)
            
            for i in range(num_iter[0]): 
                # Logger.log('ITER: %d' % (i))
                self.fitting_loss.cur_optim_step = i 
                def closure(): 
                    root_optim.zero_grad() 
                    
                    pred_data = dict() 
                    # Use current params to go through SMPL and get joints3d, verts3d, points3d 
                    body_pose = self.latent2pose(self.latent_pose) 
                    pred_data, _ = self.smpl_results(self.trans, self.root_orient, body_pose, self.betas) 
                    # compute data losses only 
                    # loss, _ = self.fitting_loss.root_fit(observed_data, pred_data) 
                    
                    # head facing constrain 
                    j_0 = pred_data["joints3d"][0,:,0]
                    j_13 = pred_data["joints3d"][0,:,13]
                    j_14 = pred_data["joints3d"][0,:,14]
                    j_o_13 = j_13 - j_0
                    j_13_14 = j_14 - j_13 
                    facing = torch.cross(j_o_13, j_13_14)
                    loss_facing = torch.sum(torch.relu(-facing[:,2])) # sub1 person, the left one, z>0 
                        
                    # align to opti-track 
                    loss_opti = self.fitting_loss.optitrack_fit(observed_data, pred_data, flag_rhand, flag_lhand)
                    loss = 1000*loss_facing + 20*loss_opti

                    loss.backward() 
                    return loss 

                root_optim.step(closure)

            body_pose = self.latent2pose(self.latent_pose)
            stage1_pred_data, _ = self.smpl_results(self.trans, self.root_orient, body_pose, self.betas)
            per_stage_outputs['stage1'] = stage1_pred_data

            # save results
            if stages_res_out is not None:
                res_betas = self.betas.clone().detach().cpu().numpy()
                res_trans = self.trans.clone().detach().cpu().numpy()
                res_root_orient = self.root_orient.clone().detach().cpu().numpy()
                res_body_pose = body_pose.clone().detach().cpu().numpy()
                pose_hand = torch.cat([ self.left_hand_pose.squeeze(0), self.right_hand_pose.squeeze(0)],dim=1) if self.left_hand_pose!=None else None
                
                for bidx, res_out_path in enumerate(stages_res_out):
                    np.savez(cur_res_out_path, betas=res_betas[bidx],
                                            trans=res_trans[bidx],
                                            root_orient=res_root_orient[bidx],
                                            pose_body=res_body_pose[bidx],
                                            pose_hand = pose_hand.cpu().numpy())
            
        # 
        # Stage II pose and wrist
        # 
        Logger.log('Optimizing stage 2 - pose and wrist for %d iterations..' % (num_iter[1]))
        cur_res_out_path = os.path.join(stages_res_out[0], 'stage2_results.npz')
        self.fitting_loss.set_stage(1)
        self.trans.requires_grad = True
        self.root_orient.requires_grad = True
        self.betas.requires_grad = True
        self.latent_pose.requires_grad = True

        smpl_opt_params = [self.trans, self.root_orient, self.betas, self.latent_pose]

        smpl_optim = torch.optim.LBFGS(smpl_opt_params,
                                    max_iter=lbfgs_max_iter,
                                    lr=lr,
                                    line_search_fn=LINE_SEARCH)

        MSELoss = nn.MSELoss()
        # gt hands pose
        gt_pose_lh=torch.cat((observed_data['lhand'][:,:,-1:],observed_data['lhand'][:,:,3:-1]), dim=2)
        gt_pose_rh=torch.cat((observed_data['rhand'][:,:,-1:],observed_data['rhand'][:,:,3:-1]), dim=2)
        # gt pose to matrix 
        gt_pose_lh_mat = quaternion_to_matrix(gt_pose_lh).to(torch.float32).to('cuda')
        gt_pose_rh_mat = quaternion_to_matrix(gt_pose_rh).to(torch.float32).to('cuda')
        lh_roty = torch.tensor([[0,0,-1],[0,1,0],[1,0,0]]).to(torch.float32).to('cuda')  # y -90° 
        rh_roty = torch.tensor([[0,0,1],[0,1,0],[-1,0,0]]).to(torch.float32).to('cuda') # y 90
        gt_pose_lh_mat =  torch.matmul(gt_pose_lh_mat, lh_roty).squeeze(0)
        gt_pose_rh_mat = torch.matmul(gt_pose_rh_mat, rh_roty).squeeze(0)

        gt_pose_lh_euler = matrix_to_euler_angles(gt_pose_lh_mat,"XYZ")
        gt_pose_rh_euler = matrix_to_euler_angles(gt_pose_rh_mat,"XYZ")
        # jitter in rotation
        flag_jitter_pose_lh =torch.sum((gt_pose_lh_euler[1:,:] - gt_pose_lh_euler[:-1,:])**2, dim=1) > 10.0
        flag_jitter_pose_rh =torch.sum((gt_pose_rh_euler[1:,:] - gt_pose_rh_euler[:-1,:])**2, dim=1) > 10.0 
        indx_jitter_pose_lh = flag_jitter_pose_lh.nonzero(as_tuple=True)[0]
        indx_jitter_pose_rh = flag_jitter_pose_rh.nonzero(as_tuple=True)[0]
        flag_lhand[indx_jitter_pose_lh] = False
        flag_rhand[indx_jitter_pose_rh] = False

        loss_record = torch.zeros(1,1)      
        for i in range(num_iter[1]):
            # Logger.log('ITER: %d' % (i))
            def closure():
                smpl_optim.zero_grad()
                
                pred_data = dict()
                # Use current params to go through SMPL and get joints3d, verts3d, points3d
                body_pose = self.latent2pose(self.latent_pose)
                
                pred_data, _ = self.smpl_results(self.trans, self.root_orient, body_pose, self.betas)
                pred_data['latent_pose'] = self.latent_pose 
                pred_data['betas'] = self.betas
                # compute data losses and pose prior
                loss, stats_dict = self.fitting_loss.smpl_fit(observed_data, pred_data, self.seq_len)
                # log_cur_stats(stats_dict, loss, iter=i)
                
                ## loss_opti: align to opti-track 
                loss_opti = self.fitting_loss.optitrack_fit(observed_data, pred_data, flag_rhand, flag_lhand)
                loss = loss + 50*loss_opti
                
                ## loss_wrist: compute hand pose [global pose] via kinematic tree  [seq_num*52*4*4]
                if i > 10:
                    trans_mat = self.body_model.get_global_transforms_foralljoints(self.betas, self.root_orient, body_pose)
                    pose_lh = matrix_to_euler_angles(trans_mat[:,20,:3,:3],"XYZ") # [seq_num. 3. 3] -> [seq_num. 3]
                    pose_rh = matrix_to_euler_angles(trans_mat[:,21,:3,:3],"XYZ")
                    loss_wrist = MSELoss(pose_lh[flag_lhand], gt_pose_lh_euler[flag_lhand]) + \
                                    MSELoss(pose_rh[flag_rhand], gt_pose_rh_euler[flag_rhand])
                    
                    ## wrist smoothness loss
                    loss_poselh_smooth = (pose_lh[1:,:] - pose_lh[:-1,:])**2
                    loss_poserh_smooth = (pose_rh[1:,:] - pose_rh[:-1,:])**2
                    loss_pose_wrist_smooth = 0.5*torch.sum(loss_poselh_smooth + loss_poserh_smooth)
                
                    loss = loss + 300*loss_wrist + loss_pose_wrist_smooth
                                    
                loss_record[0] = loss.item()
                loss.backward()
                return loss

            smpl_optim.step(closure)

        self.body_pose = self.latent2pose(self.latent_pose).detach()
        
        self.body_pose[:,:,:30] = observed_data['mmhuman'][:,:,:30] # set mm pose [lower limb] to final poses
        self.trans = torch.zeros_like(self.trans)
        self.root_orient = torch.zeros_like(self.root_orient)
        
        stage2_pred_data, _ = self.smpl_results(self.trans, self.root_orient, self.body_pose, self.betas)
        per_stage_outputs['stage2'] = stage2_pred_data

        if stages_res_out is not None:                 
            res_betas = self.betas.clone().detach().cpu().numpy()
            res_trans = self.trans.clone().detach().cpu().numpy()
            res_root_orient = self.root_orient.clone().detach().cpu().numpy()
            res_body_pose = self.body_pose.clone().detach().cpu().numpy()
            pose_hand = torch.cat([ self.left_hand_pose.squeeze(0), self.right_hand_pose.squeeze(0)],dim=1) if self.left_hand_pose!=None else None
            for bidx, res_out_path in enumerate(stages_res_out):
                cur_res_out_path = os.path.join(res_out_path, 'stage2_results.npz')
                np.savez(cur_res_out_path, betas=res_betas[bidx],
                                        trans=res_trans[bidx],
                                        root_orient=res_root_orient[bidx],
                                        pose_body=res_body_pose[bidx],
                                        pose_hand = pose_hand.cpu().numpy())
                final_loss = loss_record.detach().cpu().numpy()
                np.savetxt( os.path.join(res_out_path, 'loss %f.txt' % final_loss), final_loss)
                print("Loss:")
                print(loss_record.detach().cpu().numpy())

        final_optim_res = self.get_optim_result(self.body_pose)
        self.save_smpl_ply()
           
        return final_optim_res,  per_stage_outputs
          
    def save_smpl_ply(self):

        with torch.no_grad():
            pred_output = self.body_model.bm(
                                    body_pose=self.body_pose[0,:,:],
                                    global_orient=self.root_orient[0,:,:],
                                    transl=self.trans[0,:,:],
                                    left_hand_pose=self.left_hand_pose[0,:,:],
                                    right_hand_pose=self.right_hand_pose[0,:,:]
                                    )
        verts = pred_output.vertices.cpu().numpy()
        faces = self.body_model.bm.faces
        
        meshes_dir = os.path.join(self.args.out, "body_meshes_h2tc")
        print(f"save meshes to \"{meshes_dir}\"")
        os.makedirs(meshes_dir, exist_ok=True)
        
        n = len(verts)
        for ii in range(n):
            verts0 = np.array(verts[ii])
            mesh0 = trimesh.Trimesh(verts0, faces)
                
            # save mesh0
            fram_name =  str(ii)
            filename =  "%06d_h2tc_smplh_%s.ply" % (self.take_id, fram_name)   
            out_mesh_path = os.path.join(meshes_dir, filename)
            mesh0.export(out_mesh_path)
    
    def get_optim_result(self, body_pose=None):
        '''
        Collect final outputs into a dict.
        '''
        if body_pose is None:
            body_pose = self.latent2pose(self.latent_pose)
        optim_result = {
            'trans' : self.trans.clone().detach(),
            'root_orient' : self.root_orient.clone().detach(),
            'pose_body' : body_pose.clone().detach(),
            'betas' : self.betas.clone().detach(),
            'latent_pose' : self.latent_pose.clone().detach()   
        }
        # optim_result['latent_motion'] = self.latent_motion.clone().detach()
        if self.left_hand_pose!=None:
            pose_hand = torch.cat([ self.left_hand_pose, self.right_hand_pose],dim=2)
            optim_result['pose_hand'] = pose_hand.clone().detach()
        
        return optim_result

    def latent2pose(self, latent_pose):
        '''
        Converts VPoser latent embedding to aa body pose.
        latent_pose : B x T x D
        body_pose : B x T x J*3
        '''
        B, T, _ = latent_pose.size()
        latent_pose = latent_pose.reshape((-1, self.latent_pose_dim))
        body_pose = self.pose_prior.decode(latent_pose, output_type='matrot')
        body_pose = rotation_matrix_to_angle_axis(body_pose.reshape((B*T*J_BODY, 3, 3))).reshape((B, T, J_BODY*3))
        return body_pose

    def pose2latent(self, body_pose):
        '''
        Encodes aa body pose to VPoser latent space.
        body_pose : B x T x J*3
        latent_pose : B x T x D
        '''
        B, T, _ = body_pose.size()
        body_pose = body_pose.reshape((-1, J_BODY*3))
        latent_pose_distrib = self.pose_prior.encode(body_pose)
        latent_pose = latent_pose_distrib.mean.reshape((B, T, self.latent_pose_dim))
        return latent_pose

    def smpl_results(self, trans, root_orient, body_pose, beta):
        '''
        Forward pass of the SMPL model and populates pred_data accordingly with
        joints3d, verts3d, points3d.

        trans : B x T x 3
        root_orient : B x T x 3
        body_pose : B x T x J*3
        beta : B x D
        '''
        B, T, _ = trans.size()
        if T == 1:
            # must expand to use with body model
            trans = trans.expand((self.batch_size, self.seq_len, 3))
            root_orient = root_orient.expand((self.batch_size, self.seq_len, 3))
            body_pose = body_pose.expand((self.batch_size, self.seq_len, J_BODY*3))
        elif T != self.seq_len:
            # raise NotImplementedError('Only supports single or all steps in body model.')
            pad_size = self.seq_len - T
            trans, root_orient, body_pose = self.zero_pad_tensors([trans, root_orient, body_pose], pad_size)


        betas = beta.reshape((self.batch_size, 1, self.num_betas)).expand((self.batch_size, self.seq_len, self.num_betas))
        pose_hand = torch.cat([ self.left_hand_pose.squeeze(0), self.right_hand_pose.squeeze(0)],dim=1) if self.left_hand_pose!=None else None
        smpl_body = self.body_model(pose_body=body_pose.reshape((self.batch_size*self.seq_len, -1)), 
                                    pose_hand=pose_hand, 
                                    betas=betas.reshape((self.batch_size*self.seq_len, -1)),
                                    root_orient=root_orient.reshape((self.batch_size*self.seq_len, -1)),
                                    trans=trans.reshape((self.batch_size*self.seq_len, -1))
                                    )
        # body joints
        joints3d = smpl_body.Jtr.reshape((self.batch_size, self.seq_len, -1, 3))[:, :T]
        body_joints3d = joints3d[:,:,:len(SMPL_JOINTS),:]
        added_joints3d = joints3d[:,:,len(SMPL_JOINTS):,:]
        # ALL body vertices
        points3d = smpl_body.v.reshape((self.batch_size, self.seq_len, -1, 3))[:, :T]
        # SELECT body vertices
        verts3d = points3d[:, :T, KEYPT_VERTS, :]

        pred_data = {
            'joints3d' : body_joints3d,
            'points3d' : points3d,
            'verts3d' : verts3d,
            'joints3d_extra' : added_joints3d, # hands and selected OP vertices (if applicable) 
            'faces' : smpl_body.f # always the same, but need it for some losses
        }
        
        return pred_data, smpl_body

    def zero_pad_tensors(self, pad_list, pad_size):
        '''
        Assumes tensors in pad_list are B x T x D and pad temporal dimension
        '''
        B = pad_list[0].size(0)
        new_pad_list = []
        for pad_idx, pad_tensor in enumerate(pad_list):
            padding = torch.zeros((B, pad_size, pad_tensor.size(2))).to(pad_tensor)
            new_pad_list.append(torch.cat([pad_tensor, padding], dim=1))
        return new_pad_list
    