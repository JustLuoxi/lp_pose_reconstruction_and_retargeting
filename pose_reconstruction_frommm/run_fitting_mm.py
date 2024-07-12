# Author: Xi Luo
# Email: sunshine.just@outlook.com
# Multi-modal based human pose optimization

from pathlib import Path
import sys, os, glob
import cv2
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import time
import traceback

import numpy as np

import torch
from torch.utils.data import DataLoader

from config import parse_args
from utils.logging import mkdir
from utils.logging import Logger, cp_files
from motion_optimizer import MotionOptimizer
from human_tools.body_model import BodyModel
from h2tc_fit_dataset_mm import H2TCFitDataset
from pose_fitting.fitting_utils import NSTAGES, load_vposer 

def main(args, config_file):
    res_out_path = None
    if args.out is not None:
        mkdir(args.out)
        # create logging system
        fit_log_path = os.path.join(args.out, 'fit_' + str(int(time.time())) + '.log')
        Logger.init(fit_log_path)

        if args.save_results or args.save_stages_results:
            res_out_path = os.path.join(args.out, 'results_out')

    # save arguments used
    Logger.log('args: ' + str(args))
    # and save config
    cp_files(args.out, [config_file])
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    B = args.batch_size
    
    dataset = None
    data_fps = args.data_fps
    im_dim = None
   
    img_folder = args.data_path

    # read images
    img_paths = glob.glob(img_folder + "/left*.[jp][pn]g")
    img_paths.sort()
    img_path = img_paths[0]
    img_shape = cv2.imread(img_path).shape

    # Create dataset 
    vid_name = args.data_path.split('/')[-3]
    dataset = H2TCFitDataset(joints2d_path=None,
                                cam_mat=None,
                                seq_len=args.rgb_seq_len,
                                img_path=img_folder,
                                load_img=False,
                                video_name=vid_name,
                                args = args,
                            )
    cam_mat = dataset.cam_mat

    data_fps = args.fps
    im_dim = tuple(img_shape[:-1][::-1])

    data_loader = DataLoader(dataset, 
                            batch_size=B,
                            shuffle=args.shuffle,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False,
                            worker_init_fn=lambda _: np.random.seed())

    # weights for optimization loss terms
    loss_weights = {
        'joints2d' : args.joint2d_weight,
        'pose_prior' : args.pose_prior_weight,
        'shape_prior' : args.shape_prior_weight,
        'joints3d_smooth' : args.joint3d_smooth_weight,
        'joint_consistency' : args.joint_consistency_weight,
        'bone_length' : args.bone_length_weight,
    }

    max_loss_weights = {k : max(v) for k, v in loss_weights.items()}
    all_stage_loss_weights = []
    for sidx in range(NSTAGES):
        stage_loss_weights = {k : v[sidx] for k, v in loss_weights.items()}
        all_stage_loss_weights.append(stage_loss_weights)
        
    use_joints2d = max_loss_weights['joints2d'] > 0.0

    # must always have pose prior to optimize in latent space
    pose_prior, _ = load_vposer(cur_file_path + args.vposer)
    pose_prior = pose_prior.to(device)
    pose_prior.eval()

    if args.save_results:
        all_res_out_paths = []

    for i, data in enumerate(data_loader):
        start_t = time.time()
        # these dicts have different data depending on modality
        observed_data, gt_data = data
        observed_data = {k : v.to(device) for k, v in observed_data.items() if isinstance(v, torch.Tensor)}
        for k, v in gt_data.items():
            if isinstance(v, torch.Tensor):
                gt_data[k] = v.to(device)
        cur_batch_size = observed_data[list(observed_data.keys())[0]].size(0)
        T = observed_data['rhand'].size(1)

        seq_names = []
        for gt_idx, gt_name in enumerate(gt_data['name']):
            seq_name = gt_name + '_' + str(int(time.time())) + str(gt_idx)
            Logger.log(seq_name)
            seq_names.append(seq_name)

        cur_res_out_paths = []
        for seq_name in seq_names:
            # set current out paths based on sequence name
            if res_out_path is not None:
                cur_res_out_path = res_out_path
                mkdir(cur_res_out_path)
                cur_res_out_paths.append(cur_res_out_path)
        cur_res_out_paths = cur_res_out_paths if len(cur_res_out_paths) > 0 else None
        if cur_res_out_paths is not None  and args.save_results:
            all_res_out_paths += cur_res_out_paths

        # get body model
        Logger.log('Loading SMPL model from %s...' % (args.smplh))
        body_model_path = args.smplh
        fit_gender = body_model_path.split('/')[-2]
        num_betas = 16 if 'betas' not in gt_data else gt_data['betas'].size(2)
        body_model = BodyModel(bm_path=body_model_path,
                                num_betas=num_betas,
                                batch_size=cur_batch_size*T,
                                use_vtx_selector=use_joints2d).to(device)

        if body_model.model_type != 'smplh':
            print('Only SMPL+H model is supported for current algorithm!')
            exit()

        cam_mat = None
        if 'cam_matx' in gt_data:
            cam_mat = gt_data['cam_matx'].to(device)

        # create optimizer
        optimizer = MotionOptimizer(device,
                                    body_model,
                                    num_betas,
                                    cur_batch_size,
                                    dataset.data_len,
                                    [k for k in observed_data.keys()],
                                    all_stage_loss_weights,
                                    pose_prior,
                                    cam_mat,
                                    args.joint2d_sigma,
                                    im_dim=im_dim,
                                    args=args)

        # run optimizer
        try:
            optimizer.run(observed_data,
                            data_fps=data_fps,
                            lr=args.lr,
                            num_iter=args.num_iters,
                            lbfgs_max_iter=args.lbfgs_max_iter,
                            stages_res_out=cur_res_out_paths,
                            fit_gender=fit_gender)

            elapsed_t = time.time() - start_t
            Logger.log('Optimized sequence %d in %f s' % (i, elapsed_t))

        except Exception as e:
            Logger.log('Caught error in current optimization! Skipping...') 
            Logger.log(traceback.format_exc()) 

        if i < (len(data_loader) - 1):
            del optimizer
        del body_model
        del observed_data
        del gt_data
        torch.cuda.empty_cache()

if __name__=='__main__':
    args = parse_args(sys.argv[1:])
    config_file = sys.argv[1:][0][1:]
    main(args, config_file)