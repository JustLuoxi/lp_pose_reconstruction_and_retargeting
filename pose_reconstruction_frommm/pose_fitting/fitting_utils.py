# Author: Xi Luo
# Email: sunshine.just@outlook.com
# Multi-modal based human pose optimization

import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))
sys.path.append(os.path.join(cur_file_path, '../..'))

import shutil, glob
import os.path as osp
from pathlib import Path
import cv2
import numpy as np
import json
import torch

from human_tools.body_model import BodyModel

from utils.transforms import batch_rodrigues
from utils.logging import mkdir, Logger

NSTAGES = 2 # number of stages in the optimization

DEFAULT_FOCAL_LEN = (699.78, 699.78) # camera fx, fy

OP_NUM_JOINTS = 25 # body keypoints
OP_IGNORE_JOINTS = [1, 9, 12] # neck and left/right hip

from typing import Optional

import torch
import torch.nn.functional as F


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)

def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])

def resize_points(points_arr, num_pts):
    '''
    Either randomly subsamples or pads the given points_arr to be of the desired size.
    - points_arr : N x 3
    - num_pts : desired num point 
    '''
    is_torch = isinstance(points_arr, torch.Tensor)
    N = points_arr.size(0) if is_torch else points_arr.shape[0]
    if N > num_pts:
        samp_inds = np.random.choice(np.arange(N), size=num_pts, replace=False)
        points_arr = points_arr[samp_inds]
    elif N < num_pts:
        while N < num_pts:
            pad_size = num_pts - N
            if is_torch:
                points_arr = torch.cat([points_arr, points_arr[:pad_size]], dim=0)
                N = points_arr.size(0)
            else:
                points_arr = np.concatenate([points_arr, points_arr[:pad_size]], axis=0)
                N = points_arr.shape[0]
    return points_arr

def compute_plane_intersection(point, direction, plane):
    '''
    Given a ray defined by a point in space and a direction, compute the intersection point with the given plane.
    Detect intersection in either direction or -direction so the given ray may not actually intersect with the plane.

    Returns the intersection point as well as s such that point + s*direction = intersection_point. if s < 0 it means
    -direction intersects.

    - point : B x 3
    - direction : B x 3
    - plane : B x 4 (a, b, c, d) where (a, b, c) is the normal and (d) the offset.
    '''
    plane_normal = plane[:,:3]
    plane_off = plane[:,3]
    s = (plane_off - bdot(plane_normal, point)) / bdot(plane_normal, direction)
    itsct_pt = point + s.reshape((-1, 1))*direction
    return itsct_pt, s

def bdot(A1, A2, keepdim=False):
    ''' 
    Batched dot product.
    - A1 : B x D
    - A2 : B x D.
    Returns B.
    '''
    return (A1*A2).sum(dim=-1, keepdim=keepdim) 

def parse_floor_plane(floor_plane):
    '''
    Takes floor plane in the optimization form (Bx3 with a,b,c * d) and parses into
    (a,b,c,d) from with (a,b,c) normal facing "up in the camera frame and d the offset.
    '''
    floor_offset = torch.norm(floor_plane, dim=1, keepdim=True)
    floor_normal = floor_plane / floor_offset
    
    # in camera system -y is up, so floor plane normal y component should never be positive
    #       (assuming the camera is not sideways or upside down)
    neg_mask = floor_normal[:,1:2] > 0.0
    floor_normal = torch.where(neg_mask.expand_as(floor_normal), -floor_normal, floor_normal)
    floor_offset = torch.where(neg_mask, -floor_offset, floor_offset)
    floor_plane_4d = torch.cat([floor_normal, floor_offset], dim=1)

    return floor_plane_4d

def load_planercnn_res(res_path):
    '''
    Given a directory containing PlaneRCNN plane detection results, loads the first image result 
    and heuristically finds and returns the floor plane.
    '''
    planes_param_path = glob.glob(res_path + '/*_plane_parameters_*.npy')[0]
    planes_mask_path = glob.glob(res_path + '/*_plane_masks_*.npy')[0]
    planes_params = np.load(planes_param_path)
    planes_masks = np.load(planes_mask_path)
    
    # heuristically determine the ground plane
    #   the plane with the most labeled pixels in the bottom N rows
    nrows = 10
    label_count = np.sum(planes_masks[:, -nrows:, :], axis=(1, 2))
    floor_idx = np.argmax(label_count)
    valid_floor = False
    floor_plane = None
    while not valid_floor:
        # loop until we find a plane with many pixels on the bottom
        #       and doesn't face in the complete wrong direction
        # we assume the y component is larger than any others
        # i.e. that the floor is not > 45 degrees relative rotation from the camera
        floor_plane = planes_params[floor_idx]
        # transform to our system
        floor_plane = np.array([floor_plane[0], -floor_plane[2], floor_plane[1]])
        # determine 4D parameterization
        # for this data we know y should always be negative
        floor_offset = np.linalg.norm(floor_plane)
        floor_normal = floor_plane / floor_offset
        if floor_normal[1] > 0.0:
            floor_offset *= -1.0
            floor_normal *= -1.0
        a, b, c = floor_normal
        d = floor_offset
        floor_plane = np.array([a, b, c, d])

        valid_floor = np.abs(b) > np.abs(a) and np.abs(b) > np.abs(c)
        if not valid_floor:
            label_count[floor_idx] = 0
            floor_idx = np.argmax(label_count)

    return floor_plane


def compute_cam2prior(floor_plane, trans, root_orient, joints):
    '''
    Computes rotation and translation from the camera frame to the canonical coordinate system
    used by the motion and initial state priors.
    - floor_plane : B x 3
    - trans : B x 3
    - root_orient : B x 3
    - joints : B x J x 3
    '''
    B = floor_plane.size(0)
    if floor_plane.size(1) == 3:
        floor_plane_4d = parse_floor_plane(floor_plane)
    else:
        floor_plane_4d = floor_plane
    floor_normal = floor_plane_4d[:,:3]
    floor_trans, _ = compute_plane_intersection(trans, -floor_normal, floor_plane_4d)

    # compute prior frame axes within the camera frame
    # up is the floor_plane normal
    up_axis = floor_normal
    # right is body -x direction projected to floor plane
    root_orient_mat = batch_rodrigues(root_orient)
    body_right = -root_orient_mat[:, :, 0]
    floor_body_right, s = compute_plane_intersection(trans, body_right, floor_plane_4d)
    right_axis = floor_body_right - floor_trans 
    # body right may not actually intersect - in this case must negate axis because we have the -x
    right_axis = torch.where(s.reshape((B, 1)) < 0, -right_axis, right_axis)
    right_axis = right_axis / torch.norm(right_axis, dim=1, keepdim=True)
    # forward is their cross product
    fwd_axis = torch.cross(up_axis, right_axis)
    fwd_axis = fwd_axis / torch.norm(fwd_axis, dim=1, keepdim=True)

    prior_R = torch.stack([right_axis, fwd_axis, up_axis], dim=2)
    cam2prior_R = prior_R.transpose(2, 1)

    # translation takes translation to origin plus offset to the floor
    cam2prior_t = -trans

    _, s_root = compute_plane_intersection(joints[:,0], -floor_normal, floor_plane_4d)
    root_height = s_root.reshape((B, 1))

    return cam2prior_R, cam2prior_t, root_height

def apply_robust_weighting(res, robust_loss_type='bisquare', robust_tuning_const=4.6851):
    '''
    Returns robustly weighted squared residuals.
    - res : torch.Tensor (B x N), take the MAD over each batch dimension independently.
    '''
    robust_choices = ['none', 'bisquare']
    if robust_loss_type not in robust_choices:
        print('Not a valid robust loss: %s. Please use %s' % (robust_loss_type, str(robust_choices)))
    
    w = None
    detach_res = res.clone().detach() # don't want gradients flowing through the weights to avoid degeneracy
    if robust_loss_type == 'none':
        w = torch.ones_like(detach_res)
    elif robust_loss_type == 'bisquare':
        w = bisquare_robust_weights(detach_res, tune_const=robust_tuning_const)

    # apply weights to squared residuals
    weighted_sqr_res = w * (res**2)
    return weighted_sqr_res, w

def robust_std(res):
    ''' 
    Compute robust estimate of standarad deviation using median absolute deviation (MAD)
    of the given residuals independently over each batch dimension.

    - res : (B x N)

    Returns:
    - std : B x 1
    '''
    B = res.size(0)
    med = torch.median(res, dim=-1)[0].reshape((B,1))
    abs_dev = torch.abs(res - med)
    MAD = torch.median(abs_dev, dim=-1)[0].reshape((B, 1))
    std = MAD / 0.67449
    return std

def bisquare_robust_weights(res, tune_const=4.6851):
    '''
    Bisquare (Tukey) loss.
    See https://www.mathworks.com/help/curvefit/least-squares-fitting.html

    - residuals
    '''
    # print(res.size())
    norm_res = res / (robust_std(res) * tune_const)
    # NOTE: this should use absolute value, it's ok right now since only used for 3d point cloud residuals
        #   which are guaranteed positive, but generally this won't work)
    outlier_mask = norm_res >= 1.0

    # print(torch.sum(outlier_mask))
    # print('Outlier frac: %f' % (float(torch.sum(outlier_mask)) / res.size(1)))

    w = (1.0 - norm_res**2)**2
    w[outlier_mask] = 0.0

    return w

def gmof(res, sigma):
    """
    Geman-McClure error function
    - residual
    - sigma scaling factor
    """
    x_squared = res ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def log_cur_stats(stats_dict, loss, iter=None):
    Logger.log('LOSS: %f' % (loss.cpu().item()))
    print('----')
    for k, v in stats_dict.items():
        if isinstance(v, float):
            Logger.log('%s: %f' % (k, v))
        else:
            Logger.log('%s: %f' % (k, v.cpu().item()))
    if iter is not None:
        print('======= iter %d =======' % (int(iter)))
    else:
        print('========')

def save_optim_result(cur_res_out_paths, optim_result, per_stage_results, gt_data, observed_data, data_type,
                      optim_floor=True,
                      obs_img_paths=None,
                      obs_mask_paths=None):
    # final optim results
    res_betas = optim_result['betas'].cpu().numpy()
    res_trans = optim_result['trans'].cpu().numpy()
    res_root_orient = optim_result['root_orient'].cpu().numpy()
    res_body_pose = optim_result['pose_body'].cpu().numpy()
    res_hand_pose = optim_result['pose_hand'].cpu().numpy()
    res_contacts = None
    res_floor_plane = None
    if 'contacts' in optim_result:
        res_contacts = optim_result['contacts'].cpu().numpy()
    if 'floor_plane' in optim_result:
        res_floor_plane = optim_result['floor_plane'].cpu().numpy()
    for bidx, cur_res_out_path in enumerate(cur_res_out_paths):
        cur_res_out_path = os.path.join(cur_res_out_path, 'stage2_results.npz')
        save_dict = { 
            'betas' : res_betas[bidx],
            'trans' : res_trans[bidx],
            'root_orient' : res_root_orient[bidx],
            'pose_body' : res_body_pose[bidx],
        }
        if res_hand_pose is not None:
            save_dict['pose_hand'] = res_hand_pose[bidx]
        if res_contacts is not None:
            save_dict['contacts'] = res_contacts[bidx]
        if res_floor_plane is not None:
            save_dict['floor_plane'] = res_floor_plane[bidx]
        np.savez(cur_res_out_path, **save_dict)

    # in prior coordinate frame
    if 'stage3' in per_stage_results and optim_floor:
        res_trans = per_stage_results['stage3']['prior_trans'].detach().cpu().numpy()
        res_root_orient = per_stage_results['stage3']['prior_root_orient'].detach().cpu().numpy()
        for bidx, cur_res_out_path in enumerate(cur_res_out_paths):
            cur_res_out_path = os.path.join(cur_res_out_path, 'stage3_results_prior.npz')
            save_dict = { 
                'betas' : res_betas[bidx],
                'trans' : res_trans[bidx],
                'root_orient' : res_root_orient[bidx],
                'pose_body' : res_body_pose[bidx]
            }
            if res_contacts is not None:
                save_dict['contacts'] = res_contacts[bidx]
            np.savez(cur_res_out_path, **save_dict)

    # ground truth
    save_gt = 'betas' in gt_data and \
                'trans' in gt_data and \
                'root_orient' in gt_data and \
                'pose_body' in gt_data
    if save_gt:
        gt_betas = gt_data['betas'].cpu().numpy()
        if data_type not in ['PROX-RGB', 'PROX-RGBD']:
            gt_betas = gt_betas[:,0] # only need frame 1 for e.g. 3d data since it's the same over time.
        gt_trans = gt_data['trans'].cpu().numpy()
        gt_root_orient = gt_data['root_orient'].cpu().numpy()
        gt_body_pose = gt_data['pose_body'].cpu().numpy()
        gt_contacts = None
        if 'contacts' in gt_data:
            gt_contacts = gt_data['contacts'].cpu().numpy()
        cam_mat = None
        if 'cam_matx' in gt_data:
            cam_mat = gt_data['cam_matx'].cpu().numpy()
        for bidx, cur_res_out_path in enumerate(cur_res_out_paths):
            gt_res_name = 'proxd_results.npz' if data_type in ['PROX-RGB', 'PROX-RGBD'] else 'gt_results.npz'
            cur_gt_out_path = os.path.join(cur_res_out_path, gt_res_name)
            save_dict = { 
                'betas' : gt_betas[bidx],
                'trans' : gt_trans[bidx],
                'root_orient' : gt_root_orient[bidx],
                'pose_body' : gt_body_pose[bidx]
            }
            if gt_contacts is not None:
                save_dict['contacts'] = gt_contacts[bidx]
            if cam_mat is not None:
                save_dict['cam_mtx'] = cam_mat[bidx]
            np.savez(cur_gt_out_path, **save_dict)

            # if these are proxd results also need to save a GT with cam matrix
            if data_type in ['PROX-RGB', 'PROX-RGBD']:
                cur_gt_out_path = os.path.join(cur_res_out_path, 'gt_results.npz')
                np.savez(cur_gt_out_path, cam_mtx=cam_mat[bidx])

    elif 'joints3d' in gt_data:
        # don't have smpl params, but have 3D joints (e.g. imapper)
        gt_joints = gt_data['joints3d'].cpu().numpy()
        cam_mat = occlusions = None
        if 'cam_matx' in gt_data:
            cam_mat = gt_data['cam_matx'].cpu().numpy()
        if 'occlusions' in gt_data:
            occlusions = gt_data['occlusions'].cpu().numpy()
        for bidx, cur_res_out_path in enumerate(cur_res_out_paths):
            cur_res_out_path = os.path.join(cur_res_out_path, 'gt_results.npz')
            save_dict = { 
                'joints3d' : gt_joints[bidx]
            }
            if cam_mat is not None:
                save_dict['cam_mtx'] = cam_mat[bidx]
            if occlusions is not None:
                save_dict['occlusions'] = occlusions[bidx]
            np.savez(cur_res_out_path, **save_dict)
    elif 'cam_matx' in gt_data:
        # need the intrinsics even if we have nothing else
        cam_mat = gt_data['cam_matx'].cpu().numpy()
        for bidx, cur_res_out_path in enumerate(cur_res_out_paths):
            cur_res_out_path = os.path.join(cur_res_out_path, 'gt_results.npz')
            save_dict = { 
                'cam_mtx' : cam_mat[bidx]
            }
            np.savez(cur_res_out_path, **save_dict)

    # observations
    obs_out = {k : v.cpu().numpy() for k, v in observed_data.items() if k != 'prev_batch_overlap_res'}
    for bidx, cur_res_out_path in enumerate(cur_res_out_paths):
        obs_out_path = os.path.join(cur_res_out_path, 'observations.npz')
        cur_obs_out = {k : v[bidx] for k, v in obs_out.items() if k not in ['RGB']}
        if obs_img_paths is not None:
            cur_obs_out['img_paths'] = [frame_tup[bidx] for frame_tup in obs_img_paths]
            # print(cur_obs_out['img_paths'])
        if obs_mask_paths is not None:
            cur_obs_out['mask_paths'] = [frame_tup[bidx] for frame_tup in obs_mask_paths]
        np.savez(obs_out_path, **cur_obs_out)    


def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    Adapted from https://github.com/mkocabas/VIBE/blob/master/lib/models/spin.py
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs, 2): Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length[:,0]
    K[:,1,1] = focal_length[:,1]
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

#
# The following 2 functions are borrowed from VPoser (https://github.com/nghorbani/human_body_prior).
# See their license for usage restrictions.
#
def expid2model(expr_dir):
    from configer import Configer

    if not os.path.exists(expr_dir): raise ValueError('Could not find the experiment directory: %s' % expr_dir)

    best_model_fname = sorted(glob.glob(os.path.join(expr_dir, 'snapshots', '*.pt')), key=os.path.getmtime)[-1]
    try_num = os.path.basename(best_model_fname).split('_')[0]

    print(('Found Trained Model: %s' % best_model_fname))

    default_ps_fname = glob.glob(os.path.join(expr_dir,'*.ini'))[0]
    if not os.path.exists(
        default_ps_fname): raise ValueError('Could not find the appropriate vposer_settings: %s' % default_ps_fname)
    ps = Configer(default_ps_fname=default_ps_fname, work_dir = expr_dir, best_model_fname=best_model_fname)

    return ps, best_model_fname

def load_vposer(expr_dir, vp_model='snapshot'):
    '''
    :param expr_dir:
    :param vp_model: either 'snapshot' to use the experiment folder's code or a VPoser imported module, e.g.
    from human_body_prior.train.vposer_smpl import VPoser, then pass VPoser to this function
    :param if True will load the model definition used for training, and not the one in current repository
    :return:
    '''
    import importlib
    import os
    import torch

    ps, trained_model_fname = expid2model(expr_dir)
    if vp_model == 'snapshot':

        vposer_path = sorted(glob.glob(os.path.join(expr_dir, 'vposer_*.py')), key=os.path.getmtime)[-1]

        spec = importlib.util.spec_from_file_location('VPoser', vposer_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        vposer_pt = getattr(module, 'VPoser')(num_neurons=ps.num_neurons, latentD=ps.latentD, data_shape=ps.data_shape)
    else:
        vposer_pt = vp_model(num_neurons=ps.num_neurons, latentD=ps.latentD, data_shape=ps.data_shape)

    vposer_pt.load_state_dict(torch.load(trained_model_fname, map_location='cpu'))
    vposer_pt.eval()

    return vposer_pt, ps

if __name__=='__main__':
    test = 0