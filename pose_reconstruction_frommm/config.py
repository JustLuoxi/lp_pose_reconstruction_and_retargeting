# Author: Xi Luo
# Email: sunshine.just@outlook.com
# Configration file for multi-modal based human pose optimization

from pose_fitting.fitting_utils import NSTAGES

import argparse, importlib

class SplitLineParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        return arg_line.split()

def parse_args(argv):
    parser = SplitLineParser(fromfile_prefix_chars='@', allow_abbrev=False)

    # H2TC data options
    parser.add_argument('--data-path', type=str, required=True, help='Path to the data to fit.')
    parser.add_argument('--data-fps', type=int, default=30, help='Sampling rate of the data.')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of sequences to batch together for fitting to data.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', help="Shuffles data.")
    parser.set_defaults(shuffle=False)

    # RGB-specific options
    parser.add_argument('--rgb-seq-len', type=int, default=None, help='If none, fits the whole video at once. If given, is the max number of frames to use when splitting the video into subseqeunces for fitting.')

    # Loss weights
    parser.add_argument('--joint3d-smooth-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='L2 loss on 3D joints differences')
    parser.add_argument('--joint2d-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='L2 loss on 2D reprojection')
    parser.add_argument('--pose-prior-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='likelihood under pose prior')
    parser.add_argument('--shape-prior-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='likelihood under shape prior')
    parser.add_argument('--joint-consistency-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='L2 difference between SMPL and motion prior joints')
    parser.add_argument('--bone-length-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='L2 difference between bone lengths of motion prior joints at consecutive frames.')
    # loss options
    parser.add_argument('--joint2d-sigma', type=float, default=100.0, help='scaling for robust geman-mclure function on joint2d.')

    # smpl model path
    parser.add_argument('--smplh', type=str, default='./body_models/smplh/neutral/model.npz', help='Path to SMPLH model to use for optimization. Currently only SMPL+H is supported.')
    parser.add_argument('--gt-body-type', type=str, default='smplh', choices=['smplh'], help='Which body model to load in for GT data')
    parser.add_argument('--vposer', type=str, default='./checkpoints/vposer_v1_0', help='Path to VPoser checkpoint.')

    # optimization options
    parser.add_argument('--lr', type=float, default=1.0, help='step size during optimization')
    parser.add_argument('--num-iters', type=int, nargs=NSTAGES, default=[30, 80, 70], help='The number of optimization iterations at each stage (3 stages total)')
    parser.add_argument('--lbfgs-max-iter', type=int, default=20, help='The number of max optim iterations per LBFGS step.')

    # options to save/visualize results
    parser.add_argument('--out', type=str, default=None, help='Output path to save fitting results/visualizations to.')

    parser.add_argument('--save-results', dest='save_results', action='store_true', help="Saves final optimized and GT smpl results and observations")
    parser.set_defaults(save_results=False)
    parser.add_argument('--save-stages-results', dest='save_stages_results', action='store_true', help="Saves intermediate optimized results")
    parser.set_defaults(save_stages_results=False)
    
    # options for H2TC and debugging
    parser.add_argument('--img-suffix', type=str, default='jpg', choices=['jpg', 'png'], help="Image suffix, jpg or png")
    parser.add_argument('--catch-throw', type=str, default='catch', choices=['catch', 'throw'])
    parser.add_argument('--num_frame', type=int, default=None, help="set the output number of frames")
    parser.add_argument('--fps', type=int, default=60, help="set the fitting fps of dataset. The dataset capturing fps is 60")
    
    # mmhuamn options
    parser.add_argument('--mmhuman', type=str, required=True, help='Path to the human pose from mmhuman3d.')

    known_args, unknown_args = parser.parse_known_args(argv)

    return known_args