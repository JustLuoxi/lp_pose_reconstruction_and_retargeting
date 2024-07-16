# Author: Xi Luo
# Email: sunshine.just@outlook.com
# SMPLH motion to FBX animation
# Heavily inspired by and borrowed from  https://github.com/softcat477/SMPL-to-FBX

from FbxReadWriter import FbxReadWrite
from SMPLXObject import SMPLXObjects
import argparse
import tqdm

import os
cur_file_path = os.path.dirname(os.path.realpath(__file__))

def getArg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_motion_base', type=str, required=True)
    parser.add_argument('--fbx_source_path', type=str, required=True)
    parser.add_argument('--output_base', type=str, required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = getArg()
    input_motion_base = args.input_motion_base
    fbx_source_path = args.fbx_source_path
    output_base = args.output_base

    # get the smplx objects and its parameters
    smplObjects = SMPLXObjects(input_motion_base)
    
    for pkl_name, smpl_params in tqdm.tqdm(smplObjects):
        try:
            fbx_path = os.path.join(cur_file_path,fbx_source_path)
            fbxReadWrite = FbxReadWrite(fbx_path)
            fbxReadWrite.addAnimation(pkl_name, smpl_params)
            fbxReadWrite.writeFbx(output_base, pkl_name)
        except Exception as e:
            fbxReadWrite.destroy()
            print ("- - Distroy")
            raise e
        finally:
            fbxReadWrite.destroy()

