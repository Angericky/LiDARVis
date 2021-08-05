import os
import numpy as np
from pathlib import Path
import copy 
import re
import multiprocessing
import argparse
import tools.io as io
from tools.trans import trans_based_on_pose


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmt", choices=["xyz", "npy"], default="npy")
    parser.add_argument("--save_pc", action="store_true")
    parser.add_argument("--save_box", action="store_true")
    args = parser.parse_args()
    return args


def load_det_result(proposed_result_folder):
    """ 
        Args: 
            str, the result folder where saves labels
        Return: 
            list(result_infos)
                each frame:
                'labels': (M), str
                'boxes': (M, 8), float, (h, w, l, x, y, z, theta, conf)
    """
    result_frame_names = []
    for frame_file in os.listdir(proposed_result_folder):
        if frame_file.endswith('txt'):
            result_frame_names.append(frame_file)
    result_frame_names.sort()

    result_infos = {}
    for result_frame_name in result_frame_names:
        result_frame_path = os.path.join(proposed_result_folder, result_frame_name)

        labels, boxes = io.load_det_labels_from_single_frame(result_frame_path) # (Type, h, w, l, x, y, z, theta, conf)
        
        frame_id = result_frame_name.split('.')[0]
        result_infos[frame_id] = {"labels": labels, "boxes": boxes}
    return result_infos

 
def read_and_trans_pcd(sweep_name, base_pose, pose_dir, data_folder):
    """ 
        Args: 
            str, the sweep path,
            (4, 4), relative ego pose params loaded
        Return:
            (N, 4), points after trans
    """

    sweep_path = os.path.join(data_folder, sweep_name)
    # read 4-dim points
    sweep_pc = io.load_pcd_ACG(Path(sweep_path)) # (N, 4)
        
    # When using R&T matrix to transform point clouds, the 4th dim must be 1, or the result will be distubed.  [x, y, z, intensity] -> [x, y, z, 1]
    intensity = copy.deepcopy(sweep_pc[:, 3])
    sweep_pc[:, 3] = 1

    sweep_pose = io.load_pose(os.path.join(pose_dir, sweep_name.split('.')[0] + ".txt"))    # (4, 4)

    sweep_trans_pc = trans_based_on_pose(sweep_pc, sweep_pose, base_pose)

    sweep_trans_pc[:,3] = intensity

    return sweep_trans_pc


def run(sweep_name, base_pose, pose_dir, data_folder, output_folder, fmt):
    sweep_trans_pc = read_and_trans_pcd(sweep_name, base_pose, pose_dir, data_folder)

    output_path = os.path.join(output_folder, sweep_name.split('.')[0] + ".{}".format(fmt))
    if fmt == "npy":
        np.save(output_path, sweep_trans_pc)
    elif fmt == "xyz":
        np.savetxt(output_path, sweep_trans_pc, fmt='%.18e', delimiter=' ', newline='\n', encoding=None)
    else:
        raise NotImplementedError
    
    print("Sweep: {} has been generated".format(output_path))


if __name__ == '__main__':
    args = parse_args()

    # define args
    data_folder = "data/point_cloud_data"
    pose_dir = "data/point_cloud_pose_data"
    base_sweep_id = 1
    output_folder = "data/pcd_transform"

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # get sweep names and sequence names
    data_sweep_names = []
    sequence_names = set()
    for sweep_filename in os.listdir(data_folder):
        if sweep_filename.endswith('pcd'):
            data_sweep_names.append(sweep_filename)
            sequence_names.add(sweep_filename.split('_')[0])
    data_sweep_names.sort()
    sequence_names = sorted(list(sequence_names))

    if args.save_pc:
        # transform and save points by base ego pose 
        for sequence in sequence_names:
            base_pose = io.load_pose(os.path.join(pose_dir, "{}_{}.txt".format(sequence, base_sweep_id)))
            sequence_sweep_names = list(filter(lambda x: re.match('{}_*'.format(sequence), x) != None, data_sweep_names))  # 生成新列表
            print(sequence_sweep_names)

            pool = multiprocessing.Pool(20)
            for sweep_name in sequence_sweep_names:
                pool.apply_async(func=run, args=(sweep_name, base_pose, pose_dir, data_folder, output_folder, args.fmt))
            pool.close()
            pool.join()

    # if args.save_box:
    # transform and save boxes by base ego pose 
    # proposed_result_folder = "data/final_result_det"
    # result_infos = load_det_result(proposed_result_folder)
    # #'boxes': (M, 8), float, (h, w, l, x, y, z, theta, conf)
    # for sweep_name in result_infos.keys():
    #     boxes_8dim = result_infos[sweep_name]['boxes']
    #     boxes_coords = 
