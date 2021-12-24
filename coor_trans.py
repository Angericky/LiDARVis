import os
import numpy as np
from pathlib import Path
import copy 
import re
import multiprocessing
import argparse
# from mayavi import mlab
import cv2

import tools.io as io
from tools.trans import trans_based_on_pose
from tools.box import boxes_to_corners_3d
from tools import box as V
from draw_bev import convert_lidar_coords_to_image_coords, draw_boxes_on_bev


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmt", choices=["xyz", "npy"], default="npy")
    parser.add_argument("--save_pc", action="store_true")
    parser.add_argument("--save_box", action="store_true")
    parser.add_argument("--save_img", action="store_true")
    # local_to_global: Input order: one frame with all classes
    # cls, -1, -1, 0, 0, 0, 0, 0
    # l, w, h, x, y, z, yaw, score
    # global_to_local: Input order: one sequence in one classes
    # 0, 0, 0, 0, 0, 0, score, h, w, l, x, y, z, yaw
    parser.add_argument("--trans", choices=["local_to_global", "global_to_local"], default="local_to_global")

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

 
def read_and_trans_pcd(sweep_name, base_pose, sweep_pose, data_folder):
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

    sweep_trans_pc = trans_based_on_pose(sweep_pc, sweep_pose, base_pose)

    sweep_trans_pc[:,3] = intensity

    return sweep_trans_pc


def run(sweep_name, base_pose, pose_dir, data_folder, output_folder, fmt):
    sweep_pose = io.load_pose(os.path.join(pose_dir, sweep_name.split('.')[0] + ".txt"))    # (4, 4)
    sweep_trans_pc = read_and_trans_pcd(sweep_name, base_pose, sweep_pose, data_folder)

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
    pc_output_folder = "data/pcd_transform_xyz"
    proposed_result_folder = "data/final_result_det"
    save_img_folder = "data/trans_vis"
    box_output_folder = "data/result_det_trans"
    
    if not os.path.exists(pc_output_folder):
        os.mkdir(pc_output_folder)

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
            if sequence == '441531':
                base_sweep_id = 241
                base_pose = io.load_pose(os.path.join(pose_dir, "{}_{}.txt".format(sequence, base_sweep_id)))
                sequence_sweep_names = list(filter(lambda x: re.match('{}_*'.format(sequence), x) != None, data_sweep_names))  # 生成新列表
                print(sequence_sweep_names)
                import pdb
                pdb.set_trace()
                # pool = multiprocessing.Pool(20)
                # for sweep_name in sequence_sweep_names:
                #     pool.apply_async(func=run, args=(sweep_name, base_pose, pose_dir, data_folder, pc_output_folder, args.fmt))
                # pool.close()
                # pool.join()

                for sweep_name in sequence_sweep_names:
                    run(sweep_name, base_pose, pose_dir, data_folder, pc_output_folder, args.fmt)


    if args.save_box:
        # transform and save boxes by base ego pose 

        result_infos = load_det_result(proposed_result_folder)

        if not os.path.exists(box_output_folder):
            os.mkdir(box_output_folder)

        # 'boxes': (M, 8), float, (h, w, l, x, y, z, theta, conf) 
        for sweep_id in result_infos.keys():
            pose_path = os.path.join(pose_dir, sweep_id + ".txt")
            sweep_pose = io.load_pose(pose_path)    # (4, 4)
            boxes = np.array(result_infos[sweep_id]['boxes'])  # (M, 8)

            import pdb
            pdb.set_trace()
            boxes_to_draw = np.concatenate((boxes[:, 3:6], boxes[:, :3], boxes[:, 6:7]), axis=1)
            
            centers = boxes_to_draw[:, :3]
            centers_all = centers.reshape(-1, 3)
            dim = np.ones((centers.shape[0], 1))
            centers_4dim = np.concatenate((centers_all, dim), axis=1)

            sequence = sweep_id.split("_")[0]
            base_pose_path = os.path.join(pose_dir, "{}_{}.txt".format(sequence, base_sweep_id))
            
            if args.trans == "local_to_global":
                base_pose = io.load_pose(base_pose_path)
                ego_pose = np.fromfile(pose_path,  sep=' ')
            elif args.trans == "global_to_local":
                ego_pose = io.load_pose(base_pose_path)
                base_pose = np.fromfile(pose_path,  sep=' ')

            centers_trans = trans_based_on_pose(centers_4dim, sweep_pose, base_pose)
            centers_trans = centers_trans[:, :3]

            x,y,z,w = ego_pose[5:9]

            yaw = np.arcsin(2*(w*y - z*x))
            boxes_trans = np.zeros_like(boxes_to_draw)
            boxes_trans[:, :3] = centers_trans
            boxes_trans[:, 3:6] = boxes_to_draw[:, 3:6] # copy original size dim
            boxes_trans[:, 6] = boxes_to_draw[:, 6] 

            boxes_trans = np.concatenate((boxes_trans[:, 3:6], boxes_trans[:, :3], boxes_trans[:, 6:7]), axis=1)

            # save as det pred format
            output_file = os.path.join(box_output_folder, sweep_id + ".txt")
            with open(output_file, "a+") as f:
                for i in range(boxes_trans.shape[0]):
                    f.write("{} -1 -1 0 0 0 0 {} {} {} {} {} {} {} {}\n".format(
                    result_infos[sweep_id]["labels"][i], *boxes_trans[i], boxes[i, -1]))

            if args.save_img:
                sweep_name = sweep_id + ".pcd"
            
                # read 4-dim points
                sweep_trans_pc = read_and_trans_pcd(sweep_name, base_pose, sweep_pose, data_folder)

                # render bev image
                pc_range = 80
                resolution = 0.1
                rows = int(pc_range * 2 / resolution)

                # generate bev map
                sweep_path = os.path.join(data_folder, sweep_name)
                sweep_pc = io.load_pcd_ACG(Path(sweep_path)) # (N, 4)
                bev = convert_lidar_coords_to_image_coords(sweep_trans_pc, rows, rows, pc_range, resolution)

                # draw boxes
                bev = draw_boxes_on_bev(bev, boxes_trans, pc_range, resolution)

                # cv2.namedWindow('bev', 0)    
                # cv2.resizeWindow('bev', 800, 800)  
                # cv2.imshow("bev", bev)
            
                cv2.imwrite(os.path.join(save_img_folder, sweep_id.split('.')[0] + '.png'), bev)
            # cv2.waitKey()

            # V.draw_scenes(
            #     points=sweep_pc, ref_boxes=boxes_to_draw,
            #     ref_scores=boxes[:, 7], frame_id=int(sweep_name)
            # )
            # mlab.show(stop=True)