import os
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import colorsys

from tools import io


def convert_lidar_coords_to_image_coords(points, H, W, pc_range, resolution):
    bev = np.zeros((H, W, 3)) #
    
    x = (points[:, 0] + pc_range) / resolution
    y = (points[:, 1]+ pc_range) / resolution
    x = x.astype(np.int)
    y = y.astype(np.int)
    out_mask = np.logical_or(np.logical_or(np.logical_or(x >= bev.shape[0], x < 0), y >= bev.shape[1]), y < 0)
    
    x_index = x[~out_mask]
    y_index = y[~out_mask]
    #bev[rows - y - 1, x] = np.array([255, 255, 255]).astype(np.float32)
    bev[x_index, y_index] = np.array([255, 255, 255]).astype(np.float32)
    return bev


# adapted from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def generate_colors():
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    hex_colors = ["D32F2F", "FF4081", "#9C27B0", "#7B1FA2", "#303F9F", 
            "#448AFF", "#03A9F4", "#00BCD4", "#009688", "#4CAF50",
            "#8BC34A", "#CDDC39", "#FFEB3B", "#FFC107", "#FF9800",
            "#FF5722", "#795548", "#607D8B"]
    def Hex_to_RGB(hex):
        r = int(hex[1:3],16)
        g = int(hex[3:5],16)
        b = int(hex[5:7], 16)
        rgb = (r, g, b)
        return rgb
    color_list = []
    for hex_color in hex_colors:
        color_list.append(Hex_to_RGB(hex_color))
    return color_list


def draw_boxes_on_bev(bev, boxes, pc_range, resolution, pred=True, ids=None, color_list=None, show_ids=False):
    for idx, box in enumerate(boxes):
        center = box[3:6]
        length,wid,height = box[:3]
        yaw = box[6]

        ground=center[2] - height/2
        A=np.array([center[0]+length/2*np.cos(yaw),
                    center[1]+length/2*np.sin(yaw),ground])
        B=np.array([center[0]-length/2*np.cos(yaw),
                    center[1]-length/2*np.sin(yaw),ground])
        A1=np.array([A[0]-wid/2*np.sin(yaw),
                    A[1]+wid/2*np.cos(yaw),ground]) 
        A2=np.array([A[0]+wid/2*np.sin(yaw),
                    A[1]-wid/2*np.cos(yaw),ground])
        B1=np.array([B[0]+wid/2*np.sin(yaw),
                    B[1]-wid/2*np.cos(yaw),ground])
        B2=np.array([B[0]-wid/2*np.sin(yaw),
                    B[1]+wid/2*np.cos(yaw),ground])
        rec4points=np.vstack((A1,A2,B1,B2))

        p0 = ((A1+ pc_range) / resolution, (A1 + pc_range) / resolution)[0][:2]
        p1 = ((A2+ pc_range) / resolution, (A2 + pc_range) / resolution)[0][:2]
        p2 = ((B1+ pc_range) / resolution, (B1 + pc_range) / resolution)[0][:2]
        p3 = ((B2+ pc_range) / resolution, (B2 + pc_range) / resolution)[0][:2]

        #rows - y - 1, x

        p0 = (int(p0[1]), int(p0[0]))
        p1 = (int(p1[1]), int(p1[0]))
        p2 = (int(p2[1]), int(p2[0]))
        p3 = (int(p3[1]), int(p3[0]))

        #p0 = (rows - int(p0[0]) - 1,  int(p0[1]))
        #p1 = (rows - int(p1[0]) - 1,  int(p1[1]))
        #p2 = (rows - int(p2[0]) - 1,  int(p2[1]))
        #p3 = (rows - int(p3[0]) - 1,  int(p3[1]))

        if pred == True:
            color = (255, 255, 0)
        else:
            color = (0, 255, 255)
        
        if color_list is not None:
            color = color_list[ids[idx] % len(color_list)] 

        cv2.line(bev, p0, p1, color, 3)
        cv2.line(bev, p1, p2, color, 3)
        cv2.line(bev, p2, p3, color, 3)
        cv2.line(bev, p3, p0, color, 3)

        if show_ids: 
            id = ids[idx]
            cv2.putText(bev, str(id), p0, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 4)
    
    return bev


def mapfusion_draw_det_results(type, show_ids, show_color):
    PCDS_ROOT = "data/point_cloud_data"

    output_path = "data/det_vis"

    result_infos = {}
    if type == "det":
        proposed_result_folder = "data/final_result_det"
        result_frame_names = []
        for frame_file in os.listdir(proposed_result_folder):
            if frame_file.endswith('txt'):
                result_frame_names.append(frame_file)

        result_frame_names.sort()

        for result_frame_name in result_frame_names:
            result_frame_path = os.path.join(proposed_result_folder, result_frame_name)

            labels, boxes = io.load_det_labels_from_single_frame(result_frame_path) # (Type, h, w, l, x, y, z, theta, conf)
            
            frame_id = result_frame_name.split('.')[0]
            result_infos[frame_id] = {"labels": labels, "boxes": boxes}
    
    elif type == "trk":
        proposed_result_folder = "data/results/"
        classes_folder = os.listdir(proposed_result_folder)
        classes_folder.sort()
        
        for class_folder in classes_folder:
            classes_folder_path = os.path.join(proposed_result_folder, class_folder + "/data")
            
            result_sequence_names = []
            for sequence_file in os.listdir(classes_folder_path):
                if sequence_file.endswith('txt'):
                    result_sequence_names.append(sequence_file)
            result_sequence_names.sort()

            for result_sequence_name in result_sequence_names:
                result_sequence_path = os.path.join(classes_folder_path, result_sequence_name)
                
                sequence_info = io.load_trk_labels_from_single_sequence(result_sequence_path) # (Type, h, w, l, x, y, z, theta, conf)
            
                sequence_id = result_sequence_name.split('.')[0]
                
                for frame_id in sequence_info.keys():
                    frame_name = "{}_{}".format(sequence_id, frame_id)
                    frame_info = result_infos.get(frame_name)

                    labels = sequence_info[frame_id]["labels"]
                    boxes = sequence_info[frame_id]["boxes"]
                    ids = sequence_info[frame_id]["ids"]

                    if frame_info is None:
                        frame_info = {}
                        frame_info["labels"] = labels
                        frame_info["boxes"] = boxes
                        frame_info["ids"] = ids
                    else:
                        frame_info["labels"].extend(labels)
                        frame_info["boxes"].extend(boxes)
                        frame_info["ids"].extend(ids)

                    result_infos.update({frame_name: frame_info}) 


    save_img_folder = os.path.join("data/{}_imgs".format(type))
    if not os.path.exists(save_img_folder):
        os.mkdir(save_img_folder)

    keys = list(result_infos.keys())
    keys.sort()

    total_frames_num = len(keys)

    for i in tqdm(range(total_frames_num)):
        result_frame_name = keys[i]

        # load pcd
        frame_id = result_frame_name
        pcd_path = os.path.join(PCDS_ROOT, frame_id.split('.')[0] + '.pcd')
        pcd = io.load_pcd_ACG(Path(pcd_path))
        pcd = np.nan_to_num(pcd)

        # render bev image
        pc_range = 80
        resolution = 0.1
        rows = int(pc_range * 2 / resolution)

        # generate bev map
        bev = convert_lidar_coords_to_image_coords(pcd, rows, rows, pc_range, resolution)

        # draw boxes
        if type == "det":
            boxes = result_infos[result_frame_name]["boxes"]
            bev = draw_boxes_on_bev(bev, boxes, pc_range, resolution)
        elif type == "trk":
            boxes = result_infos[result_frame_name]["boxes"]
            ids = result_infos[result_frame_name]["ids"]
            color_list = generate_colors() if show_color else None
            bev = draw_boxes_on_bev(bev, boxes, pc_range, resolution, ids=ids, color_list=color_list, show_ids=show_ids)

        
        cv2.imwrite(os.path.join(save_img_folder, frame_id.split('.')[0] + '.png'), bev)
        #print("generate frame: {}".format(frame_id.split('.')[0]))

