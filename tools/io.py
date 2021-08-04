import os, numpy as np
from pypcd import pypcd
from pyquaternion import Quaternion     #　四元数的库


def load_pose(pose_path, offset_x = 0, offset_y = 0):
    data = np.fromfile(pose_path,  sep=' ')
    trans = data[2:5]
    trans[0] -= offset_x
    trans[1] -= offset_y
    x,y,z,w = data[5:9]

    quat = Quaternion(w,x,y,z)
    rotation = quat.rotation_matrix

    T = np.zeros((4,4), np.float32)
    T[:3, :3] = rotation
    T[:3, 3] = trans
    T[3, 3] = 1

    return T


def load_det_labels_from_single_frame(result_frame_path):
    '''Reading labels from a txt file
        
        Input: (type, -1, -1, alpha, [bbox2d_trk x 4], h, w, l, x, y, z, theta, conf]) (CenterPoint format)
        
        Return: (type, h, w, l, x, y, z, theta, conf)
    '''

    with open(result_frame_path, 'r') as F:
        data = F.readlines()
    
    labels, boxes = [], []
    for line in data:
        label = line.split()
        del label[1:8]
        
        labels.append(label[0])
        boxes.append(list(map(float, label[1:])))

    return labels, boxes

def load_trk_labels_from_single_sequence(result_sequence_path):
    '''Reading labels from a txt file
        
        Input: (frame_id, object_id, type, 0, 0, [bbox2d_trk x 4], conf, 
                h. w, l x, y, z, theta,]) (CenterPoint format)
            
            (h, w, l, x, y, z, theta) are in camera coordinate follwing KITTI convention
        
        Return: [(object_id, type, h, w, l, x, y, z, theta, conf) x frames]

        bboxes: [[], [], []]
        ids: [1,2,3]
    '''

    with open(result_sequence_path, 'r') as F:
        data = F.readlines()
    
    sequence_info = {}

    for line in data:
        label = line.split()
        del label[3:10]
        
        # !!! the frame_id range is [1, N] in original data folder, but is [0, N-1] in trk sequence result.
        frame_id = str(int(label[0]) + 1)
        frame_info = sequence_info.get(frame_id, {})
        labels = frame_info.get("labels", [])
        boxes = frame_info.get("boxes", [])
        ids = frame_info.get("ids", [])
        
        labels.append(label[2])

        # change sequence order from (h, w, l) to (l, h, w) to draw points
        box = (label[5:6] + label[3:5]) + label[6:]
        boxes.append(list(map(float, box)))

        ids.append(int(label[1]))

        frame_info = {"labels": labels, "boxes": boxes, "ids": ids}

        sequence_info[frame_id] = frame_info

    return sequence_info


def load_pcd_ACG(path, tries=2, num_point_feature=4, painted=False):
    try:
        assert path.exists()
        pc = pypcd.PointCloud.from_path(path)
        x = pc.pc_data['x']
        y = pc.pc_data['y']
        z = pc.pc_data['z']
        if 'intensity' in pc.fields:
            intensity = pc.pc_data['intensity'].astype(np.float32) / 255.0
        else:
            intensity = np.ones_like(x)
        pointcloud = np.vstack((x, y, z, intensity))
        pointcloud = pointcloud.transpose(1, 0)
        pointcloud = pointcloud.reshape(-1, 4).astype(np.float32)
        return pointcloud
    except:
        print("path: {} not exists".format(path))
