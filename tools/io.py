import os, numpy as np

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