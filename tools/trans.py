import numpy as np
 
 
def trans_based_on_pose(pc, T, T_base):
    #pc_new = T_base @ np.linalg.inv(T) @ pc
    #pc_new = np.linalg.inv(T_base) @ T @ pc

    T_offset = np.zeros_like(T)
    T_offset[:3, 3] = T[:3, 3]
    pc_new = np.linalg.inv(T_base - T_offset) @ (T - T_offset) @ pc.T

    #pc_new = np.linalg.inv(T_base - T_offset) @ ((T - T_offset) @ pc)
    return pc_new.T


