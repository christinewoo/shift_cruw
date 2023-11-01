import os
import os.path as osp
from natsort import natsorted
from tqdm import tqdm
import numpy as np
import math

## Goes through dt list and sort dt & gt into seq lists in gt, dt, dicts
def load_seq_idxs(dt_path):
    dts = []
    pred_list = natsorted(os.listdir(dt_path))  #2021_1120_1616-001282.txt
    for pred in pred_list:
        seq_num = pred.split('-')[0]
        if seq_num == seq:
            dts.append(pred)
    return dts

def get_obj_str(obj):
    obj[11] = str(obj[11].round(2))
    obj[12] = str(obj[12].round(2))
    obj[13] = str(obj[13].round(2))
    str_ = ' '.join(obj)
    return str_ + '\n'

def modify_objs_in_idx(path, mean_ratio, cls='Car'):
    assert(osp.exists(path))
    obj_strs = []
    with open(path) as f:
        for line in f:
            obj = line[:-1].strip().split(' ')
            if obj[0] == cls:
                obj[11] = float(obj[11]) * mean_ratio[0]
                obj[12] = float(obj[12]) * mean_ratio[1]
                obj[13] = float(obj[13]) * mean_ratio[2]
                obj_strs.append(get_obj_str(obj))
            else:
                obj_strs.append(line)
        f.close()
    return obj_strs

def shift_and_save_xyz(seq_mean_ratio, dt_path, seq_dts, save_dir, cls='Car'):
    for idx in tqdm(seq_dts):
        obj_strs = modify_objs_in_idx(osp.join(dt_path, idx), seq_mean_ratio, cls)
        with open(osp.join(save_dir, idx), 'w') as f:
            for str in obj_strs:
                f.write(str)
            f.close()
        


## Given a sequence of matched pairs, get ratio along each x, y, z, diff
## betweeen dt and gt. Returns the averaged ratio
def get_seq_mean_ratio(pairs_dir, inf_name, seq):
    seq_dir = osp.join(pairs_dir, inf_name, seq)
    assert osp.exists(seq_dir)
    seq_list = natsorted(os.listdir(seq_dir))
    xyz_ratio = []
    for id in seq_list:
        with open(osp.join(seq_dir, id)) as f:
            for line in f: #x, y, z, theta, h, w, l
                gt, dt = line[:-2].split('%')
                gt = [float(param) for param in gt.split(' ')]
                dt = [float(param) for param in dt.split(' ')]
                dx, dy, dz = gt[0] / dt[0], gt[1] / dt[1], gt[2] / dt[2]
                xyz_ratio.append(np.array([dx, dy, dz]))
    xyz_ratio = np.array(xyz_ratio)
    xyz_mean_ratio = np.mean(xyz_ratio, axis=0)
    return xyz_mean_ratio  

if __name__ == '__main__':
    # Define path to matched pairs
    dt_path = '/mnt/disk2/christine/cruw_twcc/mono3dcft_swin_T_224_1k_4_0.0001_CRUW_20230429_1656/outputs/data'
    pairs_dir = '/mnt/nas_cruw/tmp_woo/shift_results/'
    inf_name = 'mono3dcft_swin_T_224_1k_4_0.0001_CRUW_20230429_1656' 
    
    # Create save path folder
    save_root = '/mnt/nas_cruw/tmp_woo/shifted'
    save_dir = osp.join(save_root, inf_name)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    print(f'Results saved in {save_dir}')

    # Load in the matched pairs for seq
    seq = '2021_1120_1616'
    seq_mean_ratio = get_seq_mean_ratio(pairs_dir, inf_name, seq)
    #[0.77892554 1.10289054 0.99000418]
    seq_dts = load_seq_idxs(dt_path)
    # create folder to hold modified txt per seq
    seq_save_dir = osp.join(save_dir, seq)
    if not osp.exists(seq_save_dir):
        os.makedirs(seq_save_dir)
    shift_and_save_xyz(seq_mean_ratio, dt_path, seq_dts, seq_save_dir)