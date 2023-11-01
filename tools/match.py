import os
import os.path as osp
from natsort import natsorted
from tqdm import tqdm
import numpy as np
import math
from lap import lapjv

# for each sequence 
def load_pred_old(dt_path, gt_path, cls='Car'):
    print('PRED: ', dt_path) #2021_1120_1616-001282.txt

    dts = {}
    gts = {}

    pred_list = natsorted(os.listdir(dt_path))
    for pred in tqdm(pred_list):
        seq_num = pred.split('-')[0]

        if seq_num != '2021_1120_1616':
            continue

        # get corresponding list for ground truth
        gt_file = osp.join(gt_path, pred)
        assert(osp.exists(gt_file))
        with open(gt_file) as gt_f:
            for line in gt_f:
                line = line.strip().split(' ')
                instance_type, x, y, z, theta = line[0], float(line[11]), float(line[12]), float(line[13]), float(line[14])
                if instance_type == cls:
                    if not seq_num in gts.keys():
                        gts[seq_num] = [[x, y, z, theta]]
                    else:
                        gts[seq_num] += [[x, y, z, theta]]

        # get corresponding list for predicted detection
        dt_file = osp.join(dt_path, pred)
        with open(dt_file) as dt_f:
            for line in dt_f:
                line = line.strip().split(' ')
                instance_type, x, y, z, theta = line[0], float(line[11]), float(line[12]), float(line[13]), float(line[14])
                if instance_type == cls:
                    if not seq_num in dts.keys():
                        dts[seq_num] = [[x, y, z, theta]]
                    else:
                        dts[seq_num] += [[x, y, z, theta]]
        
    return dts, gts

### Goes through dt list and sort dt & gt into seq lists in gt, dt, dicts
def load_idxs(dt_path):
    print('PRED: ', dt_path.split('/')[-3])
    dts = {}
    pred_list = natsorted(os.listdir(dt_path))  #2021_1120_1616-001282.txt
    for pred in tqdm(pred_list):
        seq_num = pred.split('-')[0]
        # put the idx into gt[seq] = list
        if not seq_num in dts.keys():
            dts[seq_num] = [pred]
        else:
            dts[seq_num] += [pred]
    return dts


def get_objs_in_idx(path, cls='Car'):
    assert(osp.exists(path))
    objs = []
    with open(path) as f:
        for line in f:
            line = line.strip().split(' ')
            instance_type, h, w, l, x, y, z, theta = line[0], float(line[8]), float(line[9]), float(line[10]), float(line[11]), float(line[12]), float(line[13]), float(line[14])
            if instance_type == cls:
                objs.append([x, y, z, theta, h, w, l])
        f.close()
    return objs

def get_dist(gt, dt, metric=1):
    # Metric #1: dist = sqrt[(xg - xd)^2 + (yg - yd)^2 + (yg - yd)^2]
    sum_ = (gt[0] - dt[0])**2 + (gt[1] - dt[1])**2 + (gt[2] - dt[2])**2
    return math.sqrt(sum_)        

def get_gt_dt_pairs(dt_path, gt_path, idx, cls='Car'):
    # Get gt, dt objects
    gt_objs = get_objs_in_idx(osp.join(gt_path, idx), cls)
    dt_objs = get_objs_in_idx(osp.join(dt_path, idx), cls)
    
    if len(gt_objs) == 0 or len(dt_objs) == 0:
        return None

    # Populate the distance(cost) matrix
    n, m = len(gt_objs), len(dt_objs)
    dist_mat = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dist_mat[i][j] = get_dist(gt_objs[i], dt_objs[j], 1)
    
    # REF: https://github.com/gatagat/lap/blob/master/lap/_lapjv.pyx
    # Hungarian assignment (x specifies the col_dt to which row_gt is assigned)
    cost, x, y = lapjv(dist_mat, extend_cost=True, cost_limit=4.0)
    
    # Match the pairs and write to cur_idx_matches list
    cur_idx_matches = [] # gt, dt
    for i in range(n):
        matched_dt_idx = x[i]
        if matched_dt_idx != -1:
            cur_idx_matches.append([gt_objs[i], dt_objs[matched_dt_idx]])
    
    return cur_idx_matches

def get_obj_str(obj):
    str_ = ' '.join(str(param) for param in obj)
    return str_

if __name__ == '__main__':
    # Define processing paths
    dt_path = '/mnt/disk2/christine/cruw_twcc/mono3dcft_swin_T_224_1k_4_0.0001_CRUW_20230429_1656/outputs/data'
    gt_path = '/mnt/nas_cruw/CRUW_2022_eval_test/'
    
    # Create save path folder
    save_root = '/mnt/nas_cruw/tmp_woo/shift_results'
    dir_name = dt_path.split('/')[-3]
    save_dir = osp.join(save_root, dir_name)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    
    # Log
    print(f'Results for {dir_name}')
    print(f'Results saved in {save_dir}')

    # Load in the idxs of labels
    dts = load_idxs(dt_path)
    for seq in dts.keys():
        cur_seq_pairs = {}
        invalid = 0
        # Create seq directory to store match pair txt
        save_path = osp.join(save_dir, seq)
        if not osp.exists(save_path):
            os.makedirs(save_path)
        # Process for each sequuence
        for idx in tqdm(dts[seq]):
            # For each label find gt_dt pair list of (gt, dt)
            cur_idx_pairs = get_gt_dt_pairs(dt_path, gt_path, idx, cls='Car')
            # DEBUG: count invalid pairs filtered 
            if cur_idx_pairs == None:
                invalid += 1
                continue
            # write current idx to txt file
            with open(osp.join(save_path, idx), 'w') as f:
                for pair in cur_idx_pairs:
                    gt_str = get_obj_str(pair[0])
                    dt_str = get_obj_str(pair[1])
                    f.write(f'{gt_str}%{dt_str}\n')
                f.close()
            cur_seq_pairs[idx[:-4]] = cur_idx_pairs
        break
        print(f'{seq} has total of {invalid} filtered.')