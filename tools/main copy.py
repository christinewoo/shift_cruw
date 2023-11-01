import os
import os.path as osp
import argparse
import datetime

parser = argparse.ArgumentParser(
    description='Post-processing')
parser.add_argument('--gt_path', default='', help='ground truth directory')
parser.add_argument('--pd_path', default='', help='prediction directory')
parser.add_argument('--tags', default='lpcg_3159', help='details of prediction')
args = parser.parse_args()

### Prep the txt ###
from object3d_kitti import get_objects_from_label
def get_label(path, frame):
    label_file = osp.join(path, frame)
    assert osp.exists(label_file)
    return get_objects_from_label(label_file)

from natsort import natsorted
from tqdm import tqdm
def txt_to_lists(gt_path, pd_path):
    gt_list = natsorted(os.listdir(gt_path))
    pd_list = natsorted(os.listdir(pd_path))

    # DEBUG for val
    gt_path = '/mnt/disk2/Data/KITTI/KITTI3D/training/label_2'

    assert len(gt_list)==len(pd_list)
    
    datum = {}
    datum['gt'] = {}
    datum['pd'] = {}
    for frame in tqdm(gt_list, desc='Datum'):
        datum['gt'].update({frame : get_label(gt_path, frame)})
        datum['pd'].update({frame : get_label(pd_path, frame)})
    
    return datum


### Match ###
import math
def get_dist(gt, dt, metric=1):
    # Metric #1: dist = sqrt[(xg - xd)^2 + (yg - yd)^2 + (yg - yd)^2]
    sum_ = (gt[0] - dt[0])**2 + (gt[1] - dt[1])**2 + (gt[2] - dt[2])**2
    return math.sqrt(sum_)  

#     1    0      Z(l) 
#   2    3       /__X(w)
#     5    4     |
#   6    7       Y(h) - bottom   
def iou(gt, pd):
    # filter out unwanted
    if gt.cls_type != 'Car':
        return 800 # arbitrary small
    
    # X and Y [2, 3, 6, 7]
    x_g1, x_g2, y_g1, y_g2 = \
        min(gt.corners3d[6][0], gt.corners3d[3][0]), max(gt.corners3d[6][0], gt.corners3d[3][0]), \
        min(gt.corners3d[6][1], gt.corners3d[3][1]), max(gt.corners3d[6][1], gt.corners3d[3][1])
    x_p1, x_p2, y_p1, y_p2 = \
        min(pd.corners3d[6][0], pd.corners3d[3][0]), max(pd.corners3d[6][0], pd.corners3d[3][0]), \
        min(pd.corners3d[6][1], pd.corners3d[3][1]), max(pd.corners3d[6][1], pd.corners3d[3][1])
    area_g = (x_g2 - x_g1) * (y_g2 - y_g1)
    area_p = (x_p2 - x_p1) * (y_p2 - y_p1)
    # Find enclosed intersection
    x_i1, x_i2, y_i1, y_i2 = max(x_p1, x_g1), min(x_p2, x_g2), max(y_p1, y_g1), min(y_p2, y_g2)
    area_i = np.clip((x_i2 - x_i1) * (y_i2 - y_i1), a_min=0, a_max=None)
    area_u = area_g + area_p - area_i
    xy_iou = area_i / area_u
    
    # X and Z -- 2, 0
    min_zg, max_zg = min(gt.corners3d[2][2], gt.corners3d[0][2]), max(gt.corners3d[2][2], gt.corners3d[0][2])
    min_zp, max_zp = min(pd.corners3d[2][2], pd.corners3d[0][2]), max(pd.corners3d[2][2], pd.corners3d[0][2])
    z_i1, z_i2 = max(min_zg, min_zp), min(max_zg, max_zp)
    z_diff = np.clip((z_i2 - z_i1), a_min=0, a_max=None) 
    
    volume_i = xy_iou * z_diff
    gt_vol = gt.h * gt.w * gt.l
    pd_vol = pd.h * pd.w * pd.l
    volume_u =  gt_vol + pd_vol - volume_i + 1e-5
    
    iou_3d = volume_i / volume_u
    
    print(get_dist(gt.loc, pd.loc))
    
    
    if iou_3d <= 0:
        print(300)
        return 300
    else:
        print(iou_3d* 100)
        return iou_3d * 100
    
    

from lap import lapjv
import numpy as np
import copy
def get_gt_dt_pairs(datum, frame, cls='Car'):
    # Get gt, dt objects
    gt_objs = datum['gt'][frame]
    dt_objs = datum['pd'][frame]
    kept_dt_ids = []
    
    if len(gt_objs) == 0 or len(dt_objs) == 0:
        return [], [-1]

    # Populate the distance(cost) matrix
    n, m = len(gt_objs), len(dt_objs)
    dist_mat = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            # dist_mat[i][j] = get_dist(gt_objs[i].loc, dt_objs[j].loc, 1)
            dist_mat[i][j] = iou(gt_objs[i], dt_objs[j])
    
    # REF: https://github.com/gatagat/lap/blob/master/lap/_lapjv.pyx
    # Hungarian assignment (x specifies the col_dt to which row_gt is assigned)
    cost, x, y = lapjv(dist_mat, extend_cost=True, cost_limit=100)
    
    # Match the pairs and write to cur_idx_matches list
    cur_idx_matches = [] # gt, dt
    for i in range(n):
        matched_dt_idx = x[i]
        if matched_dt_idx != -1 and dist_mat[i][matched_dt_idx] > 0:
            cur_idx_matches.append([gt_objs[i], dt_objs[matched_dt_idx]])
            kept_dt_ids.append(matched_dt_idx) #remove matched items from pd_list
    
    return cur_idx_matches, kept_dt_ids

### Write filtered txt results ###
def get_obj_str(obj):
    str_ = ' '.join(str(param) for param in obj)
    return str_    

def main():
    # Set up output directory
    output_path = os.path.join('./outputs', args.tags)
#datetime.datetime.now().strftime('%Y%m%d_%H%M')
    os.makedirs(output_path, exist_ok=True)
    
    # Log
    print(f'Results for {args.tags}')
    print(f'Results saved in {output_path}')
    
    # Prep txt into lists
    datum = txt_to_lists(args.gt_path, args.pd_path)
    invalids = []
    total_removed = 0
    FP_count = 0
    for frame in datum['gt'].keys():
        # For each label find gt_dt pair list of (gt, dt)
        cur_idx_pairs, kept_dt_ids = get_gt_dt_pairs(datum, frame, cls='Car')
        # DEBUG: count invalid pairs filtered 
        if cur_idx_pairs == None:
            invalids.append(frame)
            continue
        # COUNT number of FP
        for i, obj in enumerate(datum['pd'][frame]):
            if len(kept_dt_ids) == 0:
                break
            if i in kept_dt_ids:
                continue
            total_removed += 1
            if obj.score >= 0.5:
                FP_count += 1
        # write current idx to txt file
        with open(osp.join(output_path, frame), 'w') as f:
            for pair in cur_idx_pairs:
                f.write(pair[1].src)
                # dt_str = get_obj_str(pair[1])
                # f.write(f'{dt_str}\n')
            f.close()
        # break
    
    print(f'Removed total: {total_removed}, FP count: {FP_count}')
    

if __name__ == '__main__':
    main()
    
# Virconv Test: /home/ipl-pc/VirConv/output/models/kitti/VirConv-S-test/default/eval/epoch_2/test/default/final_result/data/
# Virvonv Val: /home/ipl-pc/VirConv/output/models/kitti/VirConv-S/default/eval/epoch_2/val/default/final_result/data/

#/home/ipl-pc/cmkd/output/cfgs/kitti_models/CMKD/CMKD-scd/cmkd_kitti_R50_scd_V2_2161/default/eval/epoch_16/val/default/final_result/data

####
        # write current idx to txt file
        # with open(osp.join(output_path, frame), 'w') as f:
        #     for pair in cur_idx_pairs:
        #         gt_str = get_obj_str(pair[0])
        #         dt_str = get_obj_str(pair[1])
        #         f.write(f'{gt_str}%{dt_str}\n')
        #     f.close()