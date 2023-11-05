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
from object3d_kitti import *
def get_label(path, frame):
    label_file = osp.join(path, frame)
    assert osp.exists(label_file)
    return get_objects_from_label(label_file)

from natsort import natsorted
from tqdm import tqdm
def txt_to_lists(gt_path, pd_path):
    gt_list = natsorted(os.listdir(gt_path))
    pd_list = natsorted(os.listdir(pd_path))

    # # DEBUG for val
    # gt_path = '/mnt/disk2/Data/KITTI/KITTI3D/training/label_2'

    assert len(gt_list)==len(pd_list)
    
    datum = {}
    datum['gt'] = {}
    datum['pd'] = {}
    for frame in tqdm(gt_list, desc='Datum'):
        datum['gt'].update({frame : get_label(gt_path, frame)})
        datum['pd'].update({frame : get_label(pd_path, frame)})
    
    return datum


### Matching cost and Hungarian Assignment ###
import math
from shapely.geometry import Polygon
def get_cost(gt, pd, metric='iou'):
    ### Distance as metric ###
    if metric != 'iou': 
        # dist = sqrt[(xg - xd)^2 + (yg - yd)^2 + (yg - yd)^2]
        sum_ = (gt[0] - pd[0])**2 + (gt[1] - pd[1])**2 + (gt[2] - pd[2])**2
        cost = math.sqrt(sum_)
    
    ### IOU as metric ###
    else: 
        # Filter out unwanted
        if gt.cls_type != 'Car' or pd.cls_type != 'Car':
            return 1
        # Define two polygons
        gt_0 = (gt.corners3d[0][0], gt.corners3d[0][2])
        gt_1 = (gt.corners3d[1][0], gt.corners3d[1][2])
        gt_2 = (gt.corners3d[2][0], gt.corners3d[2][2])
        gt_3 = (gt.corners3d[3][0], gt.corners3d[3][2])
        gt_poly = Polygon([gt_0, gt_1, gt_2, gt_3])
        
        pd_0 = (pd.corners3d[0][0], pd.corners3d[0][2])
        pd_1 = (pd.corners3d[1][0], pd.corners3d[1][2])
        pd_2 = (pd.corners3d[2][0], pd.corners3d[2][2])
        pd_3 = (pd.corners3d[3][0], pd.corners3d[3][2])
        pd_poly = Polygon([pd_0, pd_1, pd_2, pd_3])
        
        iou = gt_poly.intersection(pd_poly).area / gt_poly.union(pd_poly).area
        cost = 1 - iou
    return cost 

def get_iou(gt, pd):
    # Filter out unwanted
    if gt.cls_type != 'Car' or pd.cls_type != 'Car':
        return -2
    # Define two polygons
    gt_0 = (gt.corners3d[0][0], gt.corners3d[0][2])
    gt_1 = (gt.corners3d[1][0], gt.corners3d[1][2])
    gt_2 = (gt.corners3d[2][0], gt.corners3d[2][2])
    gt_3 = (gt.corners3d[3][0], gt.corners3d[3][2])
    gt_poly = Polygon([gt_0, gt_1, gt_2, gt_3])
    
    pd_0 = (pd.corners3d[0][0], pd.corners3d[0][2])
    pd_1 = (pd.corners3d[1][0], pd.corners3d[1][2])
    pd_2 = (pd.corners3d[2][0], pd.corners3d[2][2])
    pd_3 = (pd.corners3d[3][0], pd.corners3d[3][2])
    pd_poly = Polygon([pd_0, pd_1, pd_2, pd_3])
    
    iou = gt_poly.intersection(pd_poly).area / gt_poly.union(pd_poly).area
    
    if iou == 0:
        return -1
    else:
        return iou


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
            dist_mat[i][j] = (1 - get_iou(gt_objs[i], dt_objs[j]))
            # dist_mat[i][j] = get_dist(gt_objs[i].loc, dt_objs[j].loc, 1)
            # dist_mat[i][j] = get_cost(gt_objs[i], dt_objs[j], 'iou')
    # REF: https://github.com/gatagat/lap/blob/master/lap/_lapjv.pyx
    # Hungarian assignment (x specifies the col_dt to which row_gt is assigned)
    cost, x, y = lapjv(dist_mat, extend_cost=True, cost_limit=1)
    
    # Match the pairs and write to cur_idx_matches list
    cur_idx_matches = [] # gt, dt
    for i in range(n):
        matched_dt_idx = x[i]
        if matched_dt_idx != -1:
            # # check for no iou opverlaps [1 1 1 1...]
            # if np.sum(dist_mat[i]) >= m:
            #     continue
            cur_idx_matches.append([gt_objs[i], dt_objs[matched_dt_idx]])
            kept_dt_ids.append(matched_dt_idx) #remove matched items from pd_list
    
    return cur_idx_matches, kept_dt_ids

def fix_obj(cur_idx_pairs, kept_dt_ids):
    mod_objs = {}
    for pid, pair in zip(kept_dt_ids, cur_idx_pairs):
        iou = get_iou(pair[0], pair[1])
        if iou == 0:
            continue
        if iou < 0.5:
            new_obj = copy.deepcopy(pair[1])
            new_obj.loc[0] = pair[0].loc[0] + (pair[1].loc[0] - pair[0].loc[0]) * 0.42
            # new_obj.loc[1] = pair[0].loc[1] + (pair[1].loc[0] - pair[0].loc[0]) * 0.1
            new_obj.loc[2] = pair[0].loc[2] + (pair[1].loc[2] - pair[0].loc[2]) * 0.17
            new_obj.src = new_obj.to_kitti_result()
            new_obj.corners3d = new_obj.generate_corners3d()
            new_obj.dis_to_cam = np.linalg.norm(new_obj.loc)
            
            new_iou = get_iou(pair[0], new_obj)
            if new_iou < iou and (iou-new_iou) > 0.01:
                print(f'old_iou:{iou}, new_iou:{new_iou}')
                print(pair[0].src)
                print(pair[1].src)
                print(new_obj.src)
            mod_objs.update({pid : new_obj})
        else:
            mod_objs.update({pid : copy.deepcopy(pair[1])})
    return mod_objs

### Write filtered txt results ###
def get_obj_str(obj):
    str_ = ' '.join(str(param) for param in obj)
    return str_    

def main():
    # Set up output directory
    output_path = os.path.join('/mnt/disk2/christine/shift_cruw/outputs', args.tags, datetime.datetime.now().strftime('%Y%m%d_%H%M'))
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
    for frame in tqdm(datum['gt'].keys()):
        # For each label find gt_dt pair list of (gt, dt)
        cur_idx_pairs, kept_dt_ids = get_gt_dt_pairs(datum, frame, cls='Car')
        ### FIX ATTEMPT ###
        mod_idx_pairs = fix_obj(cur_idx_pairs, kept_dt_ids)
        ###################
        # COUNT number of FP
        keep = []
        for i, obj in enumerate(datum['pd'][frame]):
            if obj.cls_type != 'Car':
                keep.append(obj.src)
            elif  i in mod_idx_pairs.keys():
                keep.append(mod_idx_pairs[i].src)
            else:
                total_removed += 1

        # write current idx to txt file
        with open(osp.join(output_path, frame), 'w') as f:
            for item in keep:
                f.write(item)
            f.close()
    
    print(f'Removed total: {total_removed}')
    

if __name__ == '__main__':
    main()
    


####
        # write current idx to txt file
        # with open(osp.join(output_path, frame), 'w') as f:
        #     for pair in cur_idx_pairs:
        #         gt_str = get_obj_str(pair[0])
        #         dt_str = get_obj_str(pair[1])
        #         f.write(f'{gt_str}%{dt_str}\n')
        #     f.close()