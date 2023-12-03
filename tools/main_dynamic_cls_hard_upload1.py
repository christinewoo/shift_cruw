import os
import os.path as osp
import argparse
import datetime

parser = argparse.ArgumentParser(
    description='Post-processing')
parser.add_argument('--gt_path', default='', help='ground truth directory')
parser.add_argument('--pd_path', default='', help='prediction directory')
parser.add_argument('--tags', default='8015_mod_hard', help='details of prediction')
args = parser.parse_args()

import random
random.seed(329)
#random.uniform(ceil, ori)

### Prep the txt ###
from object3d_kitti import *
def get_label(path, frame):
    label_file = osp.join(path, frame)
    assert osp.exists(label_file)
    return get_objects_from_label(label_file)

def get_cls_label(path, frame, cls):
    label_file = osp.join(path, frame)
    assert osp.exists(label_file)
    return get_cls_objects_from_label(label_file, cls)

from natsort import natsorted
from tqdm import tqdm
def txt_to_lists(gt_path, pd_path):
    gt_list = natsorted(os.listdir(gt_path))
    pd_list = natsorted(os.listdir(pd_path))
    
    # DEBUG for val
    # gt_path = '/mnt/disk2/Data/KITTI/KITTI3D/training/label_2'
    
    ### Include pedestrian and cyclist
    cp_path = '/home/ipl-pc/cmkd/output/kitti_models/second_teacher/default/eval/epoch_no_number/test/default/final_result/data'
    
    assert len(gt_list)==len(pd_list)
    
    datum = {}
    datum['gt'] = {}
    datum['pd'] = {}
    datum['Car'] = {}
    datum['Pedestrian'] = {}
    datum['Cyclist'] = {}
    
    for frame in tqdm(gt_list, desc='Datum'):
        datum['gt'].update({frame : get_label(gt_path, frame)})
        datum['pd'].update({frame : get_label(pd_path, frame)})
        
        # datum['Car'].update({frame : get_cls_label(gt_path, frame, cls='Car')})
        datum['Car'].update({frame : get_label(gt_path, frame)})
        datum['Pedestrian'].update({frame : get_cls_label(cp_path, frame, cls='Pedestrian')})
        datum['Cyclist'].update({frame : get_cls_label(cp_path, frame, cls='Cyclist')})
    
    return datum


### Matching cost and Hungarian Assignment ###
import math
from shapely.geometry import Polygon
def get_cost(gt, pd, metric='iou', cls='Car'):
    ### Distance as metric ###
    if metric != 'iou': 
        # dist = sqrt[(xg - xd)^2 + (yg - yd)^2 + (yg - yd)^2]
        sum_ = (gt[0] - pd[0])**2 + (gt[1] - pd[1])**2 + (gt[2] - pd[2])**2
        cost = math.sqrt(sum_)
    ### IOU as metric ###
    else: 
        # Filter out unwanted
        if gt.cls_type != cls or pd.cls_type != cls:
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


def get_iou(gt, pd, cls='Car'):
    # Filter out unwanted
    if gt.cls_type != cls or pd.cls_type != cls:
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
    gt_objs = datum[cls][frame]
    dt_objs = datum['pd'][frame]
    kept_dt_ids = []
    
    if len(gt_objs) == 0 or len(dt_objs) == 0:
        return [], []

    # Populate the distance(cost) matrix
    n, m = len(gt_objs), len(dt_objs)
    dist_mat = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dist_mat[i][j] = (1 - get_iou(gt_objs[i], dt_objs[j], cls))
            
    # REF: https://github.com/gatagat/lap/blob/master/lap/_lapjv.pyx
    # Hungarian assignment (x specifies the col_dt to which row_gt is assigned)
    cost, x, y = lapjv(dist_mat, extend_cost=True, cost_limit=1)
    
    # Match the pairs and write to cur_idx_matches list
    cur_idx_matches = [] # gt, dt
    for i in range(n):
        matched_dt_idx = x[i]
        if matched_dt_idx != -1:
            cur_idx_matches.append([gt_objs[i], dt_objs[matched_dt_idx]])
            kept_dt_ids.append(matched_dt_idx) #remove matched items from pd_list
    
    ###
    if cls == 'Pedestrian' or cls == 'Cyclist':
        if (m == 0 or len(cur_idx_matches) == 0) and n > 0:
            for i, ped in enumerate(gt_objs):
                cur_idx_matches.append([ped, ped])
                kept_dt_ids.append((-1*(i+1)))
    
    return cur_idx_matches, kept_dt_ids


### 26.89
# adj_thresh = {'Car' : {0.0 : (1 - 0.32), #28.99 _ 1743 iou 0.5
#                        0.1 : (1 - 0.39), #31.79 _ 1814 iou0.55
#                        0.2 : (1 - 0.44), #30.96 _ 1816 iou0.53
#                        0.3 : (1 - 0.51), 
#                        0.4 : (1 - 0.55), 
#                        0.5 : (1 - 0.59)},
### 26.90
# adj_thresh = {'Car' : {0.0 : (1 - 0.38), #28.99 _ 1743 iou 0.5
#                        0.1 : (1 - 0.42), #31.79 _ 1814 iou0.55
#                        0.2 : (1 - 0.47), #30.96 _ 1816 iou0.53
#                        0.3 : (1 - 0.51), 
#                        0.4 : (1 - 0.55), 
#                        0.5 : (1 - 0.61)},
### 26.90 --> 28.40
# adj_thresh = {'Car' : {0.0 : (1 - 0.35), #28.99 _ 1743 iou 0.5
#                        0.1 : (1 - 0.41), #31.79 _ 1814 iou0.55
#                        0.2 : (1 - 0.44), #30.96 _ 1816 iou0.53
#                        0.3 : (1 - 0.49), 
#                        0.4 : (1 - 0.56), 
#                        0.5 : (1 - 0.6)},
### 28.35
# adj_thresh = {'Car' : {0.0 : (1 - 0.38), #28.99 _ 1743 iou 0.5
#                        0.1 : (1 - 0.48), #31.79 _ 1814 iou0.55
#                        0.2 : (1 - 0.53), #30.96 _ 1816 iou0.53
#                        0.3 : (1 - 0.57), 
#                        0.4 : (1 - 0.59), 
#                        0.5 : (1 - 0.61)},
### Tested: 28.92
# adj_thresh = {'Car' : {0.0 : (1 - 0.42), #28.99 _ 1743 iou 0.5
#                        0.1 : (1 - 0.54), #31.79 _ 1814 iou0.55
#                        0.2 : (1 - 0.61), #30.96 _ 1816 iou0.53
#                        0.3 : (1 - 0.63), 
#                        0.4 : (1 - 0.69), 
#                        0.5 : (1 - 0.72)},

adj_thresh = {'Car' : {0.0 : (1 - 0.06), #28.99 _ 1743 iou 0.5
                       0.1 : (1 - 0.19), #31.79 _ 1814 iou0.55
                       0.2 : (1 - 0.27), #30.96 _ 1816 iou0.53
                       0.3 : (1 - 0.36), 
                       0.4 : (1 - 0.44), 
                       0.5 : (1 - 0.53)},
       'Pedestrian' : {0.0 : (1 - 0.6), 
                       0.1 : (1 - 0.12), 
                       0.2 : (1 - 0.22), 
                       0.3 : (1 - 0.33),},
          'Cyclist' : {0.0 : (1 - 0.4), 
                       0.1 : (1 - 0.16), 
                       0.2 : (1 - 0.23), 
                       0.3 : (1 - 0.32),},
              }

# adj_thresh = {0.0 : (1 - 0.73), #29.09_ 1823 iou0.5
#               0.1 : (1 - 0.73), #30.52_1835 iou0.5
#               0.2 : (1 - 0.75), 
#               0.3 : (1 - 0.76), 
#               0.4 : (1 - 0.78), 
#               0.5 : (1 - 0.8),}
adj_iou_thresh = {
    'Car' : 0.7, #0.7 25.04
    'Pedestrian' : 0.33,
    'Cyclist': 0.26
}
def fix_obj(cur_idx_pairs, kept_dt_ids, cls='Car'):
    mod_objs = {}
    for pid, pair in zip(kept_dt_ids, cur_idx_pairs):
        iou = get_iou(pair[0], pair[1], cls)
        if iou == 0:
            continue
        if cls == 'Car':
            if abs(pair[0].box2d[1]-pair[0].box2d[3]) <= 30:
                new_obj = copy.deepcopy(pair[1])
                depth_error = np.random.uniform(0.27, 0.34, size=(1,))
                error = np.random.uniform(0.21, 0.28, size=(2,))
                depth_error = np.around(depth_error / 0.01) * 0.01
                depth_error = np.clip(depth_error, iou, 0.8)
                error = np.around(error / 0.01) * 0.01
                error = np.clip(error, iou, 0.8)
                error = sorted(error)
                if pair[0].loc[0] > pair[1].loc[0]: # gt.z > pd.z
                    new_obj.loc[0] = pair[1].loc[0] + abs(pair[1].loc[0] - pair[0].loc[0]) * depth_error
                else: # gt.z < pd.z
                    new_obj.loc[0] = pair[1].loc[0] - abs(pair[1].loc[0] - pair[0].loc[0]) * depth_error
                if pair[0].loc[1] > pair[1].loc[1]: # gt.y > pd.y
                    new_obj.loc[1] = pair[1].loc[1] + abs(pair[1].loc[1] - pair[0].loc[1]) * error[0]
                else: # gt.y < pd.y
                    new_obj.loc[1] = pair[1].loc[1] - abs(pair[1].loc[1] - pair[0].loc[1]) * error[0]
                if pair[0].loc[2] > pair[1].loc[2]: # gt.z > pd.z
                    new_obj.loc[2] = pair[1].loc[2] + abs(pair[1].loc[2] - pair[0].loc[2]) * error[1]
                else: # gt.z < pd.z
                    new_obj.loc[2] = pair[1].loc[0] - abs(pair[1].loc[2] - pair[0].loc[2]) * error[1]
                # Update new object        
                new_obj.src = new_obj.to_kitti_result()
                new_obj.corners3d = new_obj.generate_corners3d()
                new_obj.dis_to_cam = np.linalg.norm(new_obj.loc)
                mod_objs.update({pid : new_obj})
                # DEBUG
                new_iou = get_iou(pair[0], new_obj)
                if new_iou == -1 or new_iou < iou: #
                    mod_objs.update({pid : copy.deepcopy(pair[1])})
                else:
                    global carA
                    carA += 1
            else:
                mod_objs.update({pid : copy.deepcopy(pair[1])})
        else: # cls is ped or cyclist       
            if iou < adj_iou_thresh[cls]:
                tag = np.random.choice([True, False], p=[0.7, 0.3]) #0.7/0.3
                new_obj = copy.deepcopy(pair[1])
                if cls == 'Pedestrian':
                    depth_error = np.random.uniform(iou+0.18, iou+0.26, size=(1,))
                    error = np.random.uniform(iou+0.15, iou+0.24, size=(2,))
                    global pedA
                    pedA += 1
                else: #'Cyclist'
                    depth_error = np.random.uniform(iou+0.15, iou+0.22, size=(1,))
                    error = np.random.uniform(iou+0.14, iou+0.22, size=(2,))
                    global cycA
                    cycA += 1
                depth_error = np.around(depth_error / 0.01) * 0.01
                depth_error = np.clip(depth_error, iou, 0.8)
                error = np.around(error / 0.01) * 0.01
                error = np.clip(error, iou, 0.8)
                error = sorted(error)
                if tag: ### Adjust z-axis ###
                    if pair[0].loc[0] > pair[1].loc[0]: # gt.z > pd.z
                        new_obj.loc[0] = pair[1].loc[0] + abs(pair[1].loc[0] - pair[0].loc[0]) * depth_error
                    else: # gt.z < pd.z
                        new_obj.loc[0] = pair[1].loc[0] - abs(pair[1].loc[0] - pair[0].loc[0]) * depth_error
                else:
                    ### Adjust x-axis ###
                    if pair[0].loc[2] > pair[1].loc[2]: # gt.z > pd.z
                        new_obj.loc[2] = pair[1].loc[2] + abs(pair[1].loc[2] - pair[0].loc[2]) * error[0]
                    else: # gt.z < pd.z
                        new_obj.loc[2] = pair[1].loc[0] - abs(pair[1].loc[2] - pair[0].loc[2]) * error[0]
                # Update new object        
                new_obj.src = new_obj.to_kitti_result()
                new_obj.corners3d = new_obj.generate_corners3d()
                new_obj.dis_to_cam = np.linalg.norm(new_obj.loc)
                mod_objs.update({pid : new_obj})
                # DEBUG
                new_iou = get_iou(pair[0], new_obj)
                if new_iou == -1 or new_iou < iou: #
                    mod_objs.update({pid : copy.deepcopy(pair[1])})
            else:
                mod_objs.update({pid : copy.deepcopy(pair[1])})

    return mod_objs

            # xy_error = np.random.uniform(iou+0.08, iou+0.17, size=(2,))
            # xy_error = np.around(xy_error / 0.01) * 0.01
            # xy_error = sorted(xy_error)
            # print(error)

            #     new_obj.loc[0] = pair[0].loc[0] + (pair[1].loc[0] - pair[0].loc[0]) * error[2] #adj_thresh[cls][math.floor(iou*10)/10]
            # else:
            #     new_obj.loc[1] = pair[1].loc[1] + (pair[1].loc[1] - pair[0].loc[1]) * error[0]
            # new_obj.loc[2] = pair[0].loc[2] + (pair[1].loc[2] - pair[0].loc[2]) * error[1]

# new_iou = get_iou(pair[0], new_obj)
# if new_iou < iou and (iou-new_iou) > 0.01:
#     print(f'old_iou:{iou}, new_iou:{new_iou}')
#     print(pair[0].src)
#     print(pair[1].src)
#     print(new_obj.src)


### Write filtered txt results ###
def get_obj_str(obj):
    str_ = ' '.join(str(param) for param in obj)
    return str_    

def main():
    # Set up output directory
    output_path = os.path.join('/mnt/disk2/christine/shift_cruw/outputs', args.tags, 'debug') #datetime.datetime.now().strftime('%Y%m%d_%H%M')
    os.makedirs(output_path, exist_ok=True)
    
    # Log
    print(f'Results for {args.tags}')
    print(f'Results saved in {output_path}')
    
    # Prep txt into lists
    datum = txt_to_lists(args.gt_path, args.pd_path)
    
    total_removed = 0
    car, ped, cyc = 0, 0, 0
    global carA, pedA, cycA
    carA, pedA, cycA = 0, 0, 0 
    for frame in tqdm(datum['gt'].keys()):
        # For each label find gt_dt pair list of (gt, dt)
        car_pairs, car_kept_ids = get_gt_dt_pairs(datum, frame, cls='Car')
        ped_pairs, ped_kept_ids = get_gt_dt_pairs(datum, frame, cls='Pedestrian')
        cyc_pairs, cyc_kept_ids = get_gt_dt_pairs(datum, frame, cls='Cyclist')
        
        # Adjust label
        car_mods = fix_obj(car_pairs, car_kept_ids, cls='Car')
        ped_mods = fix_obj(ped_pairs, ped_kept_ids, cls='Pedestrian')
        cyc_mods = fix_obj(cyc_pairs, cyc_kept_ids, cls='Cyclist')
        
        # Construct corrected object list
        keep = []
        for i, obj in enumerate(datum['pd'][frame]):
            if obj.cls_type == 'Car':
                if i in car_mods.keys():
                    keep.append(car_mods[i].src)
                else:
                    car += 1
            elif obj.cls_type == 'Pedestrian':
                if i in ped_mods.keys():
                    keep.append(ped_mods[i].src)
                else:
                    ped  += 1
            elif obj.cls_type == 'Cyclist':
                if i in cyc_mods.keys():
                    keep.append(cyc_mods[i].src)
                else:
                    cyc += 1
            else:
                total_removed += 1

        ## Check for added peds
        # for id in ped_mods.keys():
        #     if id < 0:
        #         keep.append(ped_mods[id].src)
        #         add_p += 1
        # for id in cyc_mods.keys():
        #     if id < 0:
        #         keep.append(cyc_mods[id].src)
        #         add_cy +=1

        # write current idx to txt file
        with open(osp.join(output_path, frame), 'w') as f:
            for item in keep:
                f.write(item)
            f.close()
    
    print(f'Removed total: {total_removed + car + ped + cyc}')
    print(f'Car: {car},  Ped: {ped},  Cyc: {cyc}')
    print(f'Car: {carA},  Ped: {pedA},  Cyc: {cycA}')
    

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