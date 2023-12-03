import os
import os.path as osp
import argparse

parser = argparse.ArgumentParser(
    description='Post-processing')
parser.add_argument('--gt_path', default='', help='ground truth directory')
parser.add_argument('--pd_path', default='', help='prediction directory')
parser.add_argument('--tags', default='filled', help='details of prediction')
args = parser.parse_args()

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
    
    # Include pedestrian and cyclist
    cp_path = '/home/ipl-pc/cmkd/output/kitti_models/second_teacher/default/eval/epoch_no_number/test/default/final_result/data'
    
    assert len(gt_list)==len(pd_list)
    
    datum = {}
    datum['pd'] = {}
    datum['Car'] = {}
    datum['Pedestrian'] = {}
    datum['Cyclist'] = {}
    
    for frame in tqdm(gt_list, desc='Datum'):
        datum['pd'].update({frame : get_label(pd_path, frame)})
        datum['Car'].update({frame : get_cls_label(gt_path, frame, cls='Car')})
        datum['Pedestrian'].update({frame : get_cls_label(cp_path, frame, cls='Pedestrian')})
        datum['Cyclist'].update({frame : get_cls_label(cp_path, frame, cls='Cyclist')})
    
    return datum

from shapely.geometry import Polygon
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

def main():
    # Set up output directory
    output_path = os.path.join('/mnt/disk2/christine/shift_cruw/outputs/filled/debug') #datetime.datetime.now().strftime('%Y%m%d_%H%M')
    os.makedirs(output_path, exist_ok=True)
    print(f'Results saved in {output_path}')
    
    # Prep txt into lists
    datum = txt_to_lists(args.gt_path, args.pd_path)
    
    car, ped, cyc = 0, 0, 0
    car_ceil = int(datum['Car'].keys() * 0.1)
    ped_ceil = int(datum['Pedestrian'].keys() * 0.1)
    cyc_ceil = int(datum['Cyclist'].keys() * 0.1)
    for frame in tqdm(datum['pd'].keys()):
        # For each label find gt_dt pair list of (gt, dt)
        car_pairs, car_kept_ids = get_gt_dt_pairs(datum, frame, cls='Car')
        ped_pairs, ped_kept_ids = get_gt_dt_pairs(datum, frame, cls='Pedestrian')
        cyc_pairs, cyc_kept_ids = get_gt_dt_pairs(datum, frame, cls='Cyclist')


if __name__ == '__main__':
    main()