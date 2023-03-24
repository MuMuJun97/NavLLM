import json
import jsonlines
from transformers import AutoTokenizer, PretrainedConfig
import random
import os
import numpy as np
from tqdm import tqdm

def load_nav_graphs(connectivity_dir,anno_data,soon_data):
    ''' Load connectivity graph for each scan
    https://github.com/peteanderson80/Matterport3DSimulator/tree/master/connectivity
    Connectivity graphs indicating the navigable paths between viewpoints in each scan.
    每个json文件都包含一个标注数组，scan中的每个viewpoint都有annotation.所有annotations共享相同的基础结构
    { 注: 单位都是米
      "image_id": str,  matterport skybox prefix 文件名前缀
      "pose": [float x 16], 4x4矩阵. row major order.
        transforms matterport skyboxes to global coordinates (z-up).
        Pose matrices are based on the assumption that the camera is facing skybox image 3.
      "included": boolean, viewpoint是否包含在Simulator中,一些重叠的viewpoints将被排除在外.
      "visible": [boolean x num_viewpoints], 表明其他viewpoints能从此viewpoint看到.
      "unobstructed": [boolean x num_viewpoints], 指示到其他viewpoints的转换,这些viewpoints被认为是可以导航到的.
      "height": float, 地平面floor上方viewpoint的估计高度, Simulator不需要
    }  scans.txt contains a list of all the scan ids in the dataset.
    '''
    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3] - pose2['pose'][3]) ** 2 \
                + (pose1['pose'][7] - pose2['pose'][7]) ** 2 \
                + (pose1['pose'][11] - pose2['pose'][11]) ** 2) ** 0.5

    scans = [x.strip() for x in open(os.path.join(connectivity_dir, 'scans.txt')).readlines()]
    scan_node_data = dict()
    for scan in scans:
        scan_node_data[scan] = dict()
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            data = json.load(f)
            for i, item in enumerate(data):
                if item['included']:
                    if scan_node_data[scan].get(item['image_id'],None) is None:
                        scan_node_data[scan][item['image_id']] = dict()
                    scan_node_data[scan][item['image_id']]['position'] = np.array([item['pose'][3],item['pose'][7], item['pose'][11]])
                    scan_node_data[scan][item['image_id']]['height'] = item['height']
    return scan_node_data



if __name__ == '__main__':
    choice = 0
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if choice == 0:
        # train
        split_idx = 0  # train
        anno_file = "datasets/SOON/annotations/pretrain_obj/train_enc.jsonl"
        save_file = "datasets/SOON/annotations/pretrain_obj/train_with_qa.jsonl"
        soon_data_file = "datasets/SOON/annotations/pretrain_obj/train_pseudo_objid.jsonl"

    if choice == 2:
        # val_unseen_instrs
        split_idx = 2  # train
        anno_file = "datasets/SOON/annotations/pretrain_obj/val_unseen_instrs_enc.jsonl"
        save_file = "datasets/SOON/annotations/pretrain_obj/val_unseen_instrs_with_qa.jsonl"
        soon_data_file = "datasets/SOON/annotations/pretrain_obj/val_unseen_instrs_pseudo_objid.jsonl"

    if choice == 3:
        # val_unseen_house_enc
        split_idx = 3  # train
        anno_file = "datasets/SOON/annotations/pretrain_obj/val_unseen_house_enc.jsonl"
        save_file = "datasets/SOON/annotations/pretrain_obj/val_unseen_house_with_qa.jsonl"
        soon_data_file = "datasets/SOON/annotations/pretrain_obj/val_unseen_house_pseudo_objid.jsonl"

    connectivity_dir = "datasets/R2R/connectivity"

    anno_data = []
    with jsonlines.open(anno_file, 'r') as f:
        for item in f:
            anno_data.append(item)

    soon_data = []
    with jsonlines.open(soon_data_file, 'r') as f:
        for item in f:
            soon_data.append(item)

    scan_node_data = load_nav_graphs(connectivity_dir,anno_data,soon_data)

    pbar = tqdm(range(len(soon_data)), desc="generate QAs for dataset: ")

    str_max_len = 0
    count = -1
    new_data = []

    for idx in pbar:
        item = soon_data[idx]
        pos_vps = set()
        new_boxs = []
        for box in item['bboxes']:
            # remove bboxes without pseudo_label
            if box.get('pseudo_label', None) is not None:
                new_boxs.append(box)
                pos_vps.add(box['image_id'])
        item['bboxes'] = new_boxs

        for path_k, path in enumerate(item['path']):
            box_k = 0
            count += 1
            for instr_j, instr in enumerate(item['instructions']):
                new_item = {}
                new_item['instr_id'] = '{}-{}_{}'.format(split_idx, count, instr_j)
                cur_box_idx = box_k % len(item['bboxes'])
                cur_box = item['bboxes'][cur_box_idx]
                new_item['scan'] = cur_box['scan']
                new_item['path'] = path
                new_item['pos_vps'] = list(pos_vps)
                new_item['obj_heading'] = cur_box['target']['center']['heading']
                new_item['obj_elevation'] = cur_box['target']['center']['elevation']
                new_item['obj_pseudo_label'] = cur_box['pseudo_label']
                new_item['obj_name'] = cur_box['obj_name']
                new_item['instr_encoding'] = tokenizer.encode(instr[-2])
                new_item['instr'] = instr  # List[]
                new_item['label_id'] = idx  # soon data idx

                new_data.append(new_item)
                box_k += 1

                # add instr to nodes:
                for end_node in new_item['pos_vps']:
                    if scan_node_data[new_item['scan']][end_node].get('desc',None) is None:
                        scan_node_data[new_item['scan']][end_node]['desc'] = dict()
                    scan_node_data[new_item['scan']][end_node]['desc'][instr[-1]] = instr

                for path_node in path:
                    scan_node_data[new_item['scan']][path_node]['path'] = 1


    sum_p = 0
    sum_n = 0
    rooms = []
    for k,v in scan_node_data.items():
        curnode_txts = 0
        curnode_valid = 0
        for kk,vv in scan_node_data[k].items():
            if vv.get('desc',None) is not None:
                curnode_txts += 1
                for pre, text in vv['desc'].items():
                    rooms.append(text[2])
            if vv.get('path',None) is not None:
                curnode_valid += 1
        if curnode_txts > 0:
            p = curnode_txts/curnode_valid
            print(k, ": {:.2f} %".format(p*100))
            sum_p += p
            sum_n += 1
    print("occupancy: {:.2f} %".format(sum_p/sum_n*100))
    print(len(rooms))

    import pickle
    with open('llm/cfg/scan_node_instr.pkl','wb') as f:
        pickle.dump(scan_node_data,f)

