'''
Instruction and trajectory (view and object features) dataset
'''
import os
import json
import jsonlines
import numpy as np
import h5py
import math
from .common import get_angle_fts, get_view_rel_angles
from .common import calculate_vp_rel_pos_fts
from .common import softmax
from .common import load_nav_graphs
MAX_DIST = 30   # normalize
MAX_STEP = 10   # normalize
TRAIN_MAX_STEP = 20

class SoonTextPathData(object):
    def __init__(
            self, anno_files, img_ft_file, obj_ft_file, scanvp_cands_file, connectivity_dir,
            image_feat_size=768, image_prob_size=1000, angle_feat_size=4,
            obj_feat_size=2048, obj_prob_size=1601, max_objects=20,
            max_txt_len=100, in_memory=True, act_visited_node=False
    ):
        self.img_ft_file = img_ft_file  # '../datasets/R2R/features/pth_vit_base_patch16_224_imagenet.hdf5'
        self.obj_ft_file = obj_ft_file  # '../datasets/SOON/features/filtered_butd_bboxes.hdf5'

        self.image_feat_size = image_feat_size  # 768
        self.image_prob_size = image_prob_size  # 1000
        self.angle_feat_size = angle_feat_size  # 4
        self.obj_feat_size = obj_feat_size  # 2048
        self.obj_prob_size = obj_prob_size  # 1601

        self.obj_image_h = 480
        self.obj_image_w = 640
        self.obj_image_size = 480 * 640

        self.max_txt_len = max_txt_len  # 200
        self.max_objects = max_objects  # 100
        self.act_visited_node = act_visited_node  # False

        self.in_memory = in_memory  # True
        if self.in_memory:
            self._feature_store = {}
        # scanvp_cands_file: '../datasets/R2R/annotations/scanvp_candview_relangles.json'
        # {scan_vp: {vp: [viewidx, rel_angle_dist, rel_heading, rel_elevation]}}
        # view idx: 0-36
        # relative angle dist;
        # relative heading
        # relative elevation
        self.scanvp_cands = json.load(open(scanvp_cands_file))

        import pickle
        with open('../../datasets/SOON/graphs_v1.pkl', 'rb') as f:
            self.graphs = pickle.load(f)
        with open('../../datasets/SOON/shortest_distances_v1.pkl', 'rb') as f:
            self.shortest_distances = pickle.load(f)
        with open('../../datasets/SOON/shortest_paths_v1.pkl', 'rb') as f:
            self.shortest_paths = pickle.load(f)
        # self.graphs, self.shortest_distances, self.shortest_paths = load_nav_graphs(connectivity_dir)

        # import pickle
        # ../app/graphs_v1.pkl
        # ../app/shortest_distances_v1.pkl
        # with open('../app/shortest_paths_v1.pkl','wb') as f:
        #     pickle.dump(self.shortest_paths,f)
        # all_point_rel_angles 36张图像的朝向角度.
        self.all_point_rel_angles = [get_view_rel_angles(baseViewId=i) for i in range(36)]
        self.all_point_angle_fts = [get_angle_fts(x[:, 0], x[:, 1], self.angle_feat_size) for x in
                                    self.all_point_rel_angles]

        self.data = []
        # anno_files: SOON/annotations/pretrain_obj/train_enc.jsonl
        # 相同instruction,不同path的预训练注释文件. 提供了instruction的BERT token_ids(instr_encoding),
        # path的viewpoint str, 还提供了object的heading与orientation，以及pseudo label
        for anno_file in anno_files:
            with jsonlines.open(anno_file, 'r') as f:
                for item in f:
                    self.data.append(item)

        """
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        tokenizer.decode(item['instr_encoding'])
        # 这里的train标注同 SOON/train.json bert模型的tokens_id
        '[CLS] this is a brand new white, rectangular wooden table, which is above a few chairs, under a pot of flowers. it is in a very neat study with many books. [SEP]'
        """

        self.obj_image_h = self.obj_image_w = 600
        self.obj_image_size = 600 * 600
        self.split = anno_files[0].split('/')[-1].split('.')[0]

    def __len__(self):
        return len(self.data)

    # 注释原ReverieTextPathData的类方法
    # def get_scanvp_feature(self, scan, viewpoint):
    #     key = '%s_%s' % (scan, viewpoint) # key是scan+viewpoint,
    #     if self.in_memory and key in self._feature_store:
    #         view_fts, obj_fts, obj_attrs = self._feature_store[key]
    #     else:
    #         # 1. viewpoint的图像特征  [36,1768] [36张图像, 768图像特征+1000 head features]
    #         with h5py.File(self.img_ft_file, 'r') as f:
    #             view_fts = f[key][...].astype(np.float32)
    #
    #         obj_attrs = {}
    #         obj_fts = np.zeros((0, self.obj_feat_size + self.obj_prob_size), dtype=np.float32)
    #
    #         # 2. obj_fts是BUTD检测器检测到的物体特征 [n_objs,3649]
    #         if self.obj_ft_file is not None:
    #             """
    #             论文使用BUTD检测器,在VisualGenome上预训练过,可以检测整个全景图像的objects,覆盖1600多种object和scene classes
    #             过滤了不重要的一些classes,比如background,floor之类的.
    #             然后,根据object classes的semantic similarity语义相似性以及物体中心点与标注目标的欧式距离,
    #             选择一个检测到的物体作为Pseudo target.
    #             以这种方式, 论文将SOON数据集的object grounding setting转换为了与REVERIE数据集相似的setting,其目标是从
    #             所有candidate objects中选择一个物体.
    #             """
    #             with h5py.File(self.obj_ft_file, 'r') as f:
    #                 if key in f:
    #                     obj_fts = f[key][...].astype(np.float32)
    #                     obj_fts = obj_fts[:self.max_objects]
    #                     for attr_key, attr_value in f[key].attrs.items():
    #                         if attr_key in ['directions', 'sizes', 'bboxes', 'obj_ids']:
    #                             obj_attrs[attr_key] = attr_value[:self.max_objects]
    #         if self.in_memory:
    #             # viewpoint_fts是ViT的图像特征
    #             # obj_fts是BUTD的object detector特征
    #             # obj_attrs是object attributes.
    #             # memory 存储: 图像特征,检测到的目标特征,目标属性, 避免重复read files
    #             self._feature_store[key] = (view_fts, obj_fts, obj_attrs)
    #
    #     return view_fts, obj_fts, obj_attrs
    #
    # def get_obj_label(self, item, last_vp_objids):
    #     gt_obj_id = item['instr_id'].split('_')[1]
    #     for k, obj_id in enumerate(last_vp_objids):
    #         if obj_id == gt_obj_id:
    #             obj_label = k
    #             break
    #     else:
    #         # it occurs when the gt_objid is not in max_objects
    #         obj_label = -100  # ignore
    #         # print('No groundtruth obj_id', item['instr_id'], len(obj_ids))
    #     return obj_label


    def get_scanvp_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint) # key是scan+viewpoint,
        if self.in_memory and key in self._feature_store:
            view_fts, obj_fts, obj_attrs = self._feature_store[key]
        else:
            # 1. viewpoint的图像特征  [36,1768] [36张图像, 768图像特征+1000 head features]
            with h5py.File(self.img_ft_file, 'r') as f:
                view_fts = f[key][...].astype(np.float32)
                # scenes = dict()
                # for k in f.keys():
                #     name = k.split('_')[0]
                #     if scenes.get(name,None) is not None:
                #         scenes[name].add(k.split('_')[1])
                #     else:
                #         scenes[name] = set()
                #         scenes[name].add(k.split('_')[1])
                # print(scenes)
            obj_attrs = {}
            obj_fts = np.zeros((0, self.obj_feat_size+self.obj_prob_size), dtype=np.float32)

            # 2. obj_fts是BUTD检测器检测到的物体特征 [n_objs,3649]
            if self.obj_ft_file is not None:
                with h5py.File(self.obj_ft_file, 'r') as f:
                    if key in f:
                        obj_fts = f[key][...].astype(np.float32)
                        obj_fts = obj_fts[:self.max_objects]
                        # 检测到的物体Bounding boxes,朝向,class id.
                        for attr_key, attr_value in f[key].attrs.items():
                            if attr_key in ['directions', 'bboxes', 'obj_ids']:
                                obj_attrs[attr_key] = attr_value[:self.max_objects]
                        obj_attrs['bboxes'] = np.array(obj_attrs['bboxes']).astype(np.float32)
                        obj_attrs['sizes'] = np.zeros((len(obj_attrs['bboxes']), 2), dtype=np.float32)
                        obj_attrs['sizes'][:, 0] = obj_attrs['bboxes'][:, 2] - obj_attrs['bboxes'][:, 0]
                        obj_attrs['sizes'][:, 1] = obj_attrs['bboxes'][:, 3] - obj_attrs['bboxes'][:, 1]
            if self.in_memory:
                # memory 存储: 图像特征,检测到的目标特征,目标属性, 避免重复read files
                self._feature_store[key] = (view_fts, obj_fts, obj_attrs)

        return view_fts, obj_fts, obj_attrs

    def get_obj_label(self, item, last_vp_objids):
        obj_label = item['obj_pseudo_label']['idx']
        if obj_label >= self.max_objects:
            obj_label = -100
        return obj_label

    def get_act_labels(self, end_vp, item, gmap_vpids, gmap_visited_masks, traj_cand_vpids):
        scan = item['scan']
        pos_vps = item['pos_vps']
        if end_vp in pos_vps:
            global_act_label = local_act_label = 0
        else:
            global_act_label = local_act_label = -100
            # global: unvisited vp
            cand_min_dist = float('inf')
            for k, cand_vp in enumerate(gmap_vpids):
                if (k > 0) and (not gmap_visited_masks[k]):
                    min_dist = min([self.shortest_distances[scan][end_vp][cand_vp] \
                                    + self.shortest_distances[scan][cand_vp][pos_vp] for pos_vp in pos_vps])
                    if min_dist < cand_min_dist:
                        cand_min_dist = min_dist
                        global_act_label = k  # [stop] is 0
            # local:
            cand_min_dist = float('inf')
            for k, cand_vp in enumerate(traj_cand_vpids[-1]):
                min_dist = min([self.shortest_distances[scan][end_vp][cand_vp] \
                                + self.shortest_distances[scan][cand_vp][pos_vp] for pos_vp in pos_vps])
                if min_dist < cand_min_dist:
                    cand_min_dist = min_dist
                    local_act_label = k + 1  # [stop] is 0
        return global_act_label, local_act_label

    def get_input(
            self, idx, end_vp_type, return_img_probs=False, return_act_label=False,
            return_obj_label=False, end_vp=None
    ):
        if end_vp_type == 'pos':
            # self.data[0] 存储datasets/SOON/annotations/pretrain_obj/train_enc.jsonl的标注数据
            # ['path'] 是List['str'],列表里存储viewpoint字符串
            end_vp = self.data[idx]['path'][-1] # end viewpoint

        ####################################################################################
        """
        Args:
            idx:
            end_vp_type: 为'pos'时,返回viewpoint字符串的位置.
            return_img_probs:
            return_act_label:
            return_obj_label:
            end_vp:

        Returns:

        """
        # item为标注好的数据.
        item = self.data[idx]
        scan = item['scan']
        # vp表示viewpoint,即为Matterport3DSimulator中的一个视点,
        # item['path']存储了路径，也就是List['viewpoint']
        # start_vp: start viewpoint str
        start_vp = item['path'][0]
        # 起点的航向角度 0
        start_heading = item.get('heading', 0)

        # 终点的viewpoint str
        pos_vps = item['pos_vps']

        # 起点到终点的位置
        gt_path = item['path']

        # end_vp: 路径的最后一个viewpoint
        if end_vp is None:
            if end_vp_type == 'pos':
                # 当位置viewpoint只有一个时, len(pos_vps)=1, np.random.randint(len(pos_vps))=0
                # end_vp = pos_vps[0]
                end_vp = pos_vps[np.random.randint(len(pos_vps))]
            elif end_vp_type == 'neg_in_gt_path':
                end_vps = [vp for vp in gt_path if vp not in pos_vps]
                if len(end_vps) == 0:
                    end_vps = gt_path
                end_vp = end_vps[np.random.randint(len(end_vps))]
            elif end_vp_type == 'neg_others':
                noneg_vp_set = set(pos_vps + gt_path)
                end_vps = [vp for vp in self.graphs[scan].nodes.keys() if vp not in noneg_vp_set]
                end_vp = end_vps[np.random.randint(len(end_vps))]

        # Graph出来的最短路径
        gt_path = self.shortest_paths[scan][start_vp][end_vp]

        # 返回朝向与高度, 倒数第2个node -> 倒数第1个node(即终点)
        cur_heading, cur_elevation = self.get_cur_angle(scan, gt_path, start_heading)

        if len(gt_path) > TRAIN_MAX_STEP:  # > 20
            # truncate trajectory 对轨迹长度进行截断,限制20个viewpoint
            gt_path = gt_path[:TRAIN_MAX_STEP] + [end_vp]

        traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
            last_vp_angles, last_vp_objids = self.get_traj_pano_fts(scan, gt_path)

        # global: the first token is [stop]
        gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists = \
            self.get_gmap_inputs(scan, gt_path, cur_heading, cur_elevation)

        # local: the first token is [stop]; 返回 [STOP 1+36+n_objs, 7(position feats)+7(pos_feats)]
        vp_pos_fts = self.get_vp_pos_fts(scan, start_vp, end_vp,
                                         traj_cand_vpids[-1], cur_heading, cur_elevation, len(traj_nav_types[-1]))

        outs = {
            # 1）标注id
            'instr_id': item['instr_id'],

            # 2) instruction的BERT token_ids (max:200长度)
            'instr_encoding': item['instr_encoding'][:self.max_txt_len],

            # 3）gt path上每个viewpoint的全景图像特征, List[ (36,768) ]
            'traj_view_img_fts': [x[:, :self.image_feat_size] for x in traj_view_img_fts],

            # 4) gt_path上每个node的 detected_object_feats, List[ (n_objs, 2048) ]
            'traj_obj_img_fts': [x[:, :self.obj_feat_size] for x in traj_obj_img_fts],

            # 5) gt_path上每个node的location features, List[ (n_views+n_objs, 7=4 angle_feats + 3 box_feats) ]
            'traj_loc_fts': traj_loc_fts,  # List[ (n, 7) ]

            # 6) path中每个node的navigation types, [candidate nodes view_idx 1, non view_idx 0, objs_idx 2]
            'traj_nav_types': traj_nav_types,

            # 7) 每个path node的相邻导航可达节点strings
            'traj_cand_vpids': traj_cand_vpids,

            # 8)
            'traj_vpids': gt_path,

            # 9) stop_token None + 访问过的node_viewpoint_str + 未访问的导航可达node_viewpoint_str
            'gmap_vpids': gmap_vpids,

            # 10) [STOP 0, visited: 1,2,3,..., unvisited: 0,0,...]
            'gmap_step_ids': gmap_step_ids,

            # 11) [STOP 0, visited: 1,1,..., unvisited: 0,0,...]
            'gmap_visited_masks': gmap_visited_masks,

            # 12) gmap: graph map,导航图中每个节点到终点的position features
            'gmap_pos_fts': gmap_pos_fts,

            # 13) 导航图中每个node之间的distances matrix
            'gmap_pair_dists': gmap_pair_dists,

            # 终点处的position features (STOP 1+n_views+n_objs, 7+7) 7(position feats)+7(pos_feats)
            'vp_pos_fts': vp_pos_fts,
            # 'vp_objids': last_vp_objids,

            # 终点处的angles (n_views+n_objs, 2: heading+elevation)
            'vp_angles': last_vp_angles,
        }

        if return_obj_label:
            outs['obj_labels'] = self.get_obj_label(item, last_vp_objids)

        if return_act_label:
            global_act_label, local_act_label = self.get_act_labels(
                end_vp, item, gmap_vpids, gmap_visited_masks, traj_cand_vpids
            )
            outs['global_act_labels'] = global_act_label
            outs['local_act_labels'] = local_act_label

        if return_img_probs:
            # TODO: whether adding gmap img probs # [36,1000]取softmax==>
            outs['vp_view_probs'] = softmax(traj_view_img_fts[-1][:, self.image_feat_size:], dim=1)
            outs['vp_obj_probs'] = softmax(traj_obj_img_fts[-1][:, self.obj_feat_size:], dim=1)  # [n_objs,1601 classes]

        # 增加object name
        if item['obj_pseudo_label']['obj_name'] is not None:
            outs['obj_name'] = item['obj_pseudo_label']['obj_name']
        else:
            outs['obj_name'] = ''

        if item.get('Q1', None) is not None:
            for i in range(5):
                outs['Q{}'.format(i + 1)] = item['Q{}'.format(i + 1)]
                outs['A{}'.format(i + 1)] = item['A{}'.format(i + 1)]

        return outs

    def get_cur_angle(self, scan, path, start_heading):
        if len(path) < 2:
            heading = start_heading
            elevation = 0
        else:
            prev_vp = path[-2]  # path 倒数第二个, 也就是距离 end viewpoint 最临近的node. 记为第k-1个node
            cur_vp = path[-1]  # path end viewpoint, 记为第k个node.
            # self.scanvp_cands['%s_%s'%(scan, prev_vp)]是导航图,
            # self.scanvp_cands['%s_%s'%(scan, prev_vp)][cur_vp] 存储从第k-1个node到第k个node的导航Edge
            # [0] 第k-1个node到第k个node的视图索引(这里的视图是指36张全景视图)
            viewidx = self.scanvp_cands['%s_%s' % (scan, prev_vp)][cur_vp][0]
            # 12个view为360°,每个view的角度范围为30°,view_idx % 12, 余数 x 30°即为视图方位.
            heading = (viewidx % 12) * math.radians(30)

            # elevation表示俯仰角度, 36张view image中, 0-11张是-30度
            # 12-23这12张image的俯仰角view是0度
            elevation = (viewidx // 12 - 1) * math.radians(30)
        return heading, elevation

    def get_traj_pano_fts(self, scan, path):
        '''
        Tokens in each pano: [cand_views, noncand_views, objs]
        Each token consists of (img_fts, loc_fts (ang_fts, box_fts), nav_types)
        '''
        traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids = [], [], [], [], []

        for vp in path:
            # 1. 读取 scan_viewpoint 的features 注意这里调用SoonTextPathData的get_scanvp_feature()方法.
            # (1) view_img_feats [36,1768]; (2) detected object feats; (3) detected object attrs
            view_fts, obj_img_fts, obj_attrs = self.get_scanvp_feature(scan, vp)

            view_img_fts, view_angles, cand_vpids = [], [], []

            # 当前节点的相邻导航可达node: cand views 候选的viewpoints
            # self.scanvp_cands['%s_%s'%(scan, vp)] Dict[字典key为viewpoint_str,]
            # self.scanvp_cands['%s_%s'%(scan, vp)] 为List[view_idx,float,float,float]
            nav_cands = self.scanvp_cands['%s_%s' % (scan, vp)]
            used_viewidxs = set()
            for k, v in nav_cands.items():
                used_viewidxs.add(v[0])  # nav_cands中列表v存储4个元素 [idx,] v[0]表示image_idx,
                view_img_fts.append(view_fts[v[0]])  # view_fts[v[0]]选择36张图像中的第idx张
                # TODO: whether using correct heading at each step
                view_angle = self.all_point_rel_angles[12][v[0]]
                view_angles.append(
                    [view_angle[0] + v[2], view_angle[1] + v[3]])  # v[2] v[3] relative heading, relative elevation
                cand_vpids.append(k)
            # non cand views
            view_img_fts.extend([view_fts[idx] for idx in range(36) if idx not in used_viewidxs])
            view_angles.extend([self.all_point_rel_angles[12][idx] for idx in range(36) if idx not in used_viewidxs])
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)  # (n_views, dim_ft) 36个图像视角,[36,1768]
            view_angles = np.stack(view_angles, 0)  # (n_views, 2) [heading,elevation]

            # VLN-DUET论文: The orientation feature [11] contains
            # sin(·) and cos(·) values for heading and elevation angles.
            # 构建朝向特征.
            view_ang_fts = get_angle_fts(view_angles[:, 0], view_angles[:, 1],
                                         self.angle_feat_size)  # self.angle_feat_size=4
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)  # view无box features, 全置为(1,1,1)

            # object features; 注意detected objects是全景图中的;
            num_objs = obj_img_fts.shape[0]
            obj_angles = np.zeros((num_objs, 2), dtype=np.float32)
            obj_ang_fts = np.zeros((num_objs, self.angle_feat_size), dtype=np.float32)
            obj_box_fts = np.zeros((num_objs, 3), dtype=np.float32)
            if num_objs > 0:
                for k, (w, h) in enumerate(obj_attrs['sizes']):
                    obj_angles[k] = obj_attrs['directions'][k]  # object bounding box的direction angle
                    obj_box_fts[k] = [h / self.obj_image_h, w / self.obj_image_w,
                                      (h * w) / self.obj_image_size]  # obj box的features, 尺寸与面积.
                obj_ang_fts = get_angle_fts(obj_angles[:, 0], obj_angles[:, 1], self.angle_feat_size)

            # combine pano features
            traj_view_img_fts.append(view_img_fts)  # [36,1768] 36个image view. ViT features
            traj_obj_img_fts.append(obj_img_fts)  # [n_objs,3649]
            traj_loc_fts.append(
                np.concatenate(
                    [np.concatenate([view_ang_fts, view_box_fts], 1),
                     np.concatenate([obj_ang_fts, obj_box_fts], 1)], axis=0
                )
            )  # [n_views+n_objs, 7=4+3] 4维度的角度feature(heading,elevation) + 3维度的box_feature(δh,δw,δh*w)
            traj_nav_types.append(
                [1] * len(cand_vpids) + [0] * (36 - len(used_viewidxs)) + [2] * len(obj_img_fts)
            )  # 导航类型 [候选节点置为1, 非候选节点置为0, 检测到的物体置为2]
            traj_cand_vpids.append(cand_vpids)  # candidate_viewpoint_ids, 候选节点idx

            last_vp_objids = obj_attrs.get('obj_ids', [])  # viewpoint_检测到的object的class_ids
            last_vp_angles = np.concatenate([view_angles, obj_angles], 0)

        # 1) traj_view_img_fts, List[], (36,dim=768+1000) 36个view image features
        # 2) traj_obj_img_fts, List[], (n_objs,dim=3649) 全景图中detected objects
        # 3) traj_loc_fts, List[], (view_nums+n_objs,7) view_nums:36,
        #       7=(heading,elevation) 4 angle features + (box_h, box_w, box_h*w) 3 box features
        # 4) traj_nav_types, List[], [1,...,0,...,2,...] candidate nodes view_idx, non view_idx, objects idx
        # 5) traj_cand_vpids, List[], [相邻导航可达node的viewpoint string,]
        # 6) last_vp_angles, array, path[-1] last viewpoint 的 angles,
        #       (view_nums+n_objs, 2 view_angles + 2 obj_angles)
        # 7) last_vp_objids, array, path[-1] last viewpoint 的 object class idx, (n_objs,)
        return traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
            last_vp_angles, last_vp_objids

    def get_gmap_inputs(self, scan, path, cur_heading, cur_elevation):
        scan_graph = self.graphs[scan]
        cur_vp = path[-1]  # gt path终点的viewpoint str.

        visited_vpids, unvisited_vpids = {}, {}

        # 从path[0]遍历到path[-1],记录下路径上所有的未访问过的nodes(即相邻的可导航nodes,但是不在路径上.)
        # path[0]->path[-1], 访问过的nodes
        for t, vp in enumerate(path):
            visited_vpids[vp] = t + 1
            if vp in unvisited_vpids:
                del unvisited_vpids[vp]
            for next_vp in self.scanvp_cands['%s_%s' % (scan, vp)].keys():
                if next_vp not in visited_vpids:
                    unvisited_vpids[next_vp] = 0

        # add [stop] token
        # gmap_vpids: stop_token + 访问过的node_viewpoint_str + 未访问的导航可达node_viewpoint_str
        gmap_vpids = [None] + list(visited_vpids.keys()) + list(unvisited_vpids.keys())

        # 为每个node标注step_id, stop+未访问过的nodes都是0, visited_viewpoint_ids:[1,2,...]
        gmap_step_ids = [0] + list(visited_vpids.values()) + list(unvisited_vpids.values())
        if self.act_visited_node:  # False
            gmap_visited_masks = [0]
            for vp in gmap_vpids[1:]:
                if vp == path[-1]:
                    gmap_visited_masks.append(1)
                else:
                    gmap_visited_masks.append(0)
        else:
            gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)

        # gmap即graph map,连通图上所有节点数即为num_gmap_vpids=STOP+visited+unvisited,
        # shape=(num_gmap_vpids, 7) 相对位置关系.
        # [angle(每个节点到终点的相对heading,elevation的角度特征):4 + dis(每个节点到终点的相对距离,最短路径距离,步长):3]
        gmap_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, gmap_vpids, cur_heading, cur_elevation)

        # 节点i与节点j之间的相对距离matrix.
        gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
        for i in range(1, len(gmap_vpids)):
            for j in range(i + 1, len(gmap_vpids)):
                gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                    self.shortest_distances[scan][gmap_vpids[i]][gmap_vpids[j]]

        return gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists

    def get_gmap_pos_fts(self, scan, cur_vp, gmap_vpids, cur_heading, cur_elevation):
        # dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation),
        #  line_dist, shortest_dist, shortest_step)
        rel_angles, rel_dists = [], []
        for vp in gmap_vpids:
            if vp is None:
                rel_angles.append([0, 0])
                rel_dists.append([0, 0, 0])
            else:
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    self.graphs[scan].nodes[cur_vp]['position'],
                    self.graphs[scan].nodes[vp]['position'],
                    base_heading=cur_heading, base_elevation=cur_elevation,
                )  # 计算路径上其他viewpoint到终点的相对heading,elevation,distance;
                rel_angles.append([rel_heading, rel_elevation])
                rel_dists.append(
                    [rel_dist / MAX_DIST, self.shortest_distances[scan][cur_vp][vp] / MAX_DIST, \
                     (len(self.shortest_paths[scan][cur_vp][vp]) - 1) / MAX_STEP]
                )  # 相对距离/30, 最短路径法的相对距离/30, 最短steps/10
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], self.angle_feat_size)
        return np.concatenate([rel_ang_fts, rel_dists], 1)

    def get_vp_pos_fts(self, scan, start_vp, cur_vp, cand_vpids, cur_heading, cur_elevation, vp_ft_len):
        # 1. cur_vp为终点的viewpoint, cand_vpids为终点的相邻nodes,
        # 即求终点处相邻nodes的位置features: dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation),
        #  line_dist, shortest_dist, shortest_step)
        cur_cand_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, cand_vpids, cur_heading, cur_elevation)

        # 2. 终点与起点之间的position features.
        cur_start_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, [start_vp], cur_heading, cur_elevation)

        # add [stop] token at beginning
        # vp_ft_len: 终点处 candidate nodes view_idx, non view idx, object idx 的总长度=36+n_objs
        vp_pos_fts = np.zeros((vp_ft_len + 1, 14), dtype=np.float32)
        vp_pos_fts[:, :7] = cur_start_pos_fts
        vp_pos_fts[1:len(cur_cand_pos_fts) + 1, 7:] = cur_cand_pos_fts

        return vp_pos_fts
