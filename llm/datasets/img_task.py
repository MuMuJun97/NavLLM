import random
import math
import numpy as np
import re
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from .common import pad_tensors, gen_seq_masks


############### Masked Region Modeling ###############
def _get_img_mask(mask_prob, num_images):
    img_mask = [np.random.rand() < mask_prob for _ in range(num_images)]
    if not any(img_mask):
        # at least mask 1
        img_mask[np.random.randint(num_images)] = True
    img_mask = torch.tensor(img_mask)
    return img_mask

def _mask_img_feat(img_feat, img_masks):
    img_masks_ext = img_masks.unsqueeze(-1).expand_as(img_feat)
    img_feat_masked = img_feat.data.masked_fill(img_masks_ext, 0)
    return img_feat_masked

def _get_targets(img_soft_label, img_masks):
    soft_label_dim = img_soft_label.size(-1)
    img_masks_ext_for_label = img_masks.unsqueeze(-1).expand_as(img_soft_label)
    label_targets = img_soft_label[img_masks_ext_for_label].contiguous().view(-1, soft_label_dim)
    return label_targets

class BaseDataset(Dataset):
    def __init__(self,tokenizer,training,source_len,summ_len):
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.summ_len = summ_len
        self.training = training

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def process_text_inputs_task(self,inputs,index,input_format="QCM"):
        if self.training:
            qi = np.random.randint(0,5) + 1 # idx % 5 + 1
        else:
            qi = index % 5 + 1
        source_text = inputs['Q{}'.format(str(qi))]
        target_text = inputs['A{}'.format(str(qi))]
        target_text = "Answer: {}".format(target_text)


        instruction = source_text[:source_text.find('Question')].replace("Context: ","")
        question = source_text[source_text.find('Question'):source_text.find('Options')]
        options = source_text[source_text.find('Options'):source_text.find('Answer')]

        # cleaning data so as to ensure data is in string type
        instruction = " ".join(instruction.split())
        question = " ".join(question.split())
        options = " ".join(options.split())
        target = " ".join(target_text.split())

        if input_format == "QCM":
            # 给定Question+Multiple Choices,预测指令.
            # context = "Context: N/A"
            input_text = "{} {}. {} Instruction:".format(question,options,target)
            output_text = "Instruction: {}".format(instruction)
            input_text = " ".join(input_text.split())
            output_text = " ".join(output_text.split())
        elif input_format == "IQCM":
            # context = "Context: N/A"
            input_text = "Instruction: {} {} {}. Answer:".format(instruction,question,options)
            output_text = target
            input_text = " ".join(input_text.split())
            output_text = " ".join(output_text.split())
        elif input_format == "VL":
            input_text = "Instruction: {} {} {}. {}".format(instruction,question,options,target)
            input_text = " ".join(input_text.split())
            output_text = " ".join(input_text.split())
        else:
            raise NotImplementedError

        source = self.tokenizer.batch_encode_plus(
            [input_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [output_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze() # [len,] tensor
        source_mask = source["attention_mask"].squeeze() # [len,] tensor
        target_ids = target["input_ids"].squeeze() # [len,]

        output = {}

        output['input_ids'] = source_ids
        output['attention_mask'] = source_mask
        output['labels'] = target_ids

        return output

class MrcDataset(BaseDataset):
    def __init__(self, nav_db, tok, mask_prob, end_vp_pos_ratio=1,
                 training=False, source_len=192, summ_len=16):
        super().__init__(tokenizer=tok,training=training,
                         source_len=source_len,summ_len=summ_len)
        self.nav_db = nav_db
        self.training = training
        self.mask_prob = mask_prob # 0.15

        self.end_vp_pos_ratio = end_vp_pos_ratio # 1

        self.source_len = source_len
        self.summ_len = summ_len

    def __len__(self):
        return len(self.nav_db.data)

    def __getitem__(self, idx):
        r = np.random.rand()
        if r < self.end_vp_pos_ratio:
            end_vp_type = 'pos'
        else:
            end_vp_type = 'neg_in_gt_path'
        inputs = self.nav_db.get_input(idx, end_vp_type, return_img_probs=True)

        # BaseDataset: process text inputs
        output = self.process_text_inputs_task(inputs,index=idx,input_format="VL")

        output['traj_view_img_fts'] = [torch.from_numpy(x) for x in inputs['traj_view_img_fts']] # List[12*(36,768)]

        # mask image 终点节点的36 view image features,
        view_mrc_masks = _get_img_mask(self.mask_prob, len(output['traj_view_img_fts'][-1]))

        output['traj_view_img_fts'][-1] = _mask_img_feat(output['traj_view_img_fts'][-1], view_mrc_masks)
        output['vp_view_probs'] = torch.from_numpy(inputs['vp_view_probs'])  # no [stop]
        output['vp_view_mrc_masks'] = view_mrc_masks
        output['traj_loc_fts'] = [torch.from_numpy(x) for x in inputs['traj_loc_fts']]
        output['traj_nav_types'] = [torch.LongTensor(x) for x in inputs['traj_nav_types']]
        output['traj_cand_vpids'] = inputs['traj_cand_vpids']
        output['traj_vpids'] = inputs['traj_vpids']

        output['gmap_vpids'] = inputs['gmap_vpids']
        output['gmap_step_ids'] = torch.LongTensor(inputs['gmap_step_ids'])
        output['gmap_visited_masks'] = torch.BoolTensor(inputs['gmap_visited_masks'])
        output['gmap_pos_fts'] = torch.from_numpy(inputs['gmap_pos_fts'])
        output['gmap_pair_dists'] = torch.from_numpy(inputs['gmap_pair_dists'])

        output['vp_pos_fts'] = torch.from_numpy(inputs['vp_pos_fts'])
        output['vp_angles'] = inputs['vp_angles']

        if 'traj_obj_img_fts' in inputs:
            output['traj_obj_img_fts'] = [torch.from_numpy(x) for x in inputs['traj_obj_img_fts']]
            if len(output['traj_obj_img_fts'][-1]) > 0:
                obj_mrc_masks = _get_img_mask(self.mask_prob, len(output['traj_obj_img_fts'][-1]))
                output['traj_obj_img_fts'][-1] = _mask_img_feat(output['traj_obj_img_fts'][-1], obj_mrc_masks)
            else:
                obj_mrc_masks = torch.zeros(0, ).bool()
            output['vp_obj_probs'] = torch.from_numpy(inputs['vp_obj_probs'])
            output['vp_obj_mrc_masks'] = obj_mrc_masks

        return output


def mrc_collate(inputs):
    batch = {
        k: [x[k] for x in inputs] for k in inputs[0].keys()
    }
    # text batches
    batch['input_ids'] = torch.stack(batch['input_ids'])
    batch['attention_mask'] = torch.stack(batch['attention_mask'])
    batch['txt_lens'] = torch.LongTensor([
        x.sum().item() for x in batch['attention_mask']
    ])

    # # text batches
    # batch['txt_lens'] = torch.LongTensor([len(x) for x in batch['txt_ids']])
    # batch['txt_ids'] = pad_sequence(batch['txt_ids'], batch_first=True, padding_value=0)

    # trajectory batches: traj_cand_vpids, traj_vpids
    batch['traj_step_lens'] = [len(x) for x in batch['traj_view_img_fts']]
    batch['traj_vp_view_lens'] = torch.LongTensor(
        sum([[len(y) for y in x] for x in batch['traj_view_img_fts']], [])
    )
    batch['traj_view_img_fts'] = pad_tensors(sum(batch['traj_view_img_fts'], []))
    batch['traj_loc_fts'] = pad_tensors(sum(batch['traj_loc_fts'], []))
    batch['traj_nav_types'] = pad_sequence(sum(batch['traj_nav_types'], []), batch_first=True, padding_value=0)

    # gmap batches: gmap_vpids
    batch['gmap_lens'] = torch.LongTensor([len(x) for x in batch['gmap_step_ids']])  # included [stop]
    batch['gmap_step_ids'] = pad_sequence(batch['gmap_step_ids'], batch_first=True, padding_value=0)
    batch['gmap_visited_masks'] = pad_sequence(batch['gmap_visited_masks'], batch_first=True, padding_value=0)
    batch['gmap_pos_fts'] = pad_tensors(batch['gmap_pos_fts'])
    max_gmap_len = max(batch['gmap_lens'])
    batch_size = len(batch['gmap_lens'])
    gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
    for i in range(batch_size):
        gmap_pair_dists[i, :batch['gmap_lens'][i], :batch['gmap_lens'][i]] = batch['gmap_pair_dists'][i]
    batch['gmap_pair_dists'] = gmap_pair_dists

    # vp batches: vp_angles
    batch['vp_lens'] = torch.LongTensor([len(x[-1]) for x in batch['vp_pos_fts']])  # included [stop]
    batch['vp_pos_fts'] = pad_tensors(batch['vp_pos_fts'])

    # vp labels
    batch['vp_view_mrc_masks'] = pad_sequence(batch['vp_view_mrc_masks'], batch_first=True, padding_value=0)
    batch['vp_view_probs'] = pad_tensors(batch['vp_view_probs'])

    if 'traj_obj_img_fts' in batch:
        batch['traj_vp_obj_lens'] = torch.LongTensor(
            sum([[len(y) for y in x] for x in batch['traj_obj_img_fts']], [])
        )
        batch['traj_obj_img_fts'] = pad_tensors(sum(batch['traj_obj_img_fts'], []))
        batch['vp_obj_mrc_masks'] = pad_sequence(batch['vp_obj_mrc_masks'], batch_first=True, padding_value=0)
        batch['vp_obj_probs'] = pad_tensors(batch['vp_obj_probs'])

    return batch


############### Single-step Action Prediction ###############
class SapDataset(BaseDataset):
    def __init__(self, nav_db, tok, end_vp_pos_ratio=0.2,
                 training=False, source_len=192, summ_len=16):
        '''Instruction Trajectory Matching'''
        super().__init__(tokenizer=tok,training=training,
                         source_len=source_len,summ_len=summ_len)
        self.nav_db = nav_db

        self.training = training

        self.end_vp_pos_ratio = end_vp_pos_ratio
        self.source_len = source_len
        self.summ_len = summ_len

    def __len__(self):
        return len(self.nav_db.data)

    def __getitem__(self, idx):
        r = np.random.rand()
        if r < self.end_vp_pos_ratio:
            end_vp_type = 'pos'
        elif r < 0.6:
            end_vp_type = 'neg_in_gt_path'
        else:
            end_vp_type = 'neg_others'
        inputs = self.nav_db.get_input(idx, end_vp_type, return_act_label=True)

        # BaseDataset: process text inputs
        output = self.process_text_inputs_task(inputs,index=idx,input_format="VL")

        output['traj_view_img_fts'] = [torch.from_numpy(x) for x in inputs['traj_view_img_fts']]
        if 'traj_obj_img_fts' in inputs:
            output['traj_obj_img_fts'] = [torch.from_numpy(x) for x in inputs['traj_obj_img_fts']]
        output['traj_loc_fts'] = [torch.from_numpy(x) for x in inputs['traj_loc_fts']]
        output['traj_nav_types'] = [torch.LongTensor(x) for x in inputs['traj_nav_types']]
        output['traj_cand_vpids'] = inputs['traj_cand_vpids']
        output['traj_vpids'] = inputs['traj_vpids']

        output['gmap_vpids'] = inputs['gmap_vpids']
        output['gmap_step_ids'] = torch.LongTensor(inputs['gmap_step_ids'])
        output['gmap_visited_masks'] = torch.BoolTensor(inputs['gmap_visited_masks'])
        output['gmap_pos_fts'] = torch.from_numpy(inputs['gmap_pos_fts'])
        output['gmap_pair_dists'] = torch.from_numpy(inputs['gmap_pair_dists'])

        output['vp_pos_fts'] = torch.from_numpy(inputs['vp_pos_fts'])
        output['vp_angles'] = inputs['vp_angles']

        output['local_act_labels'] = inputs['local_act_labels']
        output['global_act_labels'] = inputs['global_act_labels']
        return output

def sap_collate(inputs):
    batch = {
        k: [x[k] for x in inputs] for k in inputs[0].keys()
    }

    # text batches
    batch['input_ids'] = torch.stack(batch['input_ids'])
    batch['attention_mask'] = torch.stack(batch['attention_mask'])
    batch['txt_lens'] = torch.LongTensor([
        x.sum().item() for x in batch['attention_mask']
    ])

    # # text batches
    # batch['txt_lens'] = torch.LongTensor([len(x) for x in batch['txt_ids']])
    # batch['txt_ids'] = pad_sequence(batch['txt_ids'], batch_first=True, padding_value=0)

    # trajectory batches: traj_cand_vpids, traj_vpids
    batch['traj_step_lens'] = [len(x) for x in batch['traj_view_img_fts']]
    batch['traj_vp_view_lens'] = torch.LongTensor(
        sum([[len(y) for y in x] for x in batch['traj_view_img_fts']], [])
    )
    batch['traj_view_img_fts'] = pad_tensors(sum(batch['traj_view_img_fts'], []))
    if 'traj_obj_img_fts' in batch:
        batch['traj_vp_obj_lens'] = torch.LongTensor(
            sum([[len(y) for y in x] for x in batch['traj_obj_img_fts']], [])
        )
        batch['traj_obj_img_fts'] = pad_tensors(sum(batch['traj_obj_img_fts'], []))
    batch['traj_loc_fts'] = pad_tensors(sum(batch['traj_loc_fts'], []))
    batch['traj_nav_types'] = pad_sequence(sum(batch['traj_nav_types'], []), batch_first=True, padding_value=0)

    # gmap batches: gmap_vpids
    batch['gmap_lens'] = torch.LongTensor([len(x) for x in batch['gmap_step_ids']]) # included [stop]
    batch['gmap_step_ids'] = pad_sequence(batch['gmap_step_ids'], batch_first=True, padding_value=0)
    batch['gmap_visited_masks'] = pad_sequence(batch['gmap_visited_masks'], batch_first=True, padding_value=0)
    batch['gmap_pos_fts'] = pad_tensors(batch['gmap_pos_fts'])
    max_gmap_len = max(batch['gmap_lens'])
    batch_size = len(batch['gmap_lens'])
    gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
    for i in range(batch_size):
        gmap_pair_dists[i, :batch['gmap_lens'][i], :batch['gmap_lens'][i]] = batch['gmap_pair_dists'][i]
    batch['gmap_pair_dists'] = gmap_pair_dists

    # vp batches: vp_angles
    batch['vp_lens'] = torch.LongTensor([len(x[-1]) for x in batch['vp_pos_fts']])  # included [stop]
    batch['vp_pos_fts'] = pad_tensors(batch['vp_pos_fts'])

    # action labels
    batch['local_act_labels'] = torch.LongTensor(batch['local_act_labels'])
    batch['global_act_labels'] = torch.LongTensor(batch['global_act_labels'])
    return batch


############### Object Grounding ###############
class OGDataset(BaseDataset):
    def __init__(self, nav_db, tok,
                 training=False, source_len=192, summ_len=16):
        super().__init__(tokenizer=tok,training=training,
                         source_len=source_len,summ_len=summ_len)
        self.nav_db = nav_db

    def __len__(self):
        return len(self.nav_db.data)

    def __getitem__(self, idx):
        inputs = self.nav_db.get_input(idx, 'pos', return_obj_label=True)

        # BaseDataset: process text inputs
        output = self.process_text_inputs_task(inputs,index=idx)

        output['traj_view_img_fts'] = [torch.from_numpy(x) for x in inputs['traj_view_img_fts']]
        output['traj_obj_img_fts'] = [torch.from_numpy(x) for x in inputs['traj_obj_img_fts']]
        output['traj_loc_fts'] = [torch.from_numpy(x) for x in inputs['traj_loc_fts']]
        output['traj_nav_types'] = [torch.LongTensor(x) for x in inputs['traj_nav_types']]
        output['traj_cand_vpids'] = inputs['traj_cand_vpids']
        output['traj_vpids'] = inputs['traj_vpids']

        output['gmap_vpids'] = inputs['gmap_vpids']
        output['gmap_step_ids'] = torch.LongTensor(inputs['gmap_step_ids'])
        output['gmap_visited_masks'] = torch.BoolTensor(inputs['gmap_visited_masks'])
        output['gmap_pos_fts'] = torch.from_numpy(inputs['gmap_pos_fts'])
        output['gmap_pair_dists'] = torch.from_numpy(inputs['gmap_pair_dists'])

        output['vp_pos_fts'] = torch.from_numpy(inputs['vp_pos_fts'])
        output['vp_angles'] = inputs['vp_angles']

        output['obj_labels'] = inputs['obj_labels']
        return output

def og_collate(inputs):
    batch = {
        k: [x[k] for x in inputs] for k in inputs[0].keys()
    }

    # text batches
    batch['input_ids'] = torch.stack(batch['input_ids'])
    batch['attention_mask'] = torch.stack(batch['attention_mask'])
    batch['txt_lens'] = torch.LongTensor([
        x.sum().item() for x in batch['attention_mask']
    ])

    # # text batches
    # batch['txt_lens'] = torch.LongTensor([len(x) for x in batch['txt_ids']])
    # batch['txt_ids'] = pad_sequence(batch['txt_ids'], batch_first=True, padding_value=0)

    # trajectory batches: traj_cand_vpids, traj_vpids
    batch['traj_step_lens'] = [len(x) for x in batch['traj_view_img_fts']]
    batch['traj_vp_view_lens'] = torch.LongTensor(
        sum([[len(y) for y in x] for x in batch['traj_view_img_fts']], [])
    )
    batch['traj_vp_obj_lens'] = torch.LongTensor(
        sum([[len(y) for y in x] for x in batch['traj_obj_img_fts']], [])
    )
    batch['traj_view_img_fts'] = pad_tensors(sum(batch['traj_view_img_fts'], []))
    batch['traj_obj_img_fts'] = pad_tensors(sum(batch['traj_obj_img_fts'], []))
    batch['traj_loc_fts'] = pad_tensors(sum(batch['traj_loc_fts'], []))
    batch['traj_nav_types'] = pad_sequence(sum(batch['traj_nav_types'], []), batch_first=True, padding_value=0)

    # gmap batches: gmap_vpids
    batch['gmap_lens'] = torch.LongTensor([len(x) for x in batch['gmap_step_ids']]) # included [stop]
    batch['gmap_step_ids'] = pad_sequence(batch['gmap_step_ids'], batch_first=True, padding_value=0)
    batch['gmap_visited_masks'] = pad_sequence(batch['gmap_visited_masks'], batch_first=True, padding_value=0)
    batch['gmap_pos_fts'] = pad_tensors(batch['gmap_pos_fts'])
    max_gmap_len = max(batch['gmap_lens'])
    batch_size = len(batch['gmap_lens'])
    gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
    for i in range(batch_size):
        gmap_pair_dists[i, :batch['gmap_lens'][i], :batch['gmap_lens'][i]] = batch['gmap_pair_dists'][i]
    batch['gmap_pair_dists'] = gmap_pair_dists

    # vp batches: vp_angles
    batch['vp_lens'] = torch.LongTensor([len(x[-1]) for x in batch['vp_pos_fts']])  # included [stop]
    batch['vp_pos_fts'] = pad_tensors(batch['vp_pos_fts'])

    # vp labels
    batch['obj_labels'] = torch.LongTensor(batch['obj_labels'])
    return batch