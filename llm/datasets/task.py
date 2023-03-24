import random
import math
import numpy as np
import re
import torch
from easydict import EasyDict
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils.logger import LOGGER
from .common import pad_tensors, gen_seq_masks
from .loader import build_dataloader,MetaLoader, PrefetchLoader
from .dataset import SoonTextPathData
from .img_task import BaseDataset,\
    MrcDataset,mrc_collate,\
    SapDataset,sap_collate,\
    OGDataset,og_collate
from transformers import AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration

class QADataset(BaseDataset):
    def __init__(self, nav_db, tok, args=None,
                 training=False, source_len=192, summ_len=16):
        super().__init__(tokenizer=tok,training=training,
                         source_len=source_len,summ_len=summ_len)
        self.nav_db = nav_db  # SOON数据集
        self.training = training
        self.source_len = source_len
        self.summ_len = summ_len
        self.res_dict = {'A':0,'B':1,'C':2}

    def __len__(self):
        return len(self.nav_db)

    def __getitem__(self, idx):
        inputs = self.nav_db.get_input(idx, 'pos')

        # BaseDataset: process text inputs
        output = self.process_text_inputs_task(inputs,index=idx)

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
        return output

class QADatasetInstr(BaseDataset):
    def __init__(self, nav_db, tok, args=None,
                 training=False, source_len=192, summ_len=16):
        super().__init__(tokenizer=tok,training=training,
                         source_len=source_len,summ_len=summ_len)
        self.nav_db = nav_db  # SOON数据集
        self.training = training
        self.source_len = source_len
        self.summ_len = summ_len
        self.res_dict = {'A':0,'B':1,'C':2}

    def __len__(self):
        return len(self.nav_db)

    def __getitem__(self, idx):
        inputs = self.nav_db.get_input(idx, 'pos')

        # BaseDataset: process text inputs
        output = self.process_text_inputs_task(inputs,index=idx,input_format="IQCM")

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
        return output

def QA_collate(inputs):
    batch = {
        k: [x[k] for x in inputs] for k in inputs[0].keys()
    }
    # text batches
    batch['input_ids'] = torch.stack(batch['input_ids'])
    batch['attention_mask'] = torch.stack(batch['attention_mask'])
    batch['labels'] = torch.stack(batch['labels'])

    if batch.get('mask_label',None) is not None:
        batch['mask_label'] = torch.stack(batch['mask_label'])

    batch['txt_lens'] = torch.LongTensor([
        x.sum().item() for x in batch['attention_mask']
    ])
    # batch['ans_labels'] = torch.stack(batch['ans_labels'])

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

    return batch

def create_dataloaders(data_cfg, nav_db, tok, is_train: bool, device: torch.device, opts):
    dataloaders = {}
    for k, task_name in enumerate(data_cfg.tasks):
        if task_name == 'qa':
            task_dataset = QADataset(nav_db, tok, training=is_train)
            task_collate_fn = QA_collate
        elif task_name == 'instr':
            task_dataset = QADatasetInstr(nav_db, tok, training=is_train)
            task_collate_fn = QA_collate
        elif task_name == 'mrc':
            task_dataset = MrcDataset(nav_db, tok, opts.mrc_mask_prob, training=is_train)
            task_collate_fn = mrc_collate
        elif task_name == 'sap':
            task_dataset = SapDataset(nav_db, tok, training=is_train)
            task_collate_fn = sap_collate
        elif task_name == 'og':
            task_dataset = OGDataset(nav_db, tok, training=is_train)
            task_collate_fn = og_collate
        else:
            raise ValueError(f'Undefined task {task_name}')


        LOGGER.info("{}, {}: {} samples loaded".format(nav_db.split,task_name,len(task_dataset)))

        task_loader, pre_epoch = build_dataloader(
            task_name, task_dataset, task_collate_fn, is_train, opts
        )

        # if task_name == 'mrc':
        #     for idx, data in enumerate(task_loader):
        #         print(idx)

        if is_train:
            ratio = data_cfg.mix_ratio[k]
            dataloaders[task_name] = (task_loader, ratio, pre_epoch)
        else:
            dataloaders[task_name] = PrefetchLoader(task_loader, device)

    return dataloaders

def build_dataset(opts, device, val=True):
    # 1. "facebook/opt-iml-max-1.3b"
    #   Contrary to GPT2, OPT adds the EOS token </s> to the beginning of every prompt.
    #   Note: Make sure to pass use_fast=False when loading OPT’s tokenizer with AutoTokenizer to get the correct tokenizer.
    # 2. "google/flan-t5-small"
    #
    if 't5' in opts.Net:
        tokenizer = T5Tokenizer.from_pretrained(opts.Net)
    else:
        tokenizer = AutoTokenizer.from_pretrained(opts.Net, use_fast=False)

    # load data training set
    data_cfg = EasyDict(opts.train_datasets['SOON'])
    train_nav_db = SoonTextPathData(
        data_cfg.train_traj_files, data_cfg.img_ft_file, data_cfg.obj_ft_file,
        data_cfg.scanvp_cands_file, data_cfg.connectivity_dir,
        max_txt_len=opts.max_txt_len, # 200
        max_objects=opts.max_objects, # 100
        in_memory=True
    )
    if val:
        val_nav_db = SoonTextPathData(
            data_cfg.val_seen_traj_files, data_cfg.img_ft_file, data_cfg.obj_ft_file,
            data_cfg.scanvp_cands_file, data_cfg.connectivity_dir,
            max_txt_len=opts.max_txt_len, max_objects=opts.max_objects, in_memory=True
        )
        val2_nav_db = SoonTextPathData(
            data_cfg.val_unseen_traj_files, data_cfg.img_ft_file, data_cfg.obj_ft_file,
            data_cfg.scanvp_cands_file, data_cfg.connectivity_dir,
            max_txt_len=opts.max_txt_len, max_objects=opts.max_objects, in_memory=True
        )

    # Build data loaders
    train_dataloaders = create_dataloaders(
        data_cfg, train_nav_db, tokenizer, True, device, opts
    )
    if val:
        val_dataloaders = create_dataloaders(
            data_cfg, val_nav_db, tokenizer, False, device, opts
        )
        val2_dataloaders = create_dataloaders(
            data_cfg, val2_nav_db, tokenizer, False, device, opts
        )

    meta_loader = MetaLoader(
        train_dataloaders,
        accum_steps=opts.gradient_accumulation_steps,
        distributed=opts.local_rank != -1,
        device=device
    )
    meta_loader = PrefetchLoader(meta_loader, device)
    return meta_loader, train_dataloaders, val_dataloaders, val2_dataloaders
