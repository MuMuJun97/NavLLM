import os
import sys
# if int(sys.version[2]) == 6:
#     sys.path.append('/home/zlin/vln/points/Matterport3DSimulator/build')
#     print(os.environ['PYTHONPATH'])
# elif int(sys.version[2]) >  6:
#     sys.path.append('/home/zlin/vln/turning/Matterport3DSimulator/build')
#     print(os.environ['PYTHONPATH'])
# sys.path.append('/mnt/petrelfs/zhaolin/vln/mp3d/Matterport3DSimulator-Centos7/build')
import MatterSim
import json
import argparse
import time
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.cuda.amp as amp   # TODO

from transformers import AutoTokenizer, PretrainedConfig
from transformers import AutoModel

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, set_dropout, set_random_seed, set_cuda, wrap_model
from utils.distributed import all_gather

from optim import get_lr_sched
from optim.misc import build_optimizer

from utils.aparser import load_parser, parse_with_config

from val import validate

# dataset
from datasets.task import build_dataset

def main(opts):
    default_gpu, n_gpu, device = set_cuda(opts)
    if default_gpu:
        LOGGER.info(
            'device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}'.format(
                device, n_gpu, bool(opts.local_rank != -1), opts.fp16
            )
        )
    seed = opts.seed
    if opts.local_rank != -1:
        seed += opts.rank
    set_random_seed(seed)

    if default_gpu:
        save_training_meta(opts) # 保存模型文件到本地
        TB_LOGGER.create(os.path.join(opts.output_dir, 'logs'))
        pbar = tqdm(total=opts.num_train_steps,position=0,leave=True) # 训练步长
        model_saver = ModelSaver(os.path.join(opts.output_dir, 'ckpts'))
        add_log_to_file(os.path.join(opts.output_dir, 'logs', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    # dataset dataloader
    meta_loader, train_dataloaders, val_dataloaders, val2_dataloaders = build_dataset(opts,device)

    # model
    model_config = PretrainedConfig.from_json_file(opts.model_config)
    model_config.pretrain_tasks = list()
    model_config.output_config = opts.output_config
    for train_dataset_config in opts.train_datasets.values():
        model_config.pretrain_tasks.extend(train_dataset_config['tasks'])

    ## load checkpoint

    ## model initialize
    if 't5' in opts.Net:
        from models.t5_models import GlocalTextPathCMTPreTraining
        model_config.Net = opts.Net
        model = GlocalTextPathCMTPreTraining(model_config)
    else:
        from models.pretrain_models import GlocalTextPathCMTPreTraining
        model = GlocalTextPathCMTPreTraining(config=model_config)

    model.train()
    set_dropout(model, opts.dropout)
    model = wrap_model(model, device, opts.local_rank)

    # Prepare optimizer
    optimizer = build_optimizer(model, opts)

    global_step = 0
    LOGGER.info(f"***** Running training with {opts.world_size} GPUs *****")
    LOGGER.info("  Batch size = %d",
                opts.train_batch_size if opts.local_rank == -1 else opts.train_batch_size * opts.world_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    # to compute training statistics
    task2loss = {task: RunningMeter(f'loss/{task}')
                 for task in train_dataloaders.keys()}

    n_examples = defaultdict(int)
    n_loss_units = defaultdict(int)

    start_time = time.time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()

    # validate(model, val_dataloaders, setname='_seen')

    for step, name_batch in enumerate(meta_loader):
        name, batch = name_batch

        # forward pass
        n_examples[name] += batch['input_ids'].size(0)
        task = name.split('_')[0]

        loss = model(batch, task=task, compute_loss=True)

        n_loss_units[name] += 192
        # loss = loss.mean()  # loss is not normalized in model

        # backward pass
        if args.gradient_accumulation_steps > 1:  # average loss
            loss = loss / args.gradient_accumulation_steps

        loss.backward()
        task2loss[name](loss.item())

        # optimizer update and logging
        if (step + 1) % opts.gradient_accumulation_steps == 0:
            global_step += 1
            # learning rate scheduling
            lr_this_step = get_lr_sched(global_step, opts)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

            # log loss
            # NOTE: not gathered across GPUs for efficiency
            TB_LOGGER.log_scalar_dict({ll.name: ll.val
                                       for ll in task2loss.values()
                                       if ll.val is not None})

            TB_LOGGER.step()

            # update model params
            if opts.grad_norm != -1:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), opts.grad_norm
                )
                TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_postfix({
                'loss': loss.item(),
                'lr': lr_this_step,
                'task': task,
            })
            pbar.update(1)

            if global_step % opts.log_steps == 0:
                # monitor training throughput
                LOGGER.info(f'==============Step {global_step}===============')
                for t in train_dataloaders.keys():
                    tot_ex = n_examples[t]
                    ex_per_sec = int(tot_ex / (time.time() - start_time))
                    tot_l = n_loss_units[t]
                    l_per_sec = int(tot_l / (time.time() - start_time))
                    LOGGER.info(f'{t}: {tot_ex} examples trained at '
                                f'{ex_per_sec} ex/s')
                    TB_LOGGER.add_scalar(f'perf/{t}_ex_per_s', ex_per_sec,
                                         global_step)
                    TB_LOGGER.add_scalar(f'perf/{t}_loss_per_s', l_per_sec,
                                         global_step)
                LOGGER.info('===============================================')

            if global_step % opts.valid_steps == 0:
                LOGGER.info(f'------Step {global_step}: start validation seen------')
                validate(model, val_dataloaders, setname='_seen')
                LOGGER.info(f'------Step {global_step}: start validation unseen------')
                validate(model, val2_dataloaders, setname='_unseen')
                model_saver.save(model, global_step)

        if global_step >= opts.num_train_steps:
            break

def build_args():
    parser = load_parser()

    opts = parse_with_config(parser)

    if os.path.exists(opts.output_dir) and os.listdir(opts.output_dir):
        LOGGER.warning(
            "Output directory ({}) already exists and is not empty.".format(
                opts.output_dir
            )
        )
    LOGGER.info("Language Model: {}".format(opts.Net))

    return opts

if __name__ == '__main__':
    args = build_args()
    main(args)