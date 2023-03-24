import torch
import torch.nn.functional as F
import time
from utils.logger import LOGGER, TB_LOGGER
from utils.distributed import all_gather
from tqdm import tqdm


@torch.no_grad()
def validate_qa(model, val_loader):
    LOGGER.info("start running QA validation...")
    val_loss = 0
    n_word = 0
    n_acc = 0
    n_count = 0

    st = time.time()
    val_bar = tqdm(val_loader, desc="validate QA ...", position=0, leave=True)
    for i,batch in enumerate(val_bar):
        logits = model(batch, task='qa', compute_loss=False)
        labels = batch['labels']

        attention_mask = batch['attention_mask']

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='mean')
        val_loss += loss.item()
        n_word += labels.shape[0] * labels.shape[1]

        batch_size = batch['input_ids'].shape[0]
        for b in range(batch_size):
            cur_correct = (logits.max(dim=-1).indices[b] == labels[b])[attention_mask[b]==True].sum().item()
            cur_word = (attention_mask[b]==True).sum().item()
            cur_acc = cur_correct/cur_word
            n_acc += cur_acc
        n_count += batch_size

        val_bar.set_postfix({
            'loss': val_loss/(i+1),
            'cur_acc': cur_acc,
            'n_acc': n_acc/n_count
        })

    val_loss = sum(all_gather(val_loss))
    n_acc = sum(all_gather(n_acc))

    tot_time = time.time() - st
    val_loss /= (i+1)
    acc = n_acc / n_count

    val_log = {'loss': val_loss,
               'acc': acc,
               'tok_per_s': n_word / tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"acc: {acc * 100:.2f}%")
    LOGGER.info("val_loss: {:.2f}, acc({:.2f}%) ".format(val_loss,acc*100))
    return val_log

@torch.no_grad()
def validate_instr(model, val_loader):
    LOGGER.info("start running Instr validation...")
    val_loss = 0
    n_word = 0
    n_acc = 0
    n_count = 0

    st = time.time()
    val_bar = tqdm(val_loader, desc="validate Instr ...", position=0, leave=True)
    for i,batch in enumerate(val_bar):
        logits = model(batch, task='instr', compute_loss=False)
        labels = batch['labels']

        attention_mask = batch['attention_mask']

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='mean')
        val_loss += loss.item()
        n_word += labels.shape[0] * labels.shape[1]

        batch_size = batch['input_ids'].shape[0]
        for b in range(batch_size):
            cur_correct = (logits.max(dim=-1).indices[b] == labels[b])[attention_mask[b]==True].sum().item()
            cur_word = (attention_mask[b]==True).sum().item()
            cur_acc = cur_correct/cur_word
            n_acc += cur_acc
        n_count += batch_size

        val_bar.set_postfix({
            'loss': val_loss/(i+1),
            'cur_acc': cur_acc,
            'n_acc': n_acc/n_count
        })

    val_loss = sum(all_gather(val_loss))
    n_acc = sum(all_gather(n_acc))

    tot_time = time.time() - st
    val_loss /= (i+1)
    acc = n_acc / n_count

    val_log = {'loss': val_loss,
               'acc': acc,
               'tok_per_s': n_word / tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"acc: {acc * 100:.2f}%")
    LOGGER.info("val_loss: {:.2f}, acc({:.2f}%) ".format(val_loss,acc*100))
    return val_log


def compute_accuracy_for_soft_targets(out, labels):
    outputs = out.max(dim=-1)[1]
    labels = labels.max(dim=-1)[1]  # argmax
    n_correct = (outputs == labels).sum().item()
    return n_correct

@torch.no_grad()
def validate_mrc(model, val_loader):
    LOGGER.info("start running MRC validation...")
    val_loss = 0
    n_feat = 0
    st = time.time()
    tot_score = 0
    val_bar = tqdm(val_loader, desc="validate MRC ...", position=0, leave=True)
    for i,batch in enumerate(val_bar):
        view_logits, view_targets, obj_logits, obj_targets = \
            model(batch, task='mrc', compute_loss=False)
        view_logprobs = F.log_softmax(view_logits, dim=-1)
        obj_logprobs = F.log_softmax(obj_logits, dim=-1)
        loss = F.kl_div(view_logprobs, view_targets, reduction='sum')*2 + \
               F.kl_div(obj_logprobs, obj_targets, reduction='sum')*20
        tot_score += compute_accuracy_for_soft_targets(view_logits, view_targets) + \
                     compute_accuracy_for_soft_targets(obj_logits, obj_targets)
        val_loss += loss.item()
        n_feat += batch['vp_view_mrc_masks'].sum().item() + batch['vp_obj_mrc_masks'].sum().item()

        val_bar.set_postfix({
            'val_loss': val_loss/n_feat,
            'n_feat': n_feat,
            'val_acc': tot_score/n_feat,
        })

    val_loss = sum(all_gather(val_loss))
    tot_score = sum(all_gather(tot_score))
    n_feat = sum(all_gather(n_feat))
    tot_time = time.time() - st
    val_loss /= n_feat
    val_acc = tot_score / n_feat
    val_log = {'loss': val_loss,
               'acc': val_acc,
               'feat_per_s': n_feat / tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score: {val_acc*100:.2f} %")
    return val_log


@torch.no_grad()
def validate_sap(model, val_loader):
    LOGGER.info("start running SAP validation...")
    val_gloss, val_lloss, val_floss = 0, 0, 0
    n_gcorrect, n_lcorrect, n_fcorrect = 0, 0, 0
    n_data = 0
    st = time.time()
    val_bar = tqdm(val_loader, desc="validate SAP ...", position=0, leave=True)
    for i, batch in enumerate(val_bar):
        global_logits, local_logits, fused_logits, global_act_labels, local_act_labels = \
            model(batch, task='sap', compute_loss=False)
        val_gloss += F.cross_entropy(global_logits, global_act_labels, reduction='sum').data.item()
        val_lloss += F.cross_entropy(local_logits, local_act_labels, reduction='sum').data.item()
        val_floss += F.cross_entropy(fused_logits, global_act_labels, reduction='sum').data.item()
        n_gcorrect += torch.sum(torch.argmax(global_logits, 1) == global_act_labels).item()
        n_lcorrect += torch.sum(torch.argmax(local_logits, 1) == local_act_labels).item()
        n_fcorrect += torch.sum(torch.argmax(fused_logits, 1) == global_act_labels).item()
        n_data += len(global_act_labels)

    n_data = sum(all_gather(n_data))
    val_gloss = sum(all_gather(val_gloss)) / n_data
    val_lloss = sum(all_gather(val_lloss)) / n_data
    val_floss = sum(all_gather(val_floss)) / n_data
    gacc = sum(all_gather(n_gcorrect)) / n_data
    lacc = sum(all_gather(n_lcorrect)) / n_data
    facc = sum(all_gather(n_fcorrect)) / n_data

    tot_time = time.time() - st
    val_log = {'gloss': val_gloss, 'lloss': val_lloss, 'floss': val_floss,
               'gacc': gacc, 'lacc': lacc, 'facc': facc,
               'tok_per_s': n_data / tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"gacc: {gacc * 100:.2f}, lacc: {lacc * 100:.2f}, facc: {facc * 100:.2f}")
    return val_log

@torch.no_grad()
def validate_og(model, val_loader):
    LOGGER.info("start running Object Grounding validation...")
    val_loss = 0
    n_correct = 0
    n_data = 0
    st = time.time()
    val_bar = tqdm(val_loader, desc="validate Object Grounding ...", position=0, leave=True)
    for i, batch in enumerate(val_bar):
        scores = model(batch, task='og', compute_loss=False)
        labels = batch['obj_labels']
        loss = F.cross_entropy(scores, labels, reduction='sum')
        val_loss += loss.item()
        n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
        n_data += labels.numel()
    val_loss = sum(all_gather(val_loss))
    n_correct = sum(all_gather(n_correct))
    n_data = sum(all_gather(n_data))
    tot_time = time.time()-st
    val_loss /= n_data
    acc = n_correct / n_data
    val_log = {'loss': val_loss,
               'acc': acc,
               'tok_per_s': n_data/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"acc: {acc*100:.2f} %")
    return val_log


def validate(model, val_dataloaders, setname=''):
    model.eval()
    for task, loader in val_dataloaders.items():
        LOGGER.info(f"validate val{setname} on {task} task")
        if task.startswith('qa'):
            val_log = validate_qa(model, loader)
        elif task.startswith('mrc'):
            val_log = validate_mrc(model, loader)
        elif task.startswith('sap'):
            val_log = validate_sap(model, loader)
        elif task.startswith('og'):
            val_log = validate_og(model, loader)
        elif task.startswith('instr'):
            val_log = validate_instr(model, loader)
        else:
            raise ValueError(f'Undefined task {task}')
        val_log = {f'val{setname}_{task}_{k}': v for k, v in val_log.items()}
        TB_LOGGER.log_scalar_dict(
            {f'valid{setname}_{task}/{k}': v for k, v in val_log.items()}
        )
    model.train()