import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from collections import defaultdict
from transformers import OPTModel,OPTPreTrainedModel,T5Config
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import CausalLMOutputWithPast
from .vision_language_model import (
    ImageEmbeddings,LocalVPEncoder,GlobalMapEncoder,
    extend_neg_masks,gen_seq_masks
)
from .transformer import OPTAttention,pad_tensors_wgrad
import torch.nn.functional as F
BertLayerNorm = torch.nn.LayerNorm

class RegionClassification(nn.Module):
    " for MRC(-kl)"
    def __init__(self, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, label_dim))

    def forward(self, input_):
        output = self.net(input_)
        return output

class ClsPrediction(nn.Module):
    def __init__(self, hidden_size, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)


class OPTDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 768
        self.ffn_dim = 768
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=4,
            dropout=0.0,
            is_decoder=True,
            bias=True,
        )
        self.do_layer_norm_before = True
        self.dropout = 0.1
        self.activation_fn = nn.ReLU()

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=True
        )
        self.fc1 = nn.Linear(self.embed_dim, self.ffn_dim, bias=True)
        self.fc2 = nn.Linear(self.ffn_dim, self.embed_dim, bias=True)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class GlobalTextPathCMT(OPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.opt_model = OPTModel.from_pretrained(
            pretrained_model_name_or_path="facebook/opt-iml-max-1.3b",
            torch_dtype=torch.float16
        )
        for name, param in self.opt_model.decoder.layers.named_parameters():
            layer_idx = int(name.split('.')[0])
            if layer_idx < 20:
                param.requires_grad = False

        self.opt_output = nn.Linear(self.opt_model.config.word_embed_proj_dim,
                                    768, bias=False)

        # Embedding(2, 768)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 图像encoder
        self.img_embeddings = ImageEmbeddings(config)

        self.local_encoder = LocalVPEncoder(config)
        self.global_encoder = GlobalMapEncoder(config)

        self.gate_dense = nn.Linear(
            2*config.hidden_size, config.hidden_size
        )
        self.sigmoid = nn.Sigmoid()



    def forward(self, batch, return_gmap_embeds=True):
        output = self.forward_decoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
        )
        txt_embeds = output.logits
        txt_masks = gen_seq_masks(batch['txt_lens'], attention_mask=batch['attention_mask'])

        # trajectory embedding: 每个node的 image_view_feats + detected_obj_feats
        split_traj_embeds, split_traj_vp_lens = self.img_embeddings(
            batch['traj_view_img_fts'],
            batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'],
            batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'],
            self.token_type_embeddings
        )

        # gmap embeds
        if return_gmap_embeds:
            gmap_embeds = self.global_encoder(
                txt_embeds, txt_masks,
                split_traj_embeds, split_traj_vp_lens, batch['traj_vpids'], batch['traj_cand_vpids'], batch['gmap_vpids'],
                batch['gmap_step_ids'], batch['gmap_pos_fts'], batch['gmap_lens'],
                graph_sprels=batch['gmap_pair_dists'],
            )
        else:
            gmap_embeds = None

        # vp embeds
        vp_embeds = self.local_encoder(
            txt_embeds, txt_masks,
            split_traj_embeds, split_traj_vp_lens, batch['vp_pos_fts']
        )

        return gmap_embeds, vp_embeds

    def forward_decoder(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.opt_model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # float16-->float32
        logits = self.opt_output(outputs[0].float()).contiguous()

        loss = None

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward_qa(self, batch):
        output = self.forward_decoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
        )
        txt_embeds = output.logits
        txt_masks = gen_seq_masks(batch['txt_lens'],attention_mask=batch['attention_mask'])
        extended_txt_masks = extend_neg_masks(txt_masks)

        # trajectory embedding: 每个node的 image_view_feats + detected_obj_feats
        split_traj_embeds, split_traj_vp_lens = self.img_embeddings(
            batch['traj_view_img_fts'],
            batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'],
            batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'],
            self.token_type_embeddings
        )

        # gmap embeds
        gmap_input_embeds, gmap_masks = self.global_encoder.gmap_input_embedding(
            split_traj_embeds, split_traj_vp_lens, batch['traj_vpids'], batch['traj_cand_vpids'], batch['gmap_vpids'],
            batch['gmap_step_ids'], batch['gmap_pos_fts'], batch['gmap_lens']
        )
        gmap_txt_embeds = txt_embeds
        extended_gmap_masks = extend_neg_masks(gmap_masks)
        for layer_module in self.global_encoder.encoder.x_layers:
            gmap_txt_embeds = layer_module.forward_lang2visn(
                gmap_txt_embeds, extended_txt_masks,
                gmap_input_embeds, extended_gmap_masks,
            )

        # vp embeds
        vp_input_embeds, vp_masks = self.local_encoder.vp_input_embedding(
            split_traj_embeds, split_traj_vp_lens, batch['vp_pos_fts']
        )
        vp_txt_embeds = txt_embeds
        extended_vp_masks = extend_neg_masks(vp_masks)
        for layer_module in self.local_encoder.encoder.x_layers:
            vp_txt_embeds = layer_module.forward_lang2visn(
                vp_txt_embeds, extended_txt_masks,
                vp_input_embeds, extended_vp_masks,
            )

        # type:1
        # txt_embeds = gmap_txt_embeds + vp_txt_embeds

        # type:2
        merge = torch.cat([gmap_txt_embeds, vp_txt_embeds], dim=-1)
        gate = self.sigmoid(self.gate_dense(merge))
        txt_embeds = (1-gate) * gmap_txt_embeds + gate * vp_txt_embeds

        return txt_embeds


class GlocalTextPathCMTPreTraining(OPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.opt = GlobalTextPathCMT(config)

        self.output = 'OPTCLM' # ['OPT','T5','OPTCLM']

        if 'qa' in config.pretrain_tasks:
            if self.output == 'OPT':
                # OPTForSequenceClassification
                self.num_labels = 3
                self.decoderlayer = OPTDecoderLayer()
                # ['A','B','C']
                self.score = nn.Linear(768, self.num_labels, bias=False)
                self.pad_token_id = 1 # opt <pad> token_id=1
            elif self.output == 'T5':
                # self.up_layer = nn.Linear(in_features=768,out_features=2048,bias=False)
                # mm-cot: T5Stack decoder
                output_config_file = config.output_config
                output_config = T5Config.from_pretrained(output_config_file)
                embed_tokens = nn.Embedding(50272, 768, 1)
                self.decoderlayer = T5Stack(output_config, embed_tokens=embed_tokens)
            elif self.output == 'OPTCLM':
                self.vocab_size = 50272
                self.decoderlayer = OPTDecoderLayer()
                # the lm_head weight is automatically tied to the embed tokens weight
                self.lm_head = nn.Linear(768, self.vocab_size, bias=False)
            else:
                raise NotImplementedError

        if 'mrc' in config.pretrain_tasks:
            self.image_classifier = RegionClassification(self.config.hidden_size, self.config.image_prob_size)
            if self.config.obj_prob_size > 0 and self.config.obj_prob_size != self.config.image_prob_size:
                self.obj_classifier = RegionClassification(self.config.hidden_size, self.config.obj_prob_size)
            else:
                self.obj_classifier = None

        if 'sap' in config.pretrain_tasks:
            self.global_sap_head = ClsPrediction(self.config.hidden_size)
            self.local_sap_head = ClsPrediction(self.config.hidden_size)
            if config.glocal_fuse:
                self.sap_fuse_linear = ClsPrediction(self.config.hidden_size, input_size=self.config.hidden_size*2)
            else:
                self.sap_fuse_linear = None

        if 'og' in config.pretrain_tasks:
            self.og_head = ClsPrediction(self.config.hidden_size)

        # TODO init weights
        # TODO tie weights: clone txt word embedding中的weights给qa task的decoder.

    def forward(self, batch, task, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        if task.startswith('qa'):
            return self.forward_qa(batch,compute_loss)
        elif task.startswith('mrc'):
            return self.forward_mrc(batch,compute_loss)
        elif task.startswith('sap'):
            return self.forward_sap(batch,compute_loss)
        elif task.startswith('og'):
            return self.forward_og(batch,compute_loss)
        else:
            raise ValueError('invalid task')

    def _compute_masked_hidden(self, hidden, mask):
        '''get only the masked region (don't compute unnecessary hiddens)'''
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def forward_qa(self,batch,compute_loss):
        txt_embeds = self.opt.forward_qa(batch)

        if self.output == 'OPT':
            answer = batch['ans_labels']
            txt_embeds = self.decoderlayer(txt_embeds)[0]
            logits = self.score(txt_embeds)

            batch_size, sequence_length = batch['input_ids'].shape[:2]
            sequence_lengths = (torch.ne(batch['input_ids'], self.pad_token_id).sum(-1) - 1).to(logits.device)
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(pooled_logits.view(-1, self.num_labels), answer.view(-1))
        elif self.output == 'T5':
            labels = batch['labels']

            # _shift_right:
            decoder_start_token_id = 1
            pad_token_id = 1
            shifted_input_ids = labels.new_zeros(labels.shape)
            shifted_input_ids[..., 1:] = labels[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id
            shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
            decoder_input_ids = shifted_input_ids

            # Decode
            # hidden_states = self.up_layer(txt_embeds) # [B, 192, 768]
            decoder_outputs = self.decoderlayer(
                input_ids=decoder_input_ids,
                attention_mask=None,
                inputs_embeds=None,
                past_key_values=None,
                encoder_hidden_states=txt_embeds,  # [B, 192, 2048]
                encoder_attention_mask=batch['attention_mask'],
                head_mask=None,
                cross_attn_head_mask=None,
                use_cache=True,  # True
                output_attentions=None,
                output_hidden_states=None,
                return_dict=True,  # True
            )
        elif self.output == 'OPTCLM':
            outputs = self.decoderlayer(txt_embeds)
            logits = self.lm_head(outputs[0]).contiguous()
            loss = None
            labels = batch['labels']
            if labels is not None and compute_loss:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

                if batch.get('mask_label',None) is not None:
                    # compute mask loss
                    mask_label = batch['mask_label']
                    # Shift so that tokens < n predict n
                    shift_mask_logits = logits[..., :-1, :].contiguous()
                    shift_mask_labels = mask_label[..., 1:].contiguous()
                    # Flatten the tokens
                    loss_fct_mask = nn.CrossEntropyLoss()
                    loss_mask = loss_fct_mask(shift_mask_logits.view(-1, self.vocab_size), shift_mask_labels.view(-1))

                    loss = loss + loss_mask
        else:
            raise NotImplementedError

        if compute_loss:
            return loss
        else:
            return logits

    def forward_mrc(self, batch, compute_loss):
        _, vp_embeds = self.opt.forward(batch,return_gmap_embeds=False)

        vp_view_lens = [x[-1] for x in torch.split(batch['traj_vp_view_lens'], batch['traj_step_lens'])]
        vp_view_embeds = pad_tensors_wgrad(
            [x[1:view_len + 1] for x, view_len in zip(vp_embeds, vp_view_lens)]
        )  # [stop] at 0

        # only compute masked regions for better efficient=cy
        view_masked_output = self._compute_masked_hidden(vp_view_embeds, batch['vp_view_mrc_masks'])
        view_prediction_soft_labels = self.image_classifier(view_masked_output)
        view_mrc_targets = self._compute_masked_hidden(batch['vp_view_probs'], batch['vp_view_mrc_masks'])

        if batch['traj_obj_img_fts'] is not None:
            vp_obj_lens = [x[-1] for x in torch.split(batch['traj_vp_obj_lens'], batch['traj_step_lens'])]
            vp_obj_embeds = pad_tensors_wgrad(
                [x[view_len+1:view_len+obj_len+1] for x, view_len, obj_len in zip(vp_embeds, vp_view_lens, vp_obj_lens)]
            )
            # vp_obj_mrc_masks = vp_obj_mrc_masks[:, :vp_obj_embeds.size(1)]
            obj_masked_output = self._compute_masked_hidden(vp_obj_embeds, batch['vp_obj_mrc_masks'])
            if self.obj_classifier is None:
                obj_prediction_soft_labels = self.image_classifier(obj_masked_output)
            else:
                obj_prediction_soft_labels = self.obj_classifier(obj_masked_output)
            obj_mrc_targets = self._compute_masked_hidden(batch['vp_obj_probs'], batch['vp_obj_mrc_masks'])
        else:
            obj_prediction_soft_labels, obj_mrc_targets = None, None

        if compute_loss:
            view_prediction_soft_labels = F.log_softmax(view_prediction_soft_labels, dim=-1)
            view_mrc_loss = F.kl_div(view_prediction_soft_labels, view_mrc_targets, reduction='none').sum(dim=1)
            if obj_prediction_soft_labels is None:
                mrc_loss = view_mrc_loss
            else:
                obj_prediction_soft_labels = F.log_softmax(obj_prediction_soft_labels, dim=-1)
                obj_mrc_loss = F.kl_div(obj_prediction_soft_labels, obj_mrc_targets, reduction='none').sum(dim=1)
                mrc_loss = torch.cat([view_mrc_loss, obj_mrc_loss], 0)
            return mrc_loss.mean()
        else:
            return view_prediction_soft_labels, view_mrc_targets, obj_prediction_soft_labels, obj_mrc_targets

    def forward_sap(self, batch, compute_loss):
        batch_size = batch['input_ids'].shape[0]
        gmap_embeds, vp_embeds = self.opt.forward(batch)
        if self.sap_fuse_linear is None:
            fuse_weights = 0.5
        else:
            fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))

        global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
        global_logits.masked_fill_(batch['gmap_visited_masks'], -float('inf'))
        global_logits.masked_fill_(gen_seq_masks(batch['gmap_lens']).logical_not(), -float('inf'))

        local_logits = self.local_sap_head(vp_embeds).squeeze(2) * (1 - fuse_weights)
        vp_nav_masks = pad_tensors_wgrad(
            [x[-1] != 1 for x in torch.split(batch['traj_nav_types'], batch['traj_step_lens'])]
        )[:, :local_logits.size(1) - 1]
        vp_nav_masks = torch.cat(
            [torch.zeros(len(vp_nav_masks), 1).bool().to(vp_nav_masks.device), vp_nav_masks], 1
        )  # add [stop]
        local_logits.masked_fill_(vp_nav_masks, -float('inf'))

        # fusion
        fused_logits = torch.clone(global_logits)
        fused_logits[:, 0] += local_logits[:, 0]  # stop
        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(batch['gmap_vpids'][i], batch['gmap_visited_masks'][i]) if mask])
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate(batch['traj_cand_vpids'][i][-1]):
                if cand_vpid in visited_nodes:
                    bw_logits += local_logits[i, j + 1]
                else:
                    tmp[cand_vpid] = local_logits[i, j + 1]
            for j, vp in enumerate(batch['gmap_vpids'][i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fused_logits[i, j] += tmp[vp]
                    else:
                        fused_logits[i, j] += bw_logits

        global_act_labels = batch['global_act_labels']
        local_act_labels = batch['local_act_labels']
        if compute_loss:
            global_losses = F.cross_entropy(global_logits, global_act_labels, reduction='none')
            local_losses = F.cross_entropy(local_logits, local_act_labels, reduction='none')
            fused_losses = F.cross_entropy(fused_logits, global_act_labels, reduction='none')
            losses = global_losses + local_losses + fused_losses
            return losses.mean()
        else:
            return global_logits, local_logits, fused_logits, global_act_labels, local_act_labels

    def forward_og(self, batch, compute_loss):
        _, vp_embeds = self.opt.forward(batch,return_gmap_embeds=False)

        vp_view_lens = [x[-1] for x in torch.split(batch['traj_vp_view_lens'], batch['traj_step_lens'], 0)]
        vp_obj_lens = [x[-1] for x in torch.split(batch['traj_vp_obj_lens'], batch['traj_step_lens'], 0)]
        obj_embeds = pad_tensors_wgrad([
            x[1+view_len: 1+view_len+obj_len] for x, view_len, obj_len in zip(vp_embeds, vp_view_lens, vp_obj_lens)
        ])
        obj_masks = gen_seq_masks(torch.stack(vp_obj_lens, 0))

        obj_logits = self.og_head(obj_embeds).squeeze(2)
        obj_logits.masked_fill_(obj_masks.logical_not(), -float('inf'))

        if compute_loss:
            losses = F.cross_entropy(obj_logits, batch['obj_labels'], reduction='none')
            return losses.mean()
        else:
            return obj_logits