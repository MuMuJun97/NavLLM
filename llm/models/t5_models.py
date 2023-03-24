import torch
import torch.nn as nn
import warnings
from typing import List, Optional, Tuple, Union
from collections import defaultdict
from transformers import T5Tokenizer, T5ForConditionalGeneration,T5Config
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import CausalLMOutputWithPast
from .vision_language_model import (
    ImageEmbeddings,LocalVPEncoder,GlobalMapEncoder,
    extend_neg_masks,gen_seq_masks
)
from .transformer import OPTAttention,pad_tensors_wgrad
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import copy
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
)
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

class T5ForMultimodelGeneration(T5ForConditionalGeneration):
    # a list of re pattern of tensor names to ignore from the model when loading the model weights
    # (and avoid unnecessary warnings).
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    # a list of re pattern of tensor names to ignore from the weights when loading the model weights
    # (and avoid unnecessary warnings).
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    def __init__(self, config: T5Config):
        """
        Args:
            config: T5 config.
        Notes:
            改写的 multi-modal T5 model.
        """
        super().__init__(config)
        self.model_dim = config.d_model  # 768
        self.padding_idx = config.pad_token_id  # 0 <pad>

        # 0. word tokens embedding: Embedding(32128, 768)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 1. encoder for language
        encoder_config = copy.deepcopy(config)
        # Whether the model is used as decoder or not (in which case it's used as an encoder).
        encoder_config.is_decoder = False
        # Whether or not the model should return the last key/values attentions (not used by all models).
        encoder_config.use_cache = False
        # Whether the model is used as an encoder/decoder or not.
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, embed_tokens=self.shared)

        # 2. decoder for language + vision
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers  # 12
        self.decoder = T5Stack(decoder_config, self.shared)

        # 3. language output layer
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache # True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict # True

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn("\n\n __HEAD_MASK_WARNING_MSG ... \n\n", FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def forward_encoder(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache # True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict # True
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return encoder_outputs

    def forward_decoder(
            self,
            hidden_states,
            encoder_outputs,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache # True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict # True

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids, # from labels: right shift
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states, # from encoder： text-features
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache, # True
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, # True
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        ),decoder_outputs


class GlobalTextPathCMT(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        if 't5' in model_config.Net:
            self.language_model = T5ForMultimodelGeneration.from_pretrained(model_config.Net, device_map="auto")
        else:
            raise NotImplementedError

        ############################## for vln navigation #################################
        self.txt_encoder = nn.Linear(self.language_model.model_dim, model_config.hidden_size)
        self.txt_decoder = nn.Linear(model_config.hidden_size, self.language_model.model_dim)

        # Embedding(2, hidden_size)
        self.token_type_embeddings = nn.Embedding(model_config.type_vocab_size, model_config.hidden_size)

        # 图像encoder
        self.img_embeddings = ImageEmbeddings(model_config)

        self.local_encoder = LocalVPEncoder(model_config)
        self.global_encoder = GlobalMapEncoder(model_config)

        ########## fuse global + local ##########
        self.mha_layer = torch.nn.MultiheadAttention(
            embed_dim=model_config.hidden_size,
            kdim=model_config.hidden_size,
            vdim=model_config.hidden_size,
            num_heads=1,
            # batch_first=True # torch 1.10 API
        )
        self.gate_dense = nn.Linear(
            2*model_config.hidden_size, model_config.hidden_size
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, compute_loss, return_gmap_embeds=True):
        # text embedding
        encoder_outputs = self.language_model.forward_encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        txt_embeds = self.txt_encoder(encoder_outputs[0])
        txt_masks = gen_seq_masks(batch['txt_lens'],attention_mask=batch['attention_mask'])

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


    def forward_text(self, batch, compute_loss):
        # text embedding
        encoder_outputs = self.language_model.forward_encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        txt_embeds = self.txt_encoder(encoder_outputs[0])
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

        # fuse type 1: add
        # txt_embeds = gmap_txt_embeds + vp_txt_embeds

        # fuse type 2: learnable weight
        query, key, value = [x.transpose(1, 0) for x in (vp_txt_embeds, gmap_txt_embeds, gmap_txt_embeds)]
        gmap_att, _ = self.mha_layer(query, key, value)
        gmap_att = gmap_att.transpose(1, 0)
        merge = torch.cat([vp_txt_embeds, gmap_att], dim=-1)
        gate = self.sigmoid(self.gate_dense(merge))
        txt_embeds = (1 - gate) * vp_txt_embeds + gate * gmap_txt_embeds

        txt_embeds = self.txt_decoder(txt_embeds)

        # validation stage: labels not None!
        txt_output,decoder_outputs = self.language_model.forward_decoder(
            hidden_states=txt_embeds,
            encoder_outputs=encoder_outputs,
            labels=batch['labels'],
            attention_mask=batch['attention_mask']
        )

        return txt_output,decoder_outputs


class GlocalTextPathCMTPreTraining(GlobalTextPathCMT):
    def __init__(self, model_config):
        super().__init__(model_config)

        ###########################################################
        self.base_model = GlobalTextPathCMT(model_config)

        if 'mrc' in model_config.pretrain_tasks:
            self.image_classifier = RegionClassification(model_config.hidden_size, model_config.image_prob_size)
            if model_config.obj_prob_size > 0 and model_config.obj_prob_size != model_config.image_prob_size:
                self.obj_classifier = RegionClassification(model_config.hidden_size, model_config.obj_prob_size)
            else:
                self.obj_classifier = None

        if 'sap' in model_config.pretrain_tasks:
            self.global_sap_head = ClsPrediction(model_config.hidden_size)
            self.local_sap_head = ClsPrediction(model_config.hidden_size)
            if model_config.glocal_fuse:
                self.sap_fuse_linear = ClsPrediction(model_config.hidden_size, input_size=model_config.hidden_size*2)
            else:
                self.sap_fuse_linear = None

        if 'og' in model_config.pretrain_tasks:
            self.og_head = ClsPrediction(model_config.hidden_size)

        # TODO init weights
        # TODO tie weights: clone txt word embedding中的weights给qa task的decoder.

    def forward(self, batch, task, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        if task.startswith('qa'):
            return self.forward_qa(batch,compute_loss)
        elif task.startswith('instr'):
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
        txt_output,decoder_outputs = self.base_model.forward_text(batch,compute_loss)

        if compute_loss:
            return txt_output.loss
        else:
            return txt_output.logits

    def forward_mrc(self, batch, compute_loss):
        _, vp_embeds = self.base_model.forward(batch,compute_loss,return_gmap_embeds=False)

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
                # mrc_loss = torch.cat([view_mrc_loss, obj_mrc_loss], 0)
                mrc_loss = torch.cat([view_mrc_loss*2, obj_mrc_loss*20], 0)
            return mrc_loss.mean()
        else:
            return view_prediction_soft_labels, view_mrc_targets, obj_prediction_soft_labels, obj_mrc_targets

    def forward_sap(self, batch, compute_loss):
        batch_size = batch['input_ids'].shape[0]

        gmap_embeds, vp_embeds = self.base_model.forward(batch, compute_loss)

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
        _, vp_embeds = self.base_model.forward(batch, compute_loss,return_gmap_embeds=False)

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


# if __name__ == '__main__':
#     tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
#     model = T5ForMultimodelGeneration.from_pretrained("google/flan-t5-large", device_map="auto")
#     input_text = "Navigation question: What is the relationship between the target room and other neighboring rooms?"
#     input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
#     outputs = model.generate(input_ids, max_new_tokens=20)
#     print(tokenizer.decode(outputs[0]))