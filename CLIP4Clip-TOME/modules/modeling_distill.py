from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import copy

import torch
from torch import nn

from modules.until_module import PreTrainedModel, AllGather, CrossEn
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from modules.module_clip import CLIP, convert_weights, convert_weights_to_fp32
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

logger = logging.getLogger(__name__)
allgather = AllGather.apply

class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        pretrained_clip_name = "ViT-B/32"
        if hasattr(task_config, 'pretrained_clip_name'):
            pretrained_clip_name = task_config.pretrained_clip_name
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)

        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

        ## ===> Initialization trick [HARD CODE]
        if model.linear_patch == "3d":
            contain_conv2 = False
            for key in state_dict.keys():
                if key.find("visual.conv2.weight") > -1:
                    contain_conv2 = True
                    break
            if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                cp_weight = state_dict["clip.visual.conv1.weight"].clone()
                kernel_size = model.clip.visual.conv2.weight.size(2)
                conv2_size = model.clip.visual.conv2.weight.size()
                conv2_size = list(conv2_size)

                left_conv2_size = conv2_size.copy()
                right_conv2_size = conv2_size.copy()
                left_conv2_size[2] = (kernel_size - 1) // 2
                right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

                left_zeros, right_zeros = None, None
                if left_conv2_size[2] > 0:
                    left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
                if right_conv2_size[2] > 0:
                    right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                cat_list = []
                if left_zeros != None: cat_list.append(left_zeros)
                cat_list.append(cp_weight.unsqueeze(2))
                if right_zeros != None: cat_list.append(right_zeros)
                cp_weight = torch.cat(cat_list, dim=2)

                state_dict["clip.visual.conv2.weight"] = cp_weight

        if model.sim_header == 'tightTransf':
            contain_cross = False
            for key in state_dict.keys():
                if key.find("cross.transformer") > -1:
                    contain_cross = True
                    break
            if contain_cross is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                        continue
                    if key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict["cross."+key] = val.clone()
                            continue

        if model.sim_header == "seqLSTM" or model.sim_header == "seqTransf":
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
        ## <=== End of initialization trick

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)
        
        if task_config.frame_selection:
            assert hasattr(model.clip.visual, "g_cls_embedding")
            model.clip.visual.copy_cls_to_gcls()
            model.gf_temp = nn.Parameter(torch.tensor(0.01))
        
        if task_config.atp:
            model.atp = copy.deepcopy(model.clip.visual.transformer.resblocks[1 + task_config.atp_input_layer: 1 + task_config.atp_num_layers + task_config.atp_input_layer])
            convert_weights_to_fp32(model.atp)
            model.atp_input_layer = task_config.atp_input_layer
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            model.atp_linear = nn.Linear(vision_width, 1)
            model.gf_temp = nn.Parameter(torch.tensor(0.01))

        return model

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

class CLIP4Clip(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(CLIP4Clip, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            linear_patch=self.linear_patch, frame_selection=task_config.frame_selection,
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        self.sim_header = 'meanP'
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        if self.sim_header == "tightTransf": assert self.loose_type is False

        cross_config.max_position_embeddings = context_length
        if self.loose_type is False:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if os.getenv("enable_frame_pos_embed", False):
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, vision_width).half()
        if self.sim_header == "seqLSTM" or self.sim_header == "seqTransf":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
        if self.sim_header == "seqTransf":
            self.transformerClip = TransformerClip(width=transformer_width, layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )
        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                       batch_first=True, bidirectional=False, num_layers=1)

        self.loss_fct = CrossEn()
        # self.distill_scale = nn.Parameter(torch.ones([]) * self.task_config.distill_scale)
        self.register_buffer("distill_scale", torch.ones([]) * self.task_config.distill_scale)

        self.apply(self.init_weights)

    def forward(
        self, 
        input_ids, 
        token_type_ids, 
        attention_mask, 
        video, 
        video_mask=None, 
        frame_select=False,
        return_hidden_states=False,
        teacher_hidden_states=None,
    ):
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts
        
        sequence_output, visual_output, gf_sims, visual_hidden_states = self.get_sequence_visual_output(
            input_ids,
            token_type_ids, 
            attention_mask,
            video, 
            video_mask, 
            shaped=True, 
            video_frame=video_frame, 
            frame_selection=frame_select,
            return_gf_sims=frame_select,
            return_hidden_states=return_hidden_states,
            )

        if self.training:
            loss = 0.
            sim_matrix, *_tmp = self.get_similarity_logits(sequence_output, visual_output, attention_mask, video_mask,
                                                    shaped=True, loose_type=self.loose_type)
            sim_loss1 = self.loss_fct(sim_matrix)
            sim_loss2 = self.loss_fct(sim_matrix.T)
            sim_loss = (sim_loss1 + sim_loss2) / 2
            loss += sim_loss
            
            if teacher_hidden_states is not None:
                
                
                # loss_distill = 0.
                # for layer_idx in range(len(teacher_hidden_states)):
                #     t_hs = teacher_hidden_states[layer_idx] # teacher hidden states L,BT,C
                #     s_hs = visual_hidden_states[layer_idx] # student hidden states L,BT,C
                #     t_frame_cls = t_hs[0].view(b, video_frame, -1) # B,T,C
                #     # print(t_frame_cls.shape)
                #     num_merged_frame = b * video_frame // s_hs.shape[1]
                #     s_frame_cls = s_hs[:num_merged_frame].transpose(0, 1).contiguous().view(b, video_frame, -1) # B,T,C
                #     loss_distill += self.distill_computation(s_frame_cls, t_frame_cls, video_mask, type=self.task_config.distill_type) * self.task_config.distill_weight
                # #     print(loss_distill)
                # # import pdb; pdb.set_trace()
                # loss += loss_distill 

                t_hs = teacher_hidden_states[12] # teacher hidden states L,BT,C
                s_hs = visual_hidden_states[12] # student hidden states L,BT,C
                t_frame_cls = t_hs[0].view(b, video_frame, -1) # B,T,C
                num_merged_frame = b * video_frame // s_hs.shape[1]
                s_frame_cls = s_hs[:num_merged_frame].transpose(0, 1).contiguous().view(b, video_frame, -1) # B,T,C
                loss_distill = self.distill_computation(s_frame_cls, t_frame_cls, video_mask, type=self.task_config.distill_type) * self.task_config.distill_weight
                loss += loss_distill
            return loss, loss_distill
        else:
            return None

    def distill_computation(self, preds, tgts, video_mask, type="contrast"):
        if type == "contrast":
            return self.contrastive_distill(preds, tgts, video_mask, norm=True)
        elif type == "mse":
            return self.mse_distill(preds, tgts, video_mask)
        elif type == "cosine":
            return self.mse_distill(preds, tgts, video_mask, norm=True)
        else:
            raise NotImplemented

    def mse_distill(self, preds, tgts, video_mask, norm=False):
        # preds: B,T,C
        # tgts: B,T,C
        # video_mask: B,T
        if norm:
            preds = preds / preds.norm(dim=-1, keepdim=True)
            tgts = tgts / tgts.norm(dim=-1, keepdim=True)
        loss_mse = torch.nn.MSELoss(reduction="none")(preds, tgts) # B,T,C
        loss_mse = loss_mse.sum(-1) * video_mask
        loss_mse = loss_mse.sum() / video_mask.sum()
        return loss_mse

    def contrastive_distill(self, preds, tgts, video_mask, norm=True):
        if norm:
            preds = preds / preds.norm(dim=-1, keepdim=True)
            tgts = tgts / tgts.norm(dim=-1, keepdim=True)
        sim_preds = torch.matmul(preds, preds.transpose(1, 2)) # B,T,T
        sim_tgts = torch.matmul(tgts, tgts.transpose(1, 2)) # B,T,T
        sim_preds = sim_preds + (1 - video_mask.unsqueeze(1)) * -1e4
        sim_tgts = sim_tgts + (1 - video_mask.unsqueeze(1)) * -1e4
        loss_distill = nn.functional.kl_div(torch.log_softmax(sim_preds / self.distill_scale, dim=-1), torch.softmax(sim_tgts / self.distill_scale, dim=-1), reduction="none")
        # loss_distill = - (torch.softmax(sims_tgts / self.distill_scale, dim=-1) * torch.log_softmax(sim_preds / self.distill_scale, dim=-1))
        loss_distill = loss_distill.sum(-1)[video_mask==1].mean()
        return loss_distill
               

    @torch.no_grad()
    def construct_gf_sim_target(self, sequence_output, visual_output, video_mask):
        # sequence_output: B,1,C
        # visual_output: B,T,C
        # video_mask: B,T
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        sim_tgt = torch.matmul(sequence_output, visual_output.transpose(1, 2)) # B,1,T
        sim_tgt = sim_tgt + (1 - video_mask.unsqueeze(1)) * -1e4
        sim_tgt = torch.softmax(sim_tgt / self.gf_temp, dim=-1)
        return sim_tgt

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_hidden = self.clip.encode_text(input_ids).float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))

        return sequence_hidden

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1, frame_selection=False, return_gf_sims=False, return_hidden_states=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        if os.getenv("enable_frame_pos_embed", False):
            # 当使用token merge时，可以在patch embedding上加上frame pos embedding
            position_ids = torch.arange(video_frame, dtype=torch.long, device=video.device)
            position_ids = position_ids.unsqueeze(0).expand(bs_pair, -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids) # B,T,C
        else:
            frame_position_embeddings = None

        visual_hidden, hidden_states = self.clip.encode_image(video, video_frame=video_frame, return_hidden=True,
                        video_mask=video_mask, frame_selection=frame_selection, return_gf_sims=return_gf_sims, 
                        return_hidden_states=return_hidden_states, frame_position_embeddings=frame_position_embeddings)

        if frame_selection:
            visual_hidden, gf_sims = visual_hidden[0].float(), visual_hidden[1]
            gf_sims = gf_sims.float() if gf_sims is not None else None
        else:
            visual_hidden, gf_sims = visual_hidden.float(), None
        if hasattr(self.clip.visual, "_tome_info") and self.clip.visual._tome_info["num_cls"] > 1:
            visual_hidden = visual_hidden[:, :self.clip.visual._tome_info["num_cls"]].contiguous()
            visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.shape[-1])[:, :video_frame]
        else:
            visual_hidden = visual_hidden[:, 0]
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))

        return visual_hidden, gf_sims, hidden_states

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, 
        shaped=False, video_frame=-1, frame_selection=False, return_gf_sims=False, return_hidden_states=False):
        # fs: frame selection setting, when enabled, return the gf_sims
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        sequence_output = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)
        visual_output, gf_sims, hidden_states = self.get_visual_output(video, video_mask, 
        shaped=True, video_frame=video_frame, frame_selection=frame_selection, return_gf_sims=return_gf_sims, return_hidden_states=return_hidden_states)
        return sequence_output, visual_output, gf_sims, hidden_states

    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):

        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, pooled_output, concat_mask

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        B, T, C = visual_output.shape
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        valid_num_frame = visual_output.shape[1]
        if valid_num_frame != video_mask_un.shape[1]:
            temporal_scale = video_mask_un.shape[1] // valid_num_frame
            video_mask_un = video_mask_un.view(visual_output.shape[0], valid_num_frame, temporal_scale, -1)[:, :, 0]
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        return text_out, video_out

    def _loose_similarity(self, sequence_output, visual_output, attention_mask, video_mask, sim_header="meanP"):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        if sim_header == "meanP":
            # Default: Parameter-free type
            pass
        elif sim_header == "seqLSTM":
            # Sequential type: LSTM
            visual_output_original = visual_output
            visual_output = pack_padded_sequence(visual_output, torch.sum(video_mask, dim=-1).cpu(),
                                                 batch_first=True, enforce_sorted=False)
            visual_output, _ = self.lstm_visual(visual_output)
            if self.training: self.lstm_visual.flatten_parameters()
            visual_output, _ = pad_packed_sequence(visual_output, batch_first=True)
            visual_output = torch.cat((visual_output, visual_output_original[:, visual_output.size(1):, ...].contiguous()), dim=1)
            visual_output = visual_output + visual_output_original
        elif sim_header == "seqTransf":
            # Sequential type: Transformer Encoder
            visual_output_original = visual_output
            seq_length = visual_output.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
            position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            visual_output = visual_output + frame_position_embeddings

            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
            visual_output = self.transformerClip(visual_output, extended_video_mask)
            visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
            visual_output = visual_output + visual_output_original

        if self.training:
            visual_output = allgather(visual_output, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config)
            torch.distributed.barrier()

        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        retrieve_logits = self._get_similarity_mean_pooling(sequence_output, visual_output, video_mask)
        # retrieve_logits = self._get_similarity_topk_score_pooling(sequence_output, visual_output, video_mask, topk=12)
        # retrieve_logits = self._get_similarity_topk_embed_pooling(sequence_output, visual_output, video_mask, topk=12)

        return retrieve_logits
    
    def _get_similarity_mean_pooling(self, sequence_output, visual_output, video_mask):
        visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        logit_scale = self.clip.logit_scale.exp()
        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())
        return retrieve_logits

    def _get_similarity_topk_score_pooling(self, sequence_output, visual_output, video_mask, topk=2):
        # sequence_output: B,C
        # visual_output: B,L,C
        Bt, _ = sequence_output.shape
        Bv, L, C = visual_output.shape
        text_frame_sim = torch.matmul(sequence_output, visual_output.view(-1, C).t())  # B,C x BL,C ==> B,BL
        text_frame_sim = text_frame_sim * video_mask.view(1, -1)
        text_frame_sim = text_frame_sim.view(Bt, Bv, L).topk(topk, dim=-1)[0] # B,B,topk
        nonzero_mask = text_frame_sim != 0
        text_frame_sim = text_frame_sim.sum(-1) / nonzero_mask.sum(-1)
        
        logit_scale = self.clip.logit_scale.exp()
        retrieve_logits = logit_scale * text_frame_sim
        return retrieve_logits
    
    def _get_similarity_topk_embed_pooling(self, sequence_output, visual_output, video_mask, topk=2):
        # sequence_output: B,C
        # visual_output: B,L,C
        Bt, _ = sequence_output.shape
        Bv, L, C = visual_output.shape
        text_frame_sim = torch.matmul(sequence_output, visual_output.view(-1, C).t()) # Bt,C x BvL,C ==> Bt,BvL
        sim_topk_idx = text_frame_sim.view(Bt, Bv, L).topk(topk, dim=-1)[1].view(-1, topk, 1) # Bt,Bv,topk ==> BtxBv,topk,1
        expanded_visual_output = visual_output.unsqueeze(0).repeat(Bt, 1, 1, 1).view(Bt * Bv, L, C) # Bt,Bv,L,C => BtxBv,L,C
        expanded_video_mask = video_mask.unsqueeze(0).repeat(Bt, 1, 1).view(Bt * Bv, L) # BtxBv,L
        tgt_visual_output = expanded_visual_output.gather(dim=1, index=sim_topk_idx.repeat(1, 1, C)) # BtxBv,topk,C
        tgt_video_mask = expanded_video_mask.gather(dim=1, index=sim_topk_idx.squeeze(-1)) # BtxBv,topk
        mean_visual_output = self._mean_pooling_for_similarity_visual(tgt_visual_output, tgt_video_mask).view(Bt, Bv, C)  # BtxBv,C ==> Bt,Bv,C
        mean_visual_output = mean_visual_output / mean_visual_output.norm(dim=-1, keepdim=True)
        mean_sim = torch.matmul(sequence_output.unsqueeze(1), mean_visual_output.transpose(-1, -2)).squeeze(1) # Bt,Bv
        
        logit_scale = self.clip.logit_scale.exp()
        retrieve_logits = logit_scale * mean_sim
        return retrieve_logits

    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []

        step_size = b_text      # set smaller to reduce memory cost
        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        # due to clip text branch retrun the last hidden
        attention_mask = torch.ones(sequence_output.size(0), 1)\
            .to(device=attention_mask.device, dtype=attention_mask.dtype)

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_visual, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_visual, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)

            cross_output, pooled_output, concat_mask = \
                self._get_cross_output(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_visual)

            retrieve_logits_list.append(retrieve_logits_row)

        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits

    def get_similarity_logits(self, sequence_output, visual_output, attention_mask, video_mask, shaped=False, loose_type=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
        
        contrastive_direction = ()
        if loose_type:
            assert self.sim_header in ["meanP", "seqLSTM", "seqTransf"]
            retrieve_logits = self._loose_similarity(sequence_output, visual_output, attention_mask, video_mask, sim_header=self.sim_header)
        else:
            assert self.sim_header in ["tightTransf"]
            retrieve_logits = self._cross_similarity(sequence_output, visual_output, attention_mask, video_mask, )

        return retrieve_logits, contrastive_direction
