import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Tuple, Optional, Union, List

from modules.module_clip import ResidualAttentionBlock, VisualTransformer
from tome.merge import bipartite_matching_mapping, bipartite_soft_matching_st_clip, merge_source, merge_wavg, merge_wavg_mask
from tome.utils import parse_r

try:
    import xformers.ops as xops
except ImportError:
    xops = None
    print("Please 'pip install xformers'")

def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.transformer.resblocks), self.r)
            self._tome_info["flatten_layer_merge_type"] = self.flatten_layer_merge_type[:]
            self._tome_info["size"] = None
            self._tome_info["source"] = None
            self._tome_info["num_cls"] = 1
            self._tome_info["layer_idx"] = 0

            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer

def apply_patch(
    model: VisualTransformer, 
    trace_source: bool = False, 
    prop_attn: bool = True, 
    r: Union[List[int], int] = 0,
    type: str = "soft_s",
    min_token: int = 0,
    hierarchy: bool = False,
    token_importance = False,
    importance_alpha: float = 6.0,
    num_cls: int = 1,
    frame_flatten_type: str = "as_patch",
    average_type: str = "AVG",
    frame_flatten_layer: Union[List[int], int] = [4, 8],
    enable_frame_mask: bool = False,
    attn_frame_mask_layer: Union[List[int], int] = [],
    merge_frame_mask_layer: Union[List[int], int] = [],
    binary_frame_mask: bool = False,
    flatten_layer_merge_type: List[str] = None,
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.

    type: "soft_s" for spatial only ToMe; "soft_st" for spatial-temporal ToMe
    num_cls: 每个样本中CLS的数量，用于进行keep cls的frame merge时，同时保留两帧的CLS
    average_type: 合并token的方式，AVG就是直接src和dst取平均；EMA则要考虑dst历史上合并的token数量N，做滑动平均, i.e., res = (dst * N + src) / (N+1)
    """
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    model.r = r
    model.flatten_layer_merge_type = flatten_layer_merge_type
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.class_embedding is not None,
        "distill_token": False,
        "type": type,
        "min_token": min_token,
        "hierarchy": hierarchy,
        "token_importance": token_importance,
        "importance_alpha": importance_alpha,
        "num_cls": num_cls,
        "frame_flatten_type": frame_flatten_type,
        "average_type": average_type,
        "frame_flatten_layer": frame_flatten_layer,
        "layer_idx": 0,
        "enable_frame_mask": enable_frame_mask,
        "attn_frame_mask_layer": attn_frame_mask_layer,
        "merge_frame_mask_layer": merge_frame_mask_layer,
        "binary_frame_mask": binary_frame_mask,
        "flatten_layer_merge_type": flatten_layer_merge_type[:],
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, ResidualAttentionBlock):
            module.__class__ = ToMeResidualAttentionBlock
            module._tome_info = model._tome_info
        elif isinstance(module, nn.MultiheadAttention):
            module.__class__ = ToMeMultiheadAttention

class ToMeResidualAttentionBlock(ResidualAttentionBlock):
    def attention(self, x: torch.Tensor, size: torch.Tensor, frame_idx: torch.Tensor, binary_mask=False):
        """
        size: L,N,1
        frame_idx: B,L,num_frame
        """
        # print(self._tome_info)
        # if self._tome_info['size'] is not None:
        #     print(self._tome_info['size'].shape)
        # if self._tome_info['source'] is not None:
        #     print(self._tome_info['source'].shape) 
        #     print(self._tome_info['source'])
        # print(x.shape)
        # if size is not None:
        #     print(size.shape)
        # print(frame_idx.shape)
        # print(frame_idx)
        # print('-----')
        
        attn_mask_ = self.attn_mask
        

        # if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
        #     attn_mask_ = self.attn_mask(x.size(0))   # LND

        # attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        
        # if frame_idx is not None:
        #     bsz, num_patch, num_frame = frame_idx.shape # B,L,num_frame
        #     if not binary_mask:
        #         frame_mask = (frame_idx.unsqueeze(1) == frame_idx.unsqueeze(2)).any(dim=-1) # B,L,L,num_frame ==> B,L,L
        #     else:
        #         frame_idx = frame_idx.squeeze(-1) # B,L
        #         frame_mask_0 = frame_idx < 1.
        #         frame_mask_0 = torch.logical_and(frame_mask_0.unsqueeze(1), frame_mask_0.unsqueeze(2)) # B,L,L
        #         frame_mask_1 = frame_idx > 0.
        #         frame_mask_1 = torch.logical_and(frame_mask_1.unsqueeze(1), frame_mask_1.unsqueeze(2)) # B,L,L
        #         frame_mask = torch.logical_or(frame_mask_0, frame_mask_1)
        #     new_attn_mask = torch.zeros_like(frame_mask, dtype=x.dtype)
        #     new_attn_mask.masked_fill_(~frame_mask, float("-inf"))
        #     attn_mask = new_attn_mask.repeat(self.attn.num_heads, 1, 1) # B*num_heads,L,L
        #     attn_mask_ = attn_mask_ + attn_mask if attn_mask_ is not None else attn_mask
        
        # print(attn_mask_.shape)
        # print(attn_mask_)
        # print('-----')
        if self._tome_info['source'] is not None:
            attn_mask_ = self.update_attn_mask(x)
            
        if size is not None:
            size = size.log().transpose(0, 1).unsqueeze(1).unsqueeze(1)[...,0] # N,1,1,L
            size = size.repeat(1, self.attn.num_heads, x.size(0), 1).view(-1, x.size(0), x.size(0)) # N*num_heads, L, L
            attn_mask_ = attn_mask_ + size if attn_mask_ is not None else size
            # print(attn_mask_.dtype)
            
        if "token_importance" in self._tome_info.keys():
            need_weights = self._tome_info["token_importance"]
        else:
            need_weights = False
        attn_output, metric, attn_weights = self.attn(x, x, x, need_weights=need_weights, attn_mask=attn_mask_)

        return attn_output, metric, attn_weights # LBC; B,L,head_dim; B*num_heads, L, C

    def frame_flatten_operation(self, x, video_frame, num_cls=1, type="as_patch", permute=True, odd=False):
        # when input shape = L,B,C, permute to B,L,C first
        if permute:
            x = x.permute(1, 0, 2).contiguous()
        bsz, num_token, hidden_dim = x.shape # L,B,C

        if type == "as_patch":
            # 合并两帧，第一帧cls作为新cls，后一帧cls作为patch token
            x = x.view(bsz // 2, num_token * 2, -1) # L,B,C ==> B,L,C ==> B//2, L*2, C
        
        elif type == "mean":
            # 合并两帧，且两帧的cls均值作为新CLS
            x = x.view(bsz // video_frame, video_frame // 2, 2, num_token, -1) # B,T/2,2,L,C
            x_cls, x = x[..., :1, :], x[..., 1:, :].contiguous() # B,T/2,2,1,C  B,T/2,2,L-1,C
            x_cls = x_cls.mean(dim=-3)
            x = x.view(bsz // video_frame, video_frame // 2, 2 * (num_token - 1), -1) # B,T/2,2*(L-1),C
            x = torch.cat([x_cls, x], dim=-2)
            x = x.view(bsz // 2, 2 * num_token - 1, -1)
        
        elif type == "drop":
            # 合并两帧，第一帧CLS作为新CLS，后一帧CLS直接扔掉
            x = x.view(bsz // 2, 2, num_token, -1) # L,B,C ==> B,L,C ==> B//2, 2, L, C
            x = torch.cat([x[:, 0].contiguous(), x[:, 1, 1:]], dim=1) # B//2,L,C + B//2,L-1,C ==> B//2,2L-1,C

        elif type == "keep":
            if not odd:
                x = x.view(bsz // 2, 2, num_token, -1) # L,B,C ==> B,L,C ==> B//2, 2, L, C
                x = torch.cat([x[:, 0, :num_cls].contiguous(), x[:, 1, :num_cls].contiguous(), x[:, 0, num_cls:], x[:, 1, num_cls:]], dim=1) # CLS1 + CLS2 + Patches1 + Patches2
            else:
                x = x.view(bsz // 3, 3, num_token, -1)
                x = torch.cat([
                    x[:, 0, :num_cls].contiguous(),
                    x[:, 1, :num_cls].contiguous(),
                    x[:, 2, :num_cls].contiguous(),
                    x[:, 0, num_cls:].contiguous(),
                    x[:, 1, num_cls:].contiguous(),
                    x[:, 2, num_cls:].contiguous(),
                ], dim=1)
            # x = torch.cat([x[:, 0, :num_cls].contiguous(), x[:, 0, num_cls:], x[:, 1, :num_cls].contiguous(), x[:, 1, num_cls:]], dim=1)

        elif type == "alter":
            # token交替排列 t1 t2 t1 t2 t1 t2 ...
            x = x.view(bsz // 2, 2, num_token, -1) #  B,L,C ==> B//2, 2, L, C
            x = x.permute(1, 2).contiguous().view(bsz // 2, 2 * num_token, -1) # B//2, 2L, C

        if permute:
            return x.permute(1, 0, 2)
        else:
            return x
    
    def frame_flatten(self, x, video_frame, num_cls=1, type="as_patch"):
        if self._tome_info["hierarchy"] and self._tome_info["layer_idx"] in self._tome_info["frame_flatten_layer"]:
            x = self.frame_flatten_operation(x, video_frame, num_cls=num_cls, type=type)
            if self._tome_info["size"] is not None:
                self._tome_info["size"] = self.frame_flatten_operation(self._tome_info["size"], video_frame, num_cls=num_cls, type=type)
            self._tome_info["num_cls"] *= 2
            video_frame = video_frame // 2
        return x, video_frame
    
    def padding_last(self, x, video_frame, batch_first=False):
        if batch_first:
            num_frame, seq_len, C = x.shape
            x = x.view(num_frame // video_frame, video_frame, seq_len, -1)
            x = torch.cat([x, x[:, -1:]], dim=1).view(-1, seq_len, C)
        else:
            seq_len, num_frame, C = x.shape
            x = x.view(seq_len, num_frame // video_frame, video_frame, -1)
            x = torch.cat([x, x[:, :, -1:]], dim=-2).view(seq_len, -1, C) # L,B(T+1),C
        return x

    def frame_flatten_plus(self, x, metric, frame_idx, attn_weights, video_frame):
        # 同时将metric和attn_weights都flatten
        # x: L,BT,C
        # metric: BT,L,C
        # attn_weights: BT,L,L
        num_cls=self._tome_info["num_cls"]
        type=self._tome_info["frame_flatten_type"]
        merge_frame = self._tome_info["hierarchy"] and self._tome_info["layer_idx"] in self._tome_info["frame_flatten_layer"]
        if merge_frame:
            # if video_frame % 2:
            #     # 如果是奇数帧，则重复最后一帧
            #     x = self.padding_last(x, video_frame)
            #     metric = self.padding_last(metric, video_frame, batch_first=True)
            #     frame_idx = self.padding_last(frame_idx, video_frame, batch_first=True) if frame_idx is not None else frame_idx
            #     attn_weights = self.padding_last(attn_weights, video_frame, batch_first=True) if attn_weights is not None else None
            #     self._tome_info["size"] = self.padding_last(self._tome_info["size"], video_frame) if self._tome_info["size"] is not None else None
            #     video_frame += 1
                
            x = self.frame_flatten_operation(x, video_frame, num_cls=num_cls, type=type, odd=video_frame%2==1)
            metric = self.frame_flatten_operation(metric, video_frame, num_cls=num_cls, type=type, permute=False, odd=video_frame%2==1)
            frame_idx = self.frame_flatten_operation(frame_idx, video_frame, num_cls=num_cls, type=type, permute=False, odd=video_frame%2==1)if frame_idx is not None else frame_idx
            if attn_weights is not None:
                attn_weights = attn_weights[:, :num_cls].mean(1).squeeze(-1) # B,L
                attn_weights = self.frame_flatten_operation(attn_weights, video_frame, num_cls=num_cls, type=type, permute=False, odd=video_frame%2==1)
            if self._tome_info["size"] is not None:
                self._tome_info["size"] = self.frame_flatten_operation(self._tome_info["size"], video_frame, num_cls=num_cls, type=type, odd=video_frame%2==1)
            if self._tome_info["source"] is not None:
                merge_track = self._tome_info["source"] # BT, L', L
                bsz, num_center, num_token = merge_track.shape
                if video_frame%2==1:
                    merge_track = merge_track.view(bsz // 3, 3, num_center, num_token) # BT//2, 2, L', L
                    pad_track = torch.zeros((bsz//3, num_center, num_token), device=merge_track.device)
                    new_merge_track1 = torch.cat([merge_track[:, 0], pad_track, pad_track], dim=-1)
                    new_merge_track2 = torch.cat([pad_track, merge_track[:, 1], pad_track], dim=-1)
                    new_merge_track3 = torch.cat([pad_track, pad_track, merge_track[:, 2]], dim=-1)
                    new_merge_track = torch.stack([new_merge_track1, new_merge_track2, new_merge_track3], dim=1).view(bsz, num_center, -1)
                    
                else:
                    merge_track = merge_track.view(bsz // 2, 2, num_center, num_token) # BT//2, 2, L', L
                    pad_track = torch.zeros((bsz//2, num_center, num_token), device=merge_track.device)
                    new_merge_track1 = torch.cat([merge_track[:, 0], pad_track], dim=-1)
                    new_merge_track2 = torch.cat([pad_track, merge_track[:, 1]], dim=-1)
                    new_merge_track = torch.stack([new_merge_track1, new_merge_track2], dim=1).view(bsz, num_center, -1)
                self._tome_info["source"] = self.frame_flatten_operation(new_merge_track, video_frame, num_cls=num_cls, type=type, permute=False, odd=video_frame%2==1)

            self._tome_info["num_cls"] = self._tome_info["num_cls"] * 3 if video_frame % 2 else self._tome_info["num_cls"] * 2
            video_frame = video_frame // 2
        return x, metric, frame_idx, attn_weights, video_frame

    def token_merging(self, x, metric, frame_idx, attn_weights, video_frame, merge_frame=None):

        r = self._tome_info["r"].pop(0)
        num_token = x.shape[0]
        merge_frame = self._tome_info["hierarchy"] and self._tome_info["layer_idx"] in self._tome_info["frame_flatten_layer"] if merge_frame is None else merge_frame
        merge_type = self._tome_info["type"] if not merge_frame else "soft_f_multi_cls"
        average_type = self._tome_info["average_type"]
        size_mode = "mean" if "_f_" in merge_type else "sum"
        if self._tome_info["flatten_layer_merge_type"] is not None:
            if self._tome_info["layer_idx"] in self._tome_info["frame_flatten_layer"]:
                merge_type_flag = self._tome_info["flatten_layer_merge_type"].pop(0)
                merge_type = "soft_f_multi_cls" if merge_type_flag == "T" else "soft_s_multi_cls"
                size_mode = "mean" if "_f_" in merge_type else "sum"
        ## debug ##
        # if self._tome_info["layer_idx"] in [0,]:
        #     merge_type = "soft_f_multi_cls"
        #     size_mode = "mean"
        # if self._tome_info["layer_idx"] in [4, 6]:
        #     merge_type = "soft_s_multi_cls"
        #     size_mode = "sum"
        # if self._tome_info["layer_idx"] in [8,]:
        #     merge_type = "soft_s_multi_cls"
        #     size_mode = "sum"
        ###########
        ## debug ##
        # size_mode = "sum"
        ###########
        enable_merge_frame_mask = self._tome_info["layer_idx"] in self._tome_info["merge_frame_mask_layer"] and self._tome_info["enable_frame_mask"]

        if r > 0:
            # Apply ToMe here
            num_cls = self._tome_info["num_cls"] if self._tome_info["frame_flatten_type"] == "keep" else 1
            if attn_weights is not None:
                # print(attn_weights.shape)
                # print(attn_weights)
                attn_weights = attn_weights[:, :num_cls].mean(1) if len(attn_weights.shape) > 2 else attn_weights
            merge, _ = bipartite_matching_mapping[merge_type](
                metric,
                r,
                video_frame=video_frame,
                frame_idx=frame_idx if enable_merge_frame_mask else None,
                class_token=self._tome_info["class_token"],
                distill_token=self._tome_info["distill_token"],
                token_importance=attn_weights,
                alpha=self._tome_info["importance_alpha"],
                num_cls=self._tome_info["num_cls"],
                binary_frame_mask=self._tome_info["binary_frame_mask"],
            )

            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )

            
            x, frame_idx, self._tome_info["size"] = merge_wavg_mask(merge, x, frame_idx, self._tome_info["size"], type=average_type, size_mode=size_mode, inter_merge=merge_frame, binary_frame_mask=self._tome_info["binary_frame_mask"],)
            
        return x, frame_idx
    
    def update_attn_mask(self, x):
        num_token, bsz, hidden_dim = x.shape
        assert self._tome_info['source'] is not None
        source = self._tome_info['source']
        source = source.view(bsz, num_token, 197, -1).sum(dim=2)
        relation = source @ source.permute(0, 2, 1)
        attn_mask = torch.where(relation == 0, -1e9, 0).repeat(self.attn.num_heads, 1, 1).to(dtype=x.dtype)
        return attn_mask
        
        
    
    def token_sharing_and_merging(self, x, metric, frame_idx, attn_weights, video_frame):
        x, metric, frame_idx, attn_weights, video_frame = self.frame_flatten_plus(x, metric, frame_idx, attn_weights, video_frame)
        x, frame_idx = self.token_merging(x, metric, frame_idx, attn_weights, video_frame)
        return x, frame_idx, video_frame
    
    def init_frame_idx(self, x, video_frame, binary=False):
        if not binary:
            frame_idx = torch.arange(video_frame, device=x.device)[None, :, None].repeat(x.shape[1] // video_frame, 1, x.shape[0]) # B,T,L
        else:
            frame_idx = torch.tensor([0, 1]* (video_frame//2), device=x.device, dtype=torch.long)[None, :, None].repeat(x.shape[1] // video_frame, 1, x.shape[0])
        frame_idx = frame_idx.view(x.shape[1], x.shape[0], 1).to(x.dtype) # BT,L,1
        return frame_idx

    def forward(self, x_dict:dict):
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x, video_frame = x_dict["x"], x_dict["video_frame"]
        hidden_states, frame_idx = x_dict["hidden_states"], x_dict["frame_idx"]
        enable_attn_frame_mask = self._tome_info["layer_idx"] in self._tome_info["attn_frame_mask_layer"] and self._tome_info["enable_frame_mask"]

        # self._tome_info["enable_frame_mask"]=True的时候，才会生成frame_idx
        frame_idx = self.init_frame_idx(x, video_frame, binary=self._tome_info["binary_frame_mask"]) if frame_idx is None and self._tome_info["enable_frame_mask"] else frame_idx

        x_attn, metric, attn_weights = self.attention(self.ln_1(x), attn_size if self._tome_info["layer_idx"] > 0 else None, frame_idx if enable_attn_frame_mask else None, binary_mask=self._tome_info["binary_frame_mask"])
        x = x + x_attn

        # print(x.shape, end="")
        # 当要使用binary mask时，如果frame_idx已经是合并过的，且该层又要进行时序合并了，则需要重新初始化frame_idx
        # 记录一下每一层的shape：
        if self._tome_info["layer_idx"] == 0:
            self._tome_info["tensor_shape"] = []
            self._tome_info["tensor_shape"].append([video_frame, x.shape[0]])
        if self._tome_info["layer_idx"] in self._tome_info["frame_flatten_layer"] and self._tome_info["binary_frame_mask"]:
            frame_idx = self.init_frame_idx(x, video_frame, binary=True) if self._tome_info["enable_frame_mask"] else None
        x, frame_idx, video_frame = self.token_sharing_and_merging(x, metric, frame_idx, attn_weights, video_frame)
        x = x + self.mlp(self.ln_2(x))
        # print(x.shape)

        self._tome_info["layer_idx"] += 1
        self._tome_info["tensor_shape"].append([video_frame, x.shape[0]])
        if hidden_states is not None:
            hidden_states.append(x)

        return {
            "x": x,
            "video_frame": video_frame,
            "hidden_states": hidden_states,
            "frame_idx": frame_idx,
        }


class ToMeMultiheadAttention(nn.MultiheadAttention):
    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
            or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
            :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
            Queries are compared against key-value pairs to produce the output.
            See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
            or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
            :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
            See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
            ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
            sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
            See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
            Binary and byte masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
            value will be ignored.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
            heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
            effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
          :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
          where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
          embedding dimension ``embed_dim``.
        - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
          returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_weights=False``, returns attention weights per
          head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

        .. note::
            `batch_first` argument is ignored for unbatched inputs.
        """
        is_batched = query.dim() == 3
        why_not_fast_path = ''
        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.dropout:
            why_not_fast_path = f"dropout was {self.dropout}, required zero"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif attn_mask is not None:
            why_not_fast_path = "attn_mask was not None"
        elif query.is_nested and key_padding_mask is not None:
            why_not_fast_path = "key_padding_mask is not supported with NestedTensor input"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all([(x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]):
                why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any([x.requires_grad for x in tensor_args]):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                return torch._native_multi_head_attention(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    key_padding_mask if key_padding_mask is not None else attn_mask,
                    need_weights,
                    average_attn_weights)
        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, metric, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights)
        else:
            attn_output, metric, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), metric, attn_output_weights
        else:
            return attn_output, metric, attn_output_weights

def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True


    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if torch.overrides.has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
            average_attn_weights=average_attn_weights,
        )

    is_batched = F._mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = F._in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))
    
    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    # attn_output, attn_output_weights = F._scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    # print(q.dtype)
    attn_output = xops.memory_efficient_attention(q, k, v, p=dropout_p, scale=q.size(-1) ** -0.5, attn_bias=attn_mask)
    # attn_output = xops.memory_efficient_attention(q.reshape(bsz, num_heads, tgt_len, -1).permute(0, 2, 1, 3), 
    #                                               k.reshape(bsz, num_heads, src_len, -1).permute(0, 2, 1, 3), 
    #                                               v.reshape(bsz, num_heads, src_len, -1).permute(0, 2, 1, 3), 
    #                                               p=dropout_p, scale=q.size(-1) ** -0.5, 
    #                                               attn_bias=attn_mask.reshape(bsz, num_heads, tgt_len, src_len) if attn_mask is not None else None)
    # attn_output = attn_output.permute(0, 2, 1, 3).reshape(bsz * num_heads, tgt_len, -1)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
    # print(k.shape)

    if need_weights:
        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, k.view(bsz, num_heads, -1, head_dim).mean(1), attn_output_weights
    else:
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, k.view(bsz, num_heads, -1, head_dim).mean(1), None