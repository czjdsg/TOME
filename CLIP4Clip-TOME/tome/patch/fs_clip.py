import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Tuple, Optional, Union, List

from modules.module_clip import ResidualAttentionBlock, VisualTransformer
from tome.merge import bipartite_matching_mapping, bipartite_soft_matching_st_clip, merge_source, merge_wavg
from tome.utils import parse_r

def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.transformer.resblocks), self.r)
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
    def attention(self, x: torch.Tensor, size: torch.Tensor):
        """
        size: L,N,1
        """
        attn_mask_ = self.attn_mask
        if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
            attn_mask_ = self.attn_mask(x.size(0))   # LND

        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        if size is not None:
            size = size.log().transpose(0, 1).unsqueeze(1).unsqueeze(1)[...,0] # N,1,1,L
            size = size.repeat(1, self.attn.num_heads, x.size(0), 1).view(-1, x.size(0), x.size(0)) # N*num_heads, L, L
            attn_mask_ = attn_mask_ + size if attn_mask_ is not None else size
        if "token_importance" in self._tome_info.keys():
            need_weights = self._tome_info["token_importance"]
        else:
            need_weights = False
        attn_output, metric, attn_weights, video_k, video_v = self.attn(x, x, x, need_weights=need_weights, attn_mask=attn_mask_)

        return attn_output, metric, attn_weights, video_k, video_v # LBC; B,L,head_dim; B*num_heads, L, C

    def frame_flatten_operation(self, x, video_frame, num_cls=1, type="as_patch"):
        num_token, bsz, hidden_dim = x.shape # L,B,C
        if type == "as_patch":
            # 合并两帧，第一帧cls作为新cls，后一帧cls作为patch token
            x = x.permute(1, 0, 2).contiguous().view(bsz // 2, num_token * 2, hidden_dim) # L,B,C ==> B,L,C ==> B//2, L*2, C
        
        elif type == "mean":
            # 合并两帧，且两帧的cls均值作为新CLS
            x = x.permute(1, 0, 2).contiguous().view(bsz // video_frame, video_frame // 2, 2, num_token, -1) # B,T/2,2,L,C
            x_cls, x = x[..., :1, :], x[..., 1:, :].contiguous() # B,T/2,2,1,C  B,T/2,2,L-1,C
            x_cls = x_cls.mean(dim=-3)
            x = x.view(bsz // video_frame, video_frame // 2, 2 * (num_token - 1), -1) # B,T/2,2*(L-1),C
            x = torch.cat([x_cls, x], dim=-2)
            x = x.view(bsz // 2, -1, hidden_dim)
        
        elif type == "drop":
            # 合并两帧，第一帧CLS作为新CLS，后一帧CLS直接扔掉
            x = x.permute(1, 0, 2).contiguous().view(bsz // 2, 2, num_token, hidden_dim) # L,B,C ==> B,L,C ==> B//2, 2, L, C
            x = torch.cat([x[:, 0].contiguous(), x[:, 1, 1:]], dim=1) # B//2,L,C + B//2,L-1,C ==> B//2,2L-1,C

        elif type == "keep":
            x = x.permute(1, 0, 2).contiguous().view(bsz // 2, 2, num_token, hidden_dim) # L,B,C ==> B,L,C ==> B//2, 2, L, C
            x = torch.cat([x[:, 0, :num_cls].contiguous(), x[:, 1, :num_cls].contiguous(), x[:, 0, num_cls:], x[:, 1, num_cls:]], dim=1) # CLS1 + CLS2 + Patches1 + Patches2
            # x = torch.cat([x[:, 0, :num_cls].contiguous(), x[:, 0, num_cls:], x[:, 1, :num_cls].contiguous(), x[:, 1, num_cls:]], dim=1)

        return x.permute(1, 0, 2)
    
    def frame_flatten(self, x, video_frame, video_mask, num_cls=1, type="as_patch"):
        if self._tome_info["hierarchy"] and self._tome_info["layer_idx"] in self._tome_info["frame_flatten_layer"]:
            x = self.frame_flatten_operation(x, video_frame, num_cls=num_cls, type=type)
            if self._tome_info["size"] is not None:
                self._tome_info["size"] = self.frame_flatten_operation(self._tome_info["size"], video_frame, num_cls=num_cls, type=type)
            self._tome_info["num_cls"] *= 2
            video_frame = video_frame // 2
            video_mask = video_mask.view(-1, video_frame, 2)[..., 0]
        return x, video_frame, video_mask

    def token_merging(self, x, metric, attn_weights, video_frame):
        r = self._tome_info["r"].pop(0)
        num_token = x.shape[0]
        if r > 0:
            # Apply ToMe here
            num_cls = self._tome_info["num_cls"] if self._tome_info["frame_flatten_type"] == "keep" else 1
            token_importance = attn_weights[:, :num_cls].mean(1) if attn_weights is not None else None # B,L,L
            merge, _ = bipartite_matching_mapping[self._tome_info["type"]](
                metric,
                r,
                video_frame=video_frame,
                class_token=self._tome_info["class_token"],
                distill_token=self._tome_info["distill_token"],
                token_importance=token_importance,
                alpha=self._tome_info["importance_alpha"],
                num_cls=self._tome_info["num_cls"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"], type=self._tome_info["average_type"])
        return x

    def gather_video_feature(self, g_cls, video_k, video_v, video_mask, video_frame):
        # g_cls: 1,N,D
        # video_k, video_v: NT*num_head,L,D//num_head
        # video_mask: N,T
        tgt_len, bsz, hidden_dim = g_cls.shape
        num_head = video_k.shape[0] // bsz // video_frame
        head_dim = video_k.shape[-1]
        q_g_cls = F.linear(g_cls, self.attn.in_proj_weight[:hidden_dim], self.attn.in_proj_bias[:hidden_dim]) # 1,N,D
        q_g_cls = q_g_cls.contiguous().view(tgt_len, bsz * num_head, video_k.shape[-1]).transpose(0, 1)
        # prepare video_mask with shape: N*num_head, 1, T*L
        new_video_mask = torch.zeros_like(video_mask, dtype=q_g_cls.dtype)
        new_video_mask.masked_fill_((1-video_mask)==1, float("-inf"))
        video_mask_expanded = new_video_mask[:, None, None, :, None].expand(-1, num_head, -1, -1, video_k.shape[1]) # N,num_head,1,T,L
        video_mask_expanded = video_mask_expanded.contiguous().view(bsz * num_head, 1, -1) # N*num_head,1,TL
        if not self.attn.training:
            dropout_p = 0.
        else:
            dropout_p = self.attn.dropout
        # flatten the temporal dim of video_k and video_v
        video_k = video_k.view(bsz, video_frame, num_head, -1, head_dim).transpose(1, 2)
        video_k = video_k.contiguous().view(bsz * num_head, -1, head_dim)
        video_v = video_v.view(bsz, video_frame, num_head, -1, head_dim).transpose(1, 2)
        video_v = video_v.contiguous().view(bsz * num_head, -1, head_dim)
        g_cls_output, _ = F._scaled_dot_product_attention(q_g_cls, video_k ,video_v, video_mask_expanded, dropout_p)
        g_cls_output = g_cls_output.transpose(0, 1).contiguous().view(tgt_len * bsz, hidden_dim)
        g_cls_output = F.linear(g_cls_output, self.attn.out_proj.weight, self.attn.out_proj.bias)
        g_cls_output = g_cls_output.view(tgt_len, bsz, g_cls_output.size(1)) # 1,N,D
        return g_cls_output

    def forward(self, x_tuple: Tuple):
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x, video_frame, g_cls, video_mask, hidden_states, g_cls_states, return_gf_sims = x_tuple

        # spatial or temporal attention of video tokens
        x_attn, metric, attn_weights, video_k, video_v = self.attention(self.ln_1(x), attn_size)
        x = x + x_attn
        x = self.token_merging(x, metric, attn_weights, video_frame)
        x = x + self.mlp(self.ln_2(x)) # L,NT,D
        
        # gather video features with global CLS
        if g_cls is not None:
            g_cls = g_cls + self.gather_video_feature(self.ln_1(g_cls), video_k, video_v, video_mask, video_frame)
            g_cls = g_cls + self.mlp(self.ln_2(g_cls)) # 1,N,D

        # Optionally flatten frames
        x, video_frame, video_mask = self.frame_flatten(x, video_frame, video_mask, num_cls=self._tome_info["num_cls"], type=self._tome_info["frame_flatten_type"])
        self._tome_info["layer_idx"] += 1

        # If training, compute g_cls vs. frame_cls similarity
        if hidden_states is not None and return_gf_sims:
            hidden_states.append(x)
        if g_cls is not None and return_gf_sims:
            g_cls_states.append(g_cls)

        return (x, video_frame, g_cls, video_mask, hidden_states, g_cls_states, return_gf_sims)

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
            attn_output, metric, attn_output_weights, k, v = multi_head_attention_forward(
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
            attn_output, metric, attn_output_weights, k, v = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), metric, attn_output_weights, k, v
        else:
            return attn_output, metric, attn_output_weights, k, v

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
    attn_output, attn_output_weights = F._scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    if need_weights:
        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, k.view(bsz, num_heads, -1, head_dim).mean(1), attn_output_weights, k, v # return key for tome metric
    else:
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, k.view(bsz, num_heads, -1, head_dim).mean(1), None, k, v # 为了方便进行global token和video token之间的attention，返回k和v