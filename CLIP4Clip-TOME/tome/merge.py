# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple

import torch


def do_nothing(x, mode=None):
    return x

def bipartite_soft_matching_clip(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
    video_frame: int = 12,
    token_importance: torch.Tensor = None,
    alpha: float = 6.0,
    num_cls: int = 1,
) -> Tuple[Callable, Callable]:
    """
    the shape of input feature is (L,B,C), L is the sequence length, B is the batch size, C is the feature dim
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True) # B,L,C
        a, b = metric[..., ::2, :], metric[..., 1::2, :] # B,L//2,C
        scores = a @ b.transpose(-1, -2) # B,L//2,L//2

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1) # B,L//2
        if token_importance is not None:
            node_max = node_max - token_importance[:, ::2] * alpha
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] # B,L//2,1

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens, B,L//2 - r, 1
        src_idx = edge_idx[..., :r, :]  # Merged Tokens, B,r,1
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx) # B,r,1

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0] # B,L//2 - r, 1
    
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[::2], x[1::2] # L//2,B,C
        t1, n, c = src.shape
        unm = src.gather(dim=0, index=unm_idx.transpose(0, 1).expand(t1 - r, n, c))
        src = src.gather(dim=0, index=src_idx.transpose(0, 1).expand(r, n, c))
        dst = dst.scatter_reduce(0, dst_idx.transpose(0, 1).expand(r, n, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:1], dst[:1], unm[1:], dst[1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=0)


    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge

def bipartite_soft_matching_clip_multi_cls(
    metric: torch.Tensor,
    r: int,
    frame_idx: torch.Tensor = None,
    class_token: bool = False,
    distill_token: bool = False,
    video_frame: int = 12,
    token_importance: torch.Tensor = None,
    alpha: float = 6.0,
    num_cls: int = 1,
) -> Tuple[Callable, Callable]:
    """
    the shape of input feature is (L,B,C), L is the sequence length, B is the batch size, C is the feature dim
    用于视频merge，multi_cls表示当合并两帧时，保留每一帧的CLS，因此有多个CLS，对于这些CLS，都要把他们分配到UNM中
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.
     - frame_idx: BT,L,num_merge

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += num_cls
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True) # B,L,C
        a_idx = torch.cat([torch.arange(num_cls), torch.arange(num_cls + 1, t, 2)], dim=0)
        b_idx = torch.arange(num_cls, t, 2)
        a, b = metric[..., a_idx, :], metric[..., b_idx, :] # B,L//2,C
        # a, b = metric[..., ::2, :], metric[..., 1::2, :] # B,L//2,C
        
        scores = a @ b.transpose(-1, -2) # B,L//2,L//2

        if class_token:
            scores[..., :num_cls, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf
        if frame_idx is not None:
            frame_idx_a, frame_idx_b = frame_idx[..., a_idx, :], frame_idx[..., b_idx, :]
            frame_mask = (frame_idx_a.unsqueeze(2) == frame_idx_b.unsqueeze(1)).all(dim=-1)
            scores.masked_fill_(~frame_mask, -math.inf)

        node_max, node_idx = scores.max(dim=-1) # B,L//2
        if token_importance is not None:
            node_max = node_max - token_importance[:, a_idx] * alpha
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] # B,L//2,1

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens, B,L//2 - r, 1
        src_idx = edge_idx[..., :r, :]  # Merged Tokens, B,r,1
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx) # B,r,1

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0] # B,L//2 - r, 1
    
    def merge(x: torch.Tensor, mode="mean", permute=True) -> torch.Tensor:
        if permute:
            # src, dst = x[::2], x[1::2] # L//2,B,C
            src, dst = x[a_idx], x[b_idx] # L//2,B,C
            t1, n, c = src.shape
            unm = src.gather(dim=0, index=unm_idx.transpose(0, 1).expand(t1 - r, n, c))
            src = src.gather(dim=0, index=src_idx.transpose(0, 1).expand(r, n, c))
            dst = dst.scatter_reduce(0, dst_idx.transpose(0, 1).expand(r, n, c), src, reduce=mode)

            if distill_token:
                return torch.cat([unm[:1], dst[:1], unm[1:], dst[1:]], dim=1)
            else:
                return torch.cat([unm, dst], dim=0)
        else:
            src, dst = x[..., a_idx, :], x[..., b_idx, :] # B,L//2,C
            n, t1, c = src.shape
            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c)) # B,L//2 - r, C
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c)) # B,r,C
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode) # B,r,C

            if distill_token:
                return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
            else:
                return torch.cat([unm, dst], dim=1)


    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def bipartite_soft_matching_clip_multi_cls_v1(
    metric: torch.Tensor,
    r: int,
    frame_idx: torch.Tensor = None,
    class_token: bool = False,
    distill_token: bool = False,
    video_frame: int = 12,
    token_importance: torch.Tensor = None,
    alpha: float = 6.0,
    num_cls: int = 1,
    binary_frame_mask = False,
) -> Tuple[Callable, Callable]:
    """
    the shape of input feature is (L,B,C), L is the sequence length, B is the batch size, C is the feature dim
    除了一帧中多个CLS的方式不同：
    之前：对于这些CLS，都要把他们分配到UNM中
    现在：由于tensor[a_idx]速度太慢，还是用::2的方式进行分离，然后src和dst中的前num_cls//2都是CLS
    将dst中的当作distill token处理即可
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.
     - frame_idx: BT,L,num_merge

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += num_cls
    distill_token = True if num_cls > 1 else False
    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True) # B,L,C
        a, b = metric[..., ::2, :], metric[..., 1::2, :] # B,L//2,C
        
        scores = a @ b.transpose(-1, -2) # B,L//2,L//2

        if class_token:
            if num_cls > 1:
                scores[..., :num_cls//2, :] = -math.inf
            else:
                scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, :num_cls//2] = -math.inf
        
        if frame_idx is not None:
            if not binary_frame_mask:
                frame_idx_a, frame_idx_b = frame_idx[..., ::2, :], frame_idx[..., 1::2, :]
                frame_mask = (frame_idx_a.unsqueeze(2) == frame_idx_b.unsqueeze(1)).any(dim=-1)
                # frame_mask = (frame_idx_a.unsqueeze(2) == frame_idx_b.unsqueeze(1)).all(dim=-1)
            else:
                frame_idx_a, frame_idx_b = frame_idx[..., ::2, :].squeeze(-1), frame_idx[..., 1::2, :].squeeze(-1)
                frame_mask_a_0 = frame_idx_a < 1.
                frame_mask_b_0 = frame_idx_b < 1.
                frame_mask_0 = torch.logical_and(frame_mask_a_0.unsqueeze(2), frame_mask_b_0.unsqueeze(1)) # B,L,L
                frame_mask_a_1 = frame_idx_a > 0.
                frame_mask_b_1 = frame_idx_b > 0.
                frame_mask_1 = torch.logical_and(frame_mask_a_1.unsqueeze(2), frame_mask_b_1.unsqueeze(1)) # B,L,L
                frame_mask = torch.logical_or(frame_mask_0, frame_mask_1)
            scores.masked_fill_(~frame_mask, -math.inf)

        node_max, node_idx = scores.max(dim=-1) # B,L//2
        if token_importance is not None:
            node_max = node_max - token_importance[:, ::2] * alpha
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] # B,L//2,1

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens, B,L//2 - r, 1
        src_idx = edge_idx[..., :r, :]  # Merged Tokens, B,r,1
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx) # B,r,1

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0] # B,L//2 - r, 1
    
    def merge(x: torch.Tensor, mode="mean", permute=True) -> torch.Tensor:
        if permute:
            src, dst = x[::2], x[1::2] # L//2,B,C
            t1, n, c = src.shape
            unm = src.gather(dim=0, index=unm_idx.transpose(0, 1).expand(t1 - r, n, c))
            if mode != "keepone":
                src = src.gather(dim=0, index=src_idx.transpose(0, 1).expand(r, n, c))
                dst = dst.scatter_reduce(0, dst_idx.transpose(0, 1).expand(r, n, c), src, reduce=mode)

            if distill_token:
                return torch.cat([unm[:num_cls//2], dst[:num_cls//2], unm[num_cls//2:], dst[num_cls//2:]], dim=0)
            else:
                return torch.cat([unm, dst], dim=0)
        else:
            src, dst = x[..., ::2, :], x[..., 1::2, :] # L//2,B,C
            n, t1, c = src.shape
            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c)) # B,L//2 - r, C
            if mode != "keepone":
                src = src.gather(dim=-2, index=src_idx.expand(n, r, c)) # B,r,C
                dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode) # B,r,C

            if distill_token:
                return torch.cat([unm[:, :num_cls//2], dst[:, :num_cls//2], unm[:, num_cls//2:], dst[:, num_cls//2:]], dim=1)
            else:
                return torch.cat([unm, dst], dim=1)


    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def bipartite_soft_matching_frame_multi_cls(
    metric: torch.Tensor,
    r: int,
    frame_idx: torch.Tensor = None,
    class_token: bool = False,
    distill_token: bool = False,
    video_frame: int = 12,
    token_importance: torch.Tensor = None,
    alpha: float = 6.0,
    num_cls: int = 1,
    binary_frame_mask = False,
) -> Tuple[Callable, Callable]:
    """
    用于进行帧间合并，set a中包含CLS token和后一帧的patch，set b中包含前一帧的patch。然后将set中的patch合并到set b中。
    需要注意的是，更自然的做法是set a中包含CLS和前一帧的patch，set b中包含后一帧的patch。但是这样会有问题，即将前一帧的patch merge到后一帧patch中，但CLS确实前一帧的patch。
    暂时还不清楚这样会有什么样的影响，需要进行实验验证。
    """
    protected = 0
    if class_token:
        protected += num_cls
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)
    num_frame_token = (t - num_cls) // 2 # 每一帧的token数量

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True) # B,L,C
        a_idx = torch.cat([torch.arange(num_cls), torch.arange(num_cls + num_frame_token, t)], dim=0)
        b_idx = torch.arange(num_cls, num_cls + num_frame_token)
        a, b = metric[..., a_idx, :], metric[..., b_idx, :] # B,L//2,C
        scores = a @ b.transpose(-1, -2) # B,L//2,L//2

        if class_token:
            scores[..., :num_cls, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1) # B,L//2
        if token_importance is not None:
            node_max = node_max - token_importance[:, a_idx] * alpha
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] # B,L//2,1

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens, B,L//2 - r, 1
        src_idx = edge_idx[..., :r, :]  # Merged Tokens, B,r,1
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx) # B,r,1

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0] # B,L//2 - r, 1
    
    def merge(x: torch.Tensor, mode="mean", permute=True) -> torch.Tensor:
        if permute:
            src, dst = x[a_idx], x[b_idx] # L//2,B,C
            t1, n, c = src.shape
            unm = src.gather(dim=0, index=unm_idx.transpose(0, 1).expand(t1 - r, n, c))
            if mode != "keepone":
                src = src.gather(dim=0, index=src_idx.transpose(0, 1).expand(r, n, c))
                dst = dst.scatter_reduce(0, dst_idx.transpose(0, 1).expand(r, n, c), src, reduce=mode)

            if distill_token:
                return torch.cat([unm[:1], dst[:1], unm[1:], dst[1:]], dim=1)
            else:
                return torch.cat([unm, dst], dim=0)
        else:
            src, dst = x[..., a_idx, :], x[..., b_idx, :] # B,L//2,C
            n, t1, c = src.shape
            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c)) # B,L//2 - r, C
            if mode != "keepone":
                src = src.gather(dim=-2, index=src_idx.expand(n, r, c)) # B,r,C
                dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode) # B,r,C

            if distill_token:
                return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
            else:
                return torch.cat([unm, dst], dim=1)


    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def bipartite_soft_matching_frame_multi_cls_v2(
    metric: torch.Tensor,
    r: int,
    frame_idx: torch.Tensor = None,
    class_token: bool = False,
    distill_token: bool = False,
    video_frame: int = 12,
    token_importance: torch.Tensor = None,
    alpha: float = 6.0,
    num_cls: int = 1,
    binary_frame_mask = False,
) -> Tuple[Callable, Callable]:
    """
    输入为两帧token交替排布，此时只需要按照ToMe原来的交替采样分组，就能完成帧间的merge
    对于set a和b的选择，可以将前一帧当作set a，也可以将后一帧当作set a。目前将后一帧当作set a，因为set a会被合并到set b中
    我们想让后一帧合并到前一帧中
    """
    protected = 0
    if class_token:
        protected += num_cls
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)
    num_frame_token = (t - num_cls) // 2 # 每一帧的token数量

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True) # B,L,C
        # a中是第二帧的token，b中是第一帧的token
        a, b = metric[..., 1::2, :], metric[..., ::2, :] # B,L//2,C
        scores = a @ b.transpose(-1, -2) # B,L//2,L//2

        if class_token:
            # 输入的num_cls已经乘以2了，表示合并帧中的CLS数，a和b中只包含num_cls // 2个cls
            # 所以这里要使用num_cls // 2
            scores[..., :num_cls // 2, :] = -math.inf
            scores[..., :, :num_cls // 2] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1) # B,L//2
        if token_importance is not None:
            node_max = node_max - token_importance[:, 1::2] * alpha
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] # B,L//2,1

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens, B,L//2 - r, 1
        src_idx = edge_idx[..., :r, :]  # Merged Tokens, B,r,1
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx) # B,r,1

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0] # B,L//2 - r, 1
    
    def merge(x: torch.Tensor, mode="mean", permute=True) -> torch.Tensor:
        if permute:
            src, dst = x[1::2], x[::2] # L//2,B,C
            t1, n, c = src.shape
            unm = src.gather(dim=0, index=unm_idx.transpose(0, 1).expand(t1 - r, n, c))
            src = src.gather(dim=0, index=src_idx.transpose(0, 1).expand(r, n, c))
            dst = dst.scatter_reduce(0, dst_idx.transpose(0, 1).expand(r, n, c), src, reduce=mode)

            return torch.cat([unm[:num_cls//2], dst[:num_cls//2], unm[num_cls//2:], dst[num_cls//2:]], dim=1)
        else:
            src, dst = x[..., 1::2, :], x[..., ::2, :] # B,L//2,C
            n, t1, c = src.shape
            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c)) # B,L//2 - r, C
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c)) # B,r,C
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode) # B,r,C
            return torch.cat([unm[:, :num_cls//2], dst[:, :num_cls//2], unm[:, num_cls//2:], dst[:, num_cls//2:]], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge

def greedy_matching_frame_multi_cls(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
    video_frame: int = 12,
    token_importance: torch.Tensor = None,
    alpha: float = 6.0,
    num_cls: int = 1,
) -> Tuple[Callable, Callable]:
    """
    用于进行帧间合并，set a中包含CLS token和后一帧的patch，set b中包含前一帧的patch。然后将set中的patch合并到set b中。
    需要注意的是，更自然的做法是set a中包含CLS和前一帧的patch，set b中包含后一帧的patch。但是这样会有问题，即将前一帧的patch merge到后一帧patch中，但CLS确实前一帧的patch。
    暂时还不清楚这样会有什么样的影响，需要进行实验验证。
    """
    protected = 0
    if class_token:
        protected += num_cls
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)
    num_frame_token = (t - num_cls) // 2 # 每一帧的token数量

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True) # B,L,C
        a_idx = torch.cat([torch.arange(num_cls), torch.arange(num_cls + num_frame_token, t)], dim=0)
        b_idx = torch.arange(num_cls, num_cls + num_frame_token)
        a, b = metric[..., a_idx, :], metric[..., b_idx, :] # B,L//2,C
        scores = a @ b.transpose(-1, -2) # B,L//2,L//2

        if class_token:
            scores[..., :num_cls, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        batch_size, m, n = scores.shape
        flat_sim = scores.view(batch_size, -1)
        sorted_indices = torch.argsort(flat_sim, dim=1, descending=True)

        match_A = torch.full((batch_size, m), 0, dtype=torch.long, device=scores.device)
        score_A = torch.full((batch_size, m), -1., dtype=scores.dtype, device=scores.device)

        mask_A = torch.full((batch_size, m), False, dtype=torch.bool, device=scores.device)
        mask_B = torch.full((batch_size, n), False, dtype=torch.bool, device=scores.device)

        row_indices = sorted_indices // n
        col_indices = sorted_indices % n

        # for i in range(m * n):
        #     # B
        #     # import pdb; pdb.set_trace()
        #     if mask_A.sum() == batch_size * m or mask_B.sum() == batch_size * n:
        #         break
        #     valid = ~mask_A[torch.arange(batch_size), row_indices[:, i]] & ~mask_B[torch.arange(batch_size), col_indices[:, i]]
        #     match_A[torch.arange(batch_size)[valid], row_indices[valid, i]] = col_indices[valid, i]
        #     score_A[torch.arange(batch_size)[valid], row_indices[valid, i]] = scores[torch.arange(batch_size)[valid], row_indices[valid, i], col_indices[valid, i]]
        #     mask_A[torch.arange(batch_size)[valid], row_indices[valid, i]] = True
        #     mask_B[torch.arange(batch_size)[valid], col_indices[valid, i]] = True

        for b_id in range(batch_size):
            for i in range(m * n):
                if not mask_A[b_id, row_indices[b_id, i]] and not mask_B[b_id, col_indices[b_id, i]]:
                    match_A[b_id, row_indices[b_id, i]] = col_indices[b_id, i]
                    score_A[b_id, row_indices[b_id, i]] = scores[b_id, row_indices[b_id, i], col_indices[b_id, i]]
                    mask_A[b_id, row_indices[b_id, i]] = True
                    mask_B[b_id, col_indices[b_id, i]] = True

        node_max, node_idx = score_A, match_A
        if token_importance is not None:
            node_max = node_max - token_importance[:, a_idx] * alpha
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] # B,L//2,1

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens, B,L//2 - r, 1
        src_idx = edge_idx[..., :r, :]  # Merged Tokens, B,r,1
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx) # B,r,1

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0] # B,L//2 - r, 1
    
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        
        src, dst = x[a_idx], x[b_idx] # L//2,B,C
        t1, n, c = src.shape
        unm = src.gather(dim=0, index=unm_idx.transpose(0, 1).expand(t1 - r, n, c))
        src = src.gather(dim=0, index=src_idx.transpose(0, 1).expand(r, n, c))
        dst = dst.scatter_reduce(0, dst_idx.transpose(0, 1).expand(r, n, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:1], dst[:1], unm[1:], dst[1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=0)


    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge

def bipartite_soft_matching_st_clip(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
    video_frame: int = 12,
    token_importance: torch.Tensor = None,
    alpha: float = 6.0,
) -> Tuple[Callable, Callable]:
    """
    the shape of input feature is (L,B,C), L is the sequence length, B is the batch size, C is the feature dim
    Applies ToMe with a balanced matching set (50%, 50%).
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1
    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1] # B,L,C
    r = min(r, (t - protected) // 2)
    bsz, _, hidden_size = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True) # B,L,C
        a, b = metric[..., ::2, :].contiguous(), metric[..., 1::2, :].contiguous() # B,L//2,C
        a = a.view(bsz // video_frame, -1, hidden_size) # B,Vt1,C
        b = b.view(bsz // video_frame, -1, hidden_size) # B,Vt2,C
        scores = a @ b.transpose(-1, -2) # B,Vt1,Vt2

        if class_token:
            scores[..., torch.arange(0, a.shape[1], a.shape[1] // video_frame), :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        # scores_mask = torch.zeros_like(scores)
        # for i in range(video_frame):
        #     scores_mask[:, i * a.shape[1] // video_frame : (i+1) * a.shape[1] // video_frame, i * a.shape[1] // video_frame : (i+1) * a.shape[1] // video_frame] = 1.
        # scores += scores_mask

        node_max, node_idx = scores.max(dim=-1) # B,Vt1
        if token_importance is not None:
            node_max = node_max - token_importance[:, ::2].contiguous().view(bsz // video_frame, -1) * alpha
        node_max = node_max.view(bsz // video_frame, video_frame, -1) # B,V,t1
        edge_idx = node_max.argsort(dim=-1, descending=True) # B,V,t1
        edge_idx += torch.arange(video_frame, device=node_idx.device)[None, :, None] * (a.shape[1] // video_frame)

        unm_idx = edge_idx[..., r:].contiguous().view(bsz // video_frame, -1, 1) # Unmerged Tokens, B,Vxt1-r,1
        src_idx = edge_idx[..., :r].contiguous().view(bsz // video_frame, -1, 1)  # Merged Tokens, B,Vxr,1
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx) # B,Vxr,1

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0] # B,L//2 - r, 1
    
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        x = x.permute(1, 0, 2) # B,L,C
        hidden_dim = x.shape[-1]
        src, dst = x[..., ::2, :].contiguous(), x[..., 1::2, :].contiguous()
        src = src.view(bsz // video_frame, -1, hidden_dim) # B,Vt1,C
        dst = dst.view(bsz // video_frame, -1, hidden_dim) # B,Vt2,C
        n, Vt1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, Vt1 - video_frame * r, c)) # B,L//2 - r, C
        src = src.gather(dim=-2, index=src_idx.expand(n, video_frame * r, c)) # B,r,C
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, video_frame * r, c), src, reduce=mode) # B,r,C
        res = torch.cat([unm, dst], dim=1)

        with torch.no_grad():
            pos = torch.arange(x.shape[1] * video_frame, device=x.device)[None, :, None].repeat(x.shape[0] // video_frame, 1, 1) # B,L,1
            pos = pos.view(x.shape[0], x.shape[1], 1)
            src_pos, dst_pos = pos[..., ::2, :].contiguous(), pos[..., 1::2, :].contiguous()
            src_pos = src_pos.view(bsz // video_frame, -1, 1) # B,Vt1,1
            dst_pos = dst_pos.view(bsz // video_frame, -1, 1) # B,Vt2,C
            unm_pos = src_pos.gather(dim=-2, index=unm_idx.expand(n, Vt1 - video_frame * r, 1))
            pos_res = torch.cat([unm_pos, dst_pos], dim=1) # B,L,1
            resort_idx = pos_res.argsort(dim=-2, descending=False)
        resort_res = res.gather(dim=-2, index=resort_idx.expand(n, res.shape[1], c)).view(bsz, -1, hidden_dim).permute(1, 0, 2)
        return resort_res


    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge

def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True) # B,L,C
        a, b = metric[..., ::2, :], metric[..., 1::2, :] # B,L//2,C
        scores = a @ b.transpose(-1, -2) # B,L//2,L//2

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1) # B,L//2
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] # B,L//2,1

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens, B,L//2 - r, 1
        src_idx = edge_idx[..., :r, :]  # Merged Tokens, B,r,1
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx) # B,r,1

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0] # B,L//2 - r, 1

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :] # B,L//2,C
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c)) # B,L//2 - r, C
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c)) # B,r,C
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode) # B,r,C

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)
    
    def merge_clip(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[::2], x[1::2] # L//2,B,C
        t1, n, c = src.shape
        unm = src.gather(dim=0, index=unm_idx.transpose(0, 1).expand(t1 - r, n, c))
        src = src.gather(dim=0, index=src_idx.transpose(0, 1).expand(r, n, c))
        dst = dst.scatter_reduce(0, dst_idx.transpose(0, 1).expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:1], dst[:1], unm[1:], dst[1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=0)


    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def kth_bipartite_soft_matching(
    metric: torch.Tensor, k: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (every kth element, the rest).
    If n is the number of tokens, resulting number of tokens will be n // z.

    Input size is [batch, tokens, channels].
    z indicates the stride for the first set.
    z = 2 is equivalent to regular bipartite_soft_matching with r = 0.5 * N
    """
    if k <= 1:
        return do_nothing, do_nothing

    def split(x):
        t_rnd = (x.shape[1] // k) * k
        x = x[:, :t_rnd, :].view(x.shape[0], -1, k, x.shape[2])
        a, b = (
            x[:, :, : (k - 1), :].contiguous().view(x.shape[0], -1, x.shape[-1]),
            x[:, :, (k - 1), :],
        )
        return a, b

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        r = a.shape[1]
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, _, c = src.shape
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        n, _, c = x.shape
        dst = x

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c)).to(x.dtype)

        src = src.view(n, -1, (k - 1), c)
        dst = dst.view(n, -1, 1, c)

        out = torch.cat([src, dst], dim=-2)
        out = out.contiguous().view(n, -1, c)

        return out

    return merge, unmerge


def random_bipartite_soft_matching(
    metric: torch.Tensor, r: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (r chosen randomly, the rest).
    Input size is [batch, tokens, channels].

    This will reduce the number of tokens by r.
    """
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        B, N, _ = metric.shape
        rand_idx = torch.rand(B, N, 1, device=metric.device).argsort(dim=1)

        a_idx = rand_idx[:, :r, :]
        b_idx = rand_idx[:, r:, :]

        def split(x):
            C = x.shape[-1]
            a = x.gather(dim=1, index=a_idx.expand(B, r, C))
            b = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
            return a, b

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        C = src.shape[-1]
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        C = x.shape[-1]
        dst = x
        src = dst.gather(dim=-2, index=dst_idx.expand(B, r, C))

        out = torch.zeros(B, N, C, device=x.device, dtype=x.dtype)

        out.scatter_(dim=-2, index=a_idx.expand(B, r, C), src=src)
        out.scatter_(dim=-2, index=b_idx.expand(B, N - r, C), src=dst)

        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None, metric: torch.Tensor = None, scale = 0.5, type="AVG", size_mode="sum",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    scale to prevent overflow
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    if type == "EMA":
        x = merge(scale * x * size, mode="sum")
        size = merge(size, mode=size_mode)
        x = x / size / scale
        if metric is not None:
            metric = merge(metric, mode="mean", permute=False)

    elif type == "AVG":
        x = merge(x, mode="mean")
        size = merge(size, mode=size_mode)
        if metric is not None:
            metric = merge(metric, mode="mean", permute=False)
    
    elif type == "MAX":
        x = merge(x, mode="amax")
        size = merge(size, mode=size_mode)
        if metric is not None:
            metric = merge(metric, mode="amax", permute=False)
    
    elif type == "KEEPONE":
        x = merge(x, mode="keepone")
        size = merge(size, mode=size_mode)
        if metric is not None:
            metric = merge(metric, mode="keepone", permute=False)
    
    else:
        raise NotImplementedError

    # x = merge(x * size, mode="sum")
    # size = merge(size, mode="sum")
    # x = x / size

    if metric is not None:
        return x, size, metric
    else:
        return x, size

def merge_wavg_mask(
    merge: Callable, x: torch.Tensor, frame_idx: torch.Tensor = None, size: torch.Tensor = None, metric: torch.Tensor = None, scale = 0.5, type="AVG", size_mode="sum",
    inter_merge=True, binary_frame_mask=False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    scale to prevent overflow
    frame_idx: L,B,num_frame_merge
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    if type == "EMA":
        x = merge(scale * x * size, mode="sum")
        size = merge(size, mode=size_mode)
        x = x / size / scale
        if metric is not None:
            metric = merge(metric, mode="mean", permute=False)

    elif type == "AVG":
        x = merge(x, mode="mean")
        size = merge(size, mode=size_mode)
        if metric is not None:
            metric = merge(metric, mode="mean", permute=False)

    elif type == "MAX":
        x = merge(x, mode="amax")
        size = merge(size, mode=size_mode)
        if metric is not None:
            metric = merge(metric, mode="amax", permute=False)
    
    elif type == "KEEPONE":
        x = merge(x, mode="keepone")
        size = merge(size, mode=size_mode)
        if metric is not None:
            metric = merge(metric, mode="keepone", permute=False)

    else:
        raise NotImplementedError
    
    if inter_merge and frame_idx is not None:
        # 帧间merge，需要记录被merge的两帧的index
        if not binary_frame_mask:
            frame_idx_min = merge(frame_idx, mode="amin", permute=False) # BT,L,M
            frame_idx_max = merge(frame_idx, mode="amax", permute=False)
            frame_idx = torch.cat([frame_idx_min, frame_idx_max], dim=-1)
        else:
            frame_idx = merge(frame_idx, mode="mean", permute=False)
    elif not inter_merge and frame_idx is not None:
        # 帧内merge，只用通过mean的方式更新frame_idx即可
        # pass
        # import pdb; pdb.set_trace()
        # frame_idx = merge(frame_idx, mode="amin", permute=False)
        # frame_idx_min = merge(frame_idx, mode="amin", permute=False) # BT,L,M
        # frame_idx_max = merge(frame_idx, mode="amax", permute=False)
        # frame_idx = torch.stack([frame_idx_min, frame_idx_max], dim=-1) # BT,L,M,2
        # frame_idx = torch.cat([frame_idx[...,0:1,:].min(-1)[0], frame_idx[...,1:,:].max(-1)[0]], dim=-1)

        if not binary_frame_mask:
            frame_idx_min = merge(frame_idx, mode="amin", permute=False) # BT,L,M
            frame_idx_max = merge(frame_idx, mode="amax", permute=False)
            frame_idx = torch.cat([frame_idx_min, frame_idx_max], dim=-1)
        else:
            frame_idx = merge(frame_idx, mode="mean", permute=False)

    # x = merge(x * size, mode="sum")
    # size = merge(size, mode="sum")
    # x = x / size

    if metric is not None:
        return x, frame_idx, size, metric
    else:
        return x, frame_idx, size

def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        t, n, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)
    # print(source.shape)
    source = merge(source, mode="amax", permute=False)
    # print(source.shape)
    return source

bipartite_matching_mapping = {
    "soft_s": bipartite_soft_matching_clip,
    "soft_st": bipartite_soft_matching_st_clip,
    "soft_s_multi_cls": bipartite_soft_matching_clip_multi_cls_v1,
    "soft_f_multi_cls": bipartite_soft_matching_frame_multi_cls,
    "soft_f_multi_cls_v2": bipartite_soft_matching_frame_multi_cls_v2,
    "greedy_f_multi_cls": greedy_matching_frame_multi_cls,
    }