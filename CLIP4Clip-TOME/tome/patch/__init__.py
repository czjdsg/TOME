# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from .swag import apply_patch as swag
from .timm import apply_patch as timm
from .mae  import apply_patch as mae
from .clip import apply_patch as clip
from .fs_clip import apply_patch as fs_clip
from .frame_mask_clip import apply_patch as frame_mask_clip
from .clip_sta import apply_patch as sta_clip

__all__ = ["timm", "swag", "mae", "clip", "fs_clip", "frame_mask_clip", "sta_clip"]
