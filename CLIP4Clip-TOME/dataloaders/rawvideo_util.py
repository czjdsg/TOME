import torch as th
import numpy as np
from PIL import Image
# pytorch=1.7.1
# pip install opencv-python
import cv2
import decord
from dataloaders.rawframe_util import RawFrameExtractor
from dataloaders.frame_sampling import multi_segments_sampling, uniform_sampling
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, CenterCrop
decord.bridge.set_bridge('torch')

class ResizeMaxSize(nn.Module):

    def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fn='max', fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == 'min' else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(img, padding=[pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2], fill=self.fill)
        return img


def _convert_to_rgb(image):
    return image.convert('RGB')


# class CatGen(nn.Module):
#     def __init__(self, num=4):
#         self.num = num
#     def mixgen_batch(image, text):
#         batch_size = image.shape[0]
#         index = np.random.permutation(batch_size)

#         cat_images = []
#         for i in range(batch_size):
#             # image mixup
#             image[i,:] = lam * image[i,:] + (1 - lam) * image[index[i],:]
#             # text concat
#             text[i] = tokenizer((str(text[i]) + " " + str(text[index[i]])))[0]
#         text = torch.stack(text)
#         return image, text


def image_transform(
        image_size: int,
        is_train: bool,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        resize_longest_max: bool = False,
        fill_color: int = 0,
):
    mean = mean or (0.48145466, 0.4578275, 0.40821073)
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or (0.26862954, 0.26130258, 0.27577711)
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    normalize = Normalize(mean=mean, std=std)
    if is_train:
        return Compose([
            RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
            # _convert_to_rgb,
            # ToTensor(),
            normalize,
        ])
    else:
        if resize_longest_max:
            transforms = [
                ResizeMaxSize(image_size, fill=fill_color)
            ]
        else:
            transforms = [
                Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
            ]
        transforms.extend([
            # _convert_to_rgb,
            # ToTensor(),
            normalize,
        ])
        return Compose(transforms)

class RawVideoExtractorCV2():
    def __init__(self, centercrop=False, size=224, framerate=-1, is_train=False):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.transform = self._transform(self.size)

    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def video_to_tensor(self, video_file, preprocess, sample_fp=0, start_time=None, end_time=None):
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time
        assert sample_fp > -1

        # Samples a frame sample_fp X frames.
        cap = cv2.VideoCapture(video_file)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration

        if start_time is not None:
            start_sec, end_sec = start_time, end_time if end_time <= total_duration else total_duration
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

        interval = 1
        if sample_fp > 0:
            interval = fps // sample_fp
        else:
            sample_fp = fps
        if interval == 0: interval = 1

        inds = [ind for ind in np.arange(0, fps, interval)]
        assert len(inds) >= sample_fp
        inds = inds[:sample_fp]

        ret = True
        images, included = [], []

        for sec in np.arange(start_sec, end_sec + 1):
            if not ret: break
            sec_base = int(sec * fps)
            for ind in inds:
                cap.set(cv2.CAP_PROP_POS_FRAMES, sec_base + ind)
                ret, frame = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))

        cap.release()

        if len(images) > 0:
            video_data = th.tensor(np.stack(images))
        else:
            video_data = th.zeros(1)
        return {'video': video_data}

    def get_video_data(self, video_path, start_time=None, end_time=None):
        image_input = self.video_to_tensor(video_path, self.transform, sample_fp=self.framerate, start_time=start_time, end_time=end_time)
        return image_input

    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor

    def process_frame_order(self, raw_video_data, frame_order=0):
        # 0: ordinary order; 1: reverse order; 2: random order.
        if frame_order == 0:
            pass
        elif frame_order == 1:
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data

class ImageNorm(object):
    """Apply Normalization to Image Pixels on GPU
    """
    def __init__(self, mean, std):
        self.mean = th.tensor(mean).view(1, 3, 1, 1)
        self.std = th.tensor(std).view(1, 3, 1, 1)
        
    def __call__(self, img):

        if th.max(img) > 1 and self.mean.max() <= 1:
            img.div_(255.)
        return img.sub_(self.mean).div_(self.std)

class RawVideoExtractorDecord():
    def __init__(self, centercrop=False, size=224, framerate=-1, is_train=False):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.transform = self._transform(size)
        # self.transform = image_transform(self.size, is_train=is_train)

    def _transform(self, n_px):
        
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            # ToTensor(),
            # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def video_to_tensor(self, video_file, preprocess, sample_fp=0, start_time=None, 
                        end_time=None, max_frames=None, slice_type=None, padding=False, pad_base=4):
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time
        assert sample_fp > -1

        # Samples a frame sample_fp X frames.
        vr = decord.VideoReader(video_file, num_threads=1)
        frameCount = len(vr)
        fps = vr.get_avg_fps()

        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration

        if start_time is not None:
            start_sec, end_sec = start_time, end_time if end_time <= total_duration else total_duration

        interval = 1
        if sample_fp > 0:
            interval = fps // sample_fp
        else:
            sample_fp = fps
        if interval == 0: interval = 1

        inds = [ind for ind in np.arange(0, fps, interval)]
        assert len(inds) >= sample_fp
        inds = inds[:sample_fp]
        all_inds = []
        for sec in np.arange(start_sec, end_sec + 1):
            sec_base = int(sec * fps)
            for ind in inds:
                if ind + sec_base < frameCount:
                    all_inds.append(ind + sec_base)
        all_inds = np.array(all_inds)
        # all_inds = np.arange(start_sec, end_sec, interval, dtype=int)

        if max_frames is not None and len(all_inds) < max_frames:
            num_frames = len(all_inds)
            if num_frames % pad_base != 0:
                tgt_num_frames = (num_frames // pad_base + 1) * pad_base
                frame_idx = np.linspace(0, num_frames - 1, tgt_num_frames).astype(int)
                all_inds = all_inds[frame_idx]

        if max_frames is not None and len(all_inds) > max_frames:
            if slice_type == 0:
                all_inds = all_inds[:max_frames, ...]
            elif slice_type == 1:
                all_inds = all_inds[-max_frames:, ...]
            else:
                sample_indx = np.linspace(0, all_inds.shape[0] - 1, num=max_frames, dtype=int)
                all_inds = all_inds[sample_indx, ...]

        images = vr.get_batch(all_inds).permute(0, 3, 1, 2) # B,C,H,W

        if len(images) > 0:
            video_data = preprocess(images.float())
        else:
            video_data = th.zeros(1)
        return {'video': video_data}

    def get_video_data(self, video_path, start_time=None, end_time=None, max_frames=None, slice_type=None, padding=False, pad_base=4):
        image_input = self.video_to_tensor(video_path, self.transform, sample_fp=self.framerate, start_time=start_time, 
                        end_time=end_time, max_frames=max_frames, slice_type=slice_type, padding=padding, pad_base=pad_base)
        return image_input

    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor

    def process_frame_order(self, raw_video_data, frame_order=0):
        # 0: ordinary order; 1: reverse order; 2: random order.
        if frame_order == 0:
            pass
        elif frame_order == 1:
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data

class RawVideoExtractorDecordTSN(RawVideoExtractorDecord):
    def __init__(self, centercrop=False, size=224, framerate=-1, is_train=False):
        super(RawVideoExtractorDecordTSN, self).__init__(centercrop=centercrop, size=size, framerate=framerate)
        self.is_train = is_train
    
    def video_to_tensor(self, video_file, preprocess, sample_fp=0, start_time=None, 
                        end_time=None, max_frames=None, slice_type=None, padding=False, pad_base=4):
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time
        assert sample_fp > -1
        # Samples a frame sample_fp X frames.
        vr = decord.VideoReader(video_file, num_threads=1)
        frameCount = len(vr)
        fps = vr.get_avg_fps()

        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration

        if start_time is not None:
            start_sec, end_sec = start_time, end_time if end_time <= total_duration else total_duration

        inds = [ind for ind in np.arange(0, fps)]
        all_inds = []
        for sec in np.arange(start_sec, end_sec + 1):
            sec_base = int(sec * fps)
            for ind in inds:
                if ind + sec_base < frameCount:
                    all_inds.append(ind + sec_base)
        all_inds = np.array(all_inds)
        num_frames = min(frameCount, len(all_inds))

        max_frames = num_frames if max_frames is None else max_frames
        if self.is_train:
            sampled_inds = multi_segments_sampling(max_frames, num_frames, random_shift=True) # 训练的话，默认进行类似于TSN的random shift采样
        else:
            sampled_inds = uniform_sampling(max_frames, num_frames, twice_sample=False)
        all_inds = all_inds[sampled_inds]

        images = vr.get_batch(all_inds).permute(0, 3, 1, 2) # B,C,H,W
        if len(images) > 0:
            video_data = preprocess(images.float())
        else:
            video_data = th.zeros(1)
        return {'video': video_data}
    
# An ordinary video frame extractor based CV2
RawVideoExtractor = {
    "cv2": RawVideoExtractorCV2,
    "decord": RawVideoExtractorDecord,
    "frame": RawFrameExtractor,
    "decord_tsn": RawVideoExtractorDecordTSN,
    }