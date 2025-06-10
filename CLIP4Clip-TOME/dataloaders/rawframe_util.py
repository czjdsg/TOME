import os
import torch as th
import numpy as np
from PIL import Image
# pytorch=1.7.1
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# pip install opencv-python
import cv2
import decord
decord.bridge.set_bridge('torch')

class RawFrameExtractor():
    def __init__(self, centercrop=False, size=224, framerate=-1, is_train=False,):
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

    def video_to_tensor(self, video_file, preprocess, sample_fp=0, fps=5, start_time=None, end_time=None,
        max_frames=None, slice_type=None):
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time
        assert sample_fp > -1

        # Samples a frame sample_fp X frames.
        frame_names = os.listdir(video_file)
        print(frame_names)

        # filter files that are not images
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
        frame_names = [file for file in frame_names if file.lower().endswith(image_extensions)]
        frame_names.sort()

        frameCount = len(frame_names)

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

        if max_frames is not None and len(all_inds) > max_frames:
            if slice_type == 0:
                all_inds = all_inds[:max_frames, ...]
            elif slice_type == 1:
                all_inds = all_inds[-max_frames:, ...]
            else:
                sample_indx = np.linspace(0, all_inds.shape[0] - 1, num=max_frames, dtype=int)
                all_inds = all_inds[sample_indx, ...]

        images = []
        for ind in all_inds:
            image_path = os.path.join(video_file, frame_names[ind])
            images.append(preprocess(Image.open(image_path).convert("RGB")))

        if len(images) > 0:
            video_data = th.tensor(np.stack(images))
        else:
            video_data = th.zeros(1)
        return {'video': video_data}

    def get_video_data(self, video_path, start_time=None, end_time=None, max_frames=None, slice_type=None,):
        image_input = self.video_to_tensor(video_path, self.transform, sample_fp=self.framerate, start_time=start_time, 
                                        end_time=end_time, max_frames=max_frames, slice_type=slice_type)
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