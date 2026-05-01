import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset

CLASS_NAMES = ["normal_play", "knock_on"]


class RugbyClipDataset(Dataset):

    def __init__(self, root_dir, frames_per_clip=16, frame_size=224, augment=False):
        super().__init__()
        self.frames_per_clip = frames_per_clip
        self.frame_size = frame_size
        self.augment = augment
        self.samples = []

        for label, folder_name in enumerate(["no_knock_on", "knock_on"]):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                print(f"WARNING: folder not found - {folder_path}")
                continue
            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    self.samples.append((os.path.join(folder_path, fname), label))

        self._cache = {}

        ko_count = sum(1 for _, l in self.samples if l == 1)
        np_count = sum(1 for _, l in self.samples if l == 0)
        print(f"Loaded {len(self.samples)} clips ({ko_count} knock_on, {np_count} normal_play), augment={augment}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        if idx in self._cache:
            frames = [f.copy() for f in self._cache[idx]]
        else:
            frames = self._load_video(video_path)
            frames = self._sample_frames(frames)

            frames = [
                cv2.cvtColor(cv2.resize(f, (self.frame_size, self.frame_size)), cv2.COLOR_BGR2RGB)
                for f in frames
            ]
            self._cache[idx] = [f.copy() for f in frames]

        if self.augment:
            frames = self._augment(frames)

        # normalise to imagenet
        clip = np.stack(frames).astype(np.float32) / 255.0
        clip = np.transpose(clip, (3, 0, 1, 2))

        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1, 1)
        clip = (clip - mean) / std

        return torch.FloatTensor(clip), label

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"Could not read any frames from {path}")
        return frames

    def _sample_frames(self, frames):
        # uniform sample, pad short clips with last frame
        n = len(frames)
        if n >= self.frames_per_clip:
            indices = np.linspace(0, n - 1, self.frames_per_clip, dtype=int)
        else:
            indices = list(range(n)) + [n - 1] * (self.frames_per_clip - n)
        return [frames[i] for i in indices]

    def _augment(self, frames):
        # flip
        if random.random() > 0.5:
            frames = [cv2.flip(f, 1) for f in frames]

        # brightness/contrast
        alpha = random.uniform(0.7, 1.3)
        beta = random.randint(-30, 30)
        frames = [
            np.clip(f.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            for f in frames
        ]

        # random crop
        if random.random() > 0.5:
            h, w = frames[0].shape[:2]
            crop_ratio = random.uniform(0.75, 1.0)
            new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)
            frames = [
                cv2.resize(f[top:top+new_h, left:left+new_w], (w, h))
                for f in frames
            ]

        # noise
        if random.random() > 0.7:
            sigma = random.uniform(3, 10)
            for i in range(len(frames)):
                noise = np.random.normal(0, sigma, frames[i].shape).astype(np.float32)
                frames[i] = np.clip(frames[i].astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # grayscale
        if random.random() > 0.8:
            frames = [
                cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
                for f in frames
            ]

        # temporal jitter
        if random.random() > 0.7 and len(frames) > 4:
            n_drop = random.randint(1, max(1, len(frames) // 4))
            drop_indices = sorted(random.sample(range(len(frames)), n_drop), reverse=True)
            for di in drop_indices:
                frames.pop(di)
            while len(frames) < self.frames_per_clip:
                frames.insert(random.randint(0, len(frames) - 1),
                              frames[random.randint(0, len(frames) - 1)].copy())

        return frames


if __name__ == "__main__":
    import sys

    root = os.path.join(os.path.dirname(__file__), "knock_on_dataset")
    ds = RugbyClipDataset(root, augment=True)

    if len(ds) == 0:
        print("No clips found — check your dataset folder.")
        sys.exit(1)

    clip, label = ds[0]
    print(f"Clip shape : {clip.shape}")       # torch.Size([3, 16, 224, 224])
    print(f"Label      : {label} ({CLASS_NAMES[label]})")
    print(f"Min / Max  : {clip.min():.2f} / {clip.max():.2f}")
