import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset


# ── Class label mapping ────────────────────────────────────────────
CLASS_NAMES = ["normal_play", "knock_on"]
#                  0               1


class RugbyClipDataset(Dataset):
    """
    Parameters
    ----------
    root_dir : str
        Path to `knock_on_dataset/` (must contain `knock_on/` and
        `no_knock_on/` subfolders).
    frames_per_clip : int
        Number of frames to sample from each video (default 16).
    frame_size : int
        Spatial resolution to resize each frame to (default 224).
    augment : bool
        Whether to apply training-time augmentations.
    """

    def __init__(self, root_dir, frames_per_clip=16, frame_size=224, augment=False):
        super().__init__()
        self.frames_per_clip = frames_per_clip
        self.frame_size = frame_size
        self.augment = augment

        self.samples = []  # list of (video_path, label)

        # Scan both class folders
        for label, folder_name in enumerate(["no_knock_on", "knock_on"]):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                print(f"[WARNING] Folder not found: {folder_path}")
                continue
            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    self.samples.append((os.path.join(folder_path, fname), label))

        self._cache = {}  # RAM cache: idx -> list of resized RGB frames

        print(f"[Dataset] {len(self.samples)} clips  "
              f"({sum(1 for _, l in self.samples if l == 1)} knock_on, "
              f"{sum(1 for _, l in self.samples if l == 0)} normal_play)  "
              f"augment={augment}")

    # ───────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.samples)

    # ───────────────────────────────────────────────────────────────
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        # Return cached frames (already sampled + resized + RGB) if available
        if idx in self._cache:
            frames = [f.copy() for f in self._cache[idx]]
        else:
            # 1. Read all frames from the video
            frames = self._load_video(video_path)

            # 2. Uniformly sample exactly `frames_per_clip` frames
            frames = self._sample_frames(frames)

            # 3. Resize to (frame_size x frame_size) and BGR → RGB
            frames = [
                cv2.cvtColor(cv2.resize(f, (self.frame_size, self.frame_size)), cv2.COLOR_BGR2RGB)
                for f in frames
            ]

            # Store in cache
            self._cache[idx] = [f.copy() for f in frames]

        # 4. Apply augmentations (training only)
        if self.augment:
            frames = self._augment(frames)

        # 5. Convert to float tensor and normalise with ImageNet stats
        clip = np.stack(frames).astype(np.float32) / 255.0    # (T, H, W, C)
        clip = np.transpose(clip, (3, 0, 1, 2))                # (C, T, H, W)

        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1, 1)
        std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1, 1)
        clip = (clip - mean) / std

        return torch.FloatTensor(clip), label

    # ── helpers ────────────────────────────────────────────────────
    def _load_video(self, path):
        """Read every frame from a video file."""
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
        """
        Uniformly sample exactly `frames_per_clip` frames.
        If the video is shorter, duplicate the last frame.
        """
        n = len(frames)
        if n >= self.frames_per_clip:
            # Pick evenly-spaced indices
            indices = np.linspace(0, n - 1, self.frames_per_clip, dtype=int)
        else:
            # Pad by repeating last frame
            indices = list(range(n)) + [n - 1] * (self.frames_per_clip - n)
        return [frames[i] for i in indices]

    def _augment(self, frames):
        """
        Apply the SAME random transform to every frame in the clip
        so temporal consistency is preserved.
        """
        # --- Random horizontal flip (50 %) ---
        if random.random() > 0.5:
            frames = [cv2.flip(f, 1) for f in frames]

        # --- Random brightness / contrast ---
        alpha = random.uniform(0.7, 1.3)   # contrast (wider range)
        beta  = random.randint(-30, 30)     # brightness (wider range)
        frames = [
            np.clip(f.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            for f in frames
        ]

        # --- Random crop + resize (simulates zoom / scale variation) ---
        if random.random() > 0.5:
            h, w = frames[0].shape[:2]
            crop_ratio = random.uniform(0.75, 1.0)
            new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
            top  = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)
            frames = [
                cv2.resize(f[top:top+new_h, left:left+new_w], (w, h))
                for f in frames
            ]

        # --- Gaussian noise (30 %) ---
        if random.random() > 0.7:
            sigma = random.uniform(3, 10)
            for i in range(len(frames)):
                noise = np.random.normal(0, sigma, frames[i].shape).astype(np.float32)
                frames[i] = np.clip(frames[i].astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # --- Random grayscale conversion (20 %) ---
        if random.random() > 0.8:
            frames = [
                cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
                for f in frames
            ]

        # --- Temporal jitter: randomly drop & duplicate frames (30 %) ---
        if random.random() > 0.7 and len(frames) > 4:
            n_drop = random.randint(1, max(1, len(frames) // 4))
            drop_indices = sorted(random.sample(range(len(frames)), n_drop), reverse=True)
            for di in drop_indices:
                frames.pop(di)
            # Duplicate random frames to restore original count
            while len(frames) < self.frames_per_clip:
                frames.insert(random.randint(0, len(frames) - 1),
                              frames[random.randint(0, len(frames) - 1)].copy())

        return frames


# ═══════════════════════════════════════════════════════════════════
#  Quick self-test
# ═══════════════════════════════════════════════════════════════════
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
