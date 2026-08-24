# import torch
# import torch.nn.functional as F
# import librosa
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from data.dataset import mel_spectrogram
# import torchaudio
# from pathlib import Path

# class Config:
#     sr = 22050
#     n_fft = 1024
#     hop_length = 256
#     n_mels = 80

# cfg = Config()


# def smooth(x, win_size):
#     org = x # B, d, T
#     # print('111:',x.shape) 
#     # exit()
#     x = x.squeeze(0)
#     T = x.shape[-1]  # ← 记录原始长度
    
#     N = win_size
#     odd = N - 2 * (N // 2)
    
#     n = torch.arange(N).float()
#     hann = torch.sin(torch.pi * (n + 0.5 * (1 + odd)) / (N + odd)) ** 2
#     hann = hann / ((N + 1) // 2) ** 0.5
#     kernel = hann.unsqueeze(0).unsqueeze(0).expand(x.shape[0], 1, N)
    
#     pad = N // 2
    
#     x_padded = F.pad(x.unsqueeze(0), (pad, pad), mode='constant', value=0)
#     mel_component = F.conv1d(x_padded, weight=kernel, groups=x.shape[0], padding=0)
#     # print(mel_component.shape, 'T:', T)
#     # mel_component = mel_component[:, :, :T]  # ← 裁剪到原始长度
    
#     stride = (N + 1) // 2
#     mel_downsample = mel_component[:, :, stride//2::stride]
    
#     mel = torch.zeros_like(mel_component)  # 现在长度就是T了
#     mel[:, :, stride//2::stride] = mel_downsample
    
#     mel_padded = F.pad(mel, (pad, pad), mode='constant', value=0)
#     mel_component = F.conv1d(mel_padded, weight=kernel, groups=x.shape[0], padding=0)
#     # mel_component = mel_component[:, :, :T]  # ← 同样裁剪
    
#     mel_component = mel_component.squeeze(0) # d, T
#     # print('mel_component shape:', mel_component.shape, 'x shape:', x.shape)
#     # exit()
#     assert mel_component.shape == x.shape, f"{mel_component.shape} vs {x.shape}"
#     return mel_component
    


# wav_paths = [
#     "/vol/tensusers8/jzheng/T8_Exp/PSST_2/UUVC/result_demo/speaker_split_1.0/03_src_e--03_ref_e.wav",
#     "/vol/tensusers8/jzheng/T8_Exp/PSST_2/UUVC/result_demo/speaker_split_0.25/03_src_e--03_ref_e.wav"
# ]

# # win_sizes = [129, 13, 5]  # low / mid-low / mid-high
# labels = ['Mel1', 'Mel2', 'Mel3', 'Mel4']

# output_dir = "/vol/tensusers8/jzheng/T8_Exp/PSST_2/UUVC/result_demo/decompose_figs"
# os.makedirs(output_dir, exist_ok=True)

# # =========================
# # Process each sample
# # =========================
# for idx, path in enumerate(wav_paths):
#     p = Path(path)
#     wav, sr = torchaudio.load(str(path))
#     if sr != 22050:
#         exit(f"Sample {idx+1}: Unsupported sample rate {sr}. Please resample to 22050 Hz.")
#     mel, energy = mel_spectrogram(wav, 1025, 80, 22050, 256, 1024, 0, 8000, return_energy=True)

#     ### Get different time-scale features in the mel-spectrogram
#     mel_1 = smooth(mel, win_size=129) #low frequency component
#     mel_2 = smooth(mel, win_size=43) - smooth(mel, win_size=129) #mid-low frequency component
#     mel_3 = smooth(mel, win_size=17) - smooth(mel, win_size=43) #mid-high frequency component
#     mel_4 = mel.squeeze(0) - smooth(mel, win_size=17) #high frequency component (# fine detail / noise)
#     ###
    
#     ### original indices: window sizes=129 (1500ms), 13 (150ms), 5 (50ms), residual
#     # new indices: window sizes= 129 (1500ms), 43 (500ms), 17 (200ms), residual
#     components = [mel_1, mel_2, mel_3, mel_4]
    
#     # # ===== UNIFY THE COLOR SCALE =====
#     # all_vals = torch.cat([c.flatten() for c in components])
#     # vmin = all_vals.min().item()
#     # vmax = all_vals.max().item()
#     # print(f"Sample {idx+1}: Color scale range: [{vmin:.2f}, {vmax:.2f}]")
#     # exit()
#     plt.figure(figsize=(12, 3*4))
#     for i, comp in enumerate(components):
#         plt.subplot(4, 1, i+1)
#         # plt.imshow(comp.numpy(), aspect='auto', origin='lower', cmap='magma')
#         if i == 0:
#             plt.imshow(comp.numpy(), aspect='auto', origin='lower',
#             cmap='magma', vmin=-10, vmax=4)
#         elif i == 1:
#             plt.imshow(comp.numpy(), aspect='auto', origin='lower',
#             cmap='magma', vmin=-7, vmax=4)
#         elif i == 2:
#             plt.imshow(comp.numpy(), aspect='auto', origin='lower',
#             cmap='magma', vmin=-12, vmax=8)
#         else:
#             plt.imshow(comp.numpy(), aspect='auto', origin='lower',
#             cmap='magma', vmin=-5, vmax=4)
#         plt.colorbar(format='%+2.0f dB')
#         plt.ylabel("Frequency bin")
#         plt.title(f"Sample {idx+1} - {labels[i]}")

#     plt.xlabel("Time frames")
#     save_path = os.path.join(output_dir, f"{p.parent.name}.png")
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close()
#     print(f"Saved decomposition figure: {save_path}")


#####################################################################################

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import torchaudio

from pathlib import Path
from data.dataset import mel_spectrogram


# =====================================================
# Config
# =====================================================

class Config:
    sr = 22050
    n_fft = 1024
    hop_length = 256
    n_mels = 80

cfg = Config()


# =====================================================
# User Setting
# =====================================================

# plot_duration_sec = 0 # where is your breakfast

wav_paths = [
    "/vol/tensusers8/jzheng/T8_Exp/PSST_2/UUVC/result_demo/speaker_split_1.0/03_src_e--03_ref_e.wav",
    "/vol/tensusers8/jzheng/T8_Exp/PSST_2/UUVC/result_demo/speaker_split_0.75/03_src_e--03_ref_e.wav",
    "/vol/tensusers8/jzheng/T8_Exp/PSST_2/UUVC/result_demo/speaker_split_0.5/03_src_e--03_ref_e.wav",
    "/vol/tensusers8/jzheng/T8_Exp/PSST_2/UUVC/result_demo/speaker_split_0.25/03_src_e--03_ref_e.wav",
    "/vol/tensusers8/jzheng/T8_Exp/PSST_2/UUVC/result_demo/speaker_split_0.0/03_src_e--03_ref_e.wav",
]

labels = ["Low", "Mid-Low", "Mid-High", "Residual"]

output_dir = "/vol/tensusers8/jzheng/T8_Exp/PSST_2/UUVC/result_demo/decompose_figs"
os.makedirs(output_dir, exist_ok=True)


# =====================================================
# Smoothing (unchanged)
# =====================================================

def smooth(x, win_size):
    x = x.squeeze(0)
    N = win_size
    odd = N - 2 * (N // 2)
    n = torch.arange(N).float()
    hann = torch.sin(torch.pi * (n + 0.5 * (1 + odd)) / (N + odd)) ** 2

    hann = hann / ((N + 1) // 2) ** 0.5

    kernel = (
        hann.unsqueeze(0)
        .unsqueeze(0)
        .expand(x.shape[0], 1, N)
    )

    pad = N // 2

    x_padded = F.pad(
        x.unsqueeze(0),
        (pad, pad),
        mode="constant",
        value=0,
    )

    mel_component = F.conv1d(
        x_padded,
        weight=kernel,
        groups=x.shape[0],
        padding=0,
    )

    stride = (N + 1) // 2

    mel_downsample = mel_component[:, :, stride//2::stride]

    mel = torch.zeros_like(mel_component)
    mel[:, :, stride // 2 :: stride] = mel_downsample

    mel_padded = F.pad(mel, (pad, pad), mode="constant", value=0)

    mel_component = F.conv1d(
        mel_padded,
        weight=kernel,
        groups=x.shape[0],
        padding=0,
    )

    return mel_component.squeeze(0)


# =====================================================
# Figure
# =====================================================

num_samples = len(wav_paths)
num_components = 4

fig, axes = plt.subplots(
    num_samples,
    num_components,
    figsize=(4 * num_components, 3 * num_samples),
    sharex=True,
    sharey=True,
)

if num_samples == 1:
    axes = np.expand_dims(axes, axis=0)


# =====================================================
# Store data first (important)
# =====================================================

all_data = []

for idx, path in enumerate(wav_paths):

    wav, sr = torchaudio.load(path)

    if sr != cfg.sr:
        raise ValueError(f"{path} has wrong SR")

    mel, _ = mel_spectrogram(
        wav,
        1025,
        80,
        22050,
        256,
        1024,
        0,
        8000,
        return_energy=True,
    )

    duration = wav.shape[-1]/sr #cfg.hop_length / cfg.sr

    mel_1 = smooth(mel, 129)
    mel_2 = smooth(mel, 43) - smooth(mel, 129)
    mel_3 = smooth(mel, 17) - smooth(mel, 43)
    mel_4 = mel.squeeze(0) - smooth(mel, 17)

    comps = [mel_1, mel_2, mel_3, mel_4]

    all_data.append((comps, duration))


# =====================================================
# GLOBAL COLOR SCALE (IMPORTANT)
# =====================================================

all_vals = torch.cat(
    [c.flatten() for comps, _ in all_data for c in comps]
).numpy()

vmin = np.percentile(all_vals, 1)
vmax = np.percentile(all_vals, 99)

print(f"Global scale: [{vmin:.2f}, {vmax:.2f}]")


# =====================================================
# Plot
# =====================================================
label_y = [
    r"$\alpha = 1.0$",
    r"$\alpha = 0.75$",
    r"$\alpha = 0.5$",
    r"$\alpha = 0.25$",
    r"$\alpha = 0.0$",
]
for i, (comps, duration) in enumerate(all_data):

    for j, comp in enumerate(comps):

        ax = axes[i, j]

        extent = [0, duration, 0, comp.shape[0]]

        im = ax.imshow(
            comp.numpy(),
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )

        if i == 0:
            ax.set_title(labels[j], fontsize=12)

        if j == 0:
            ax.set_ylabel(label_y[i])

        if i == num_samples - 1:
            ax.set_xlabel("Time (s)")
        
        if j == 0 and i < 4 or j == 1 and i < 3 or j == 2 and i < 2 or j == 3 and i < 1:
            for spine in ax.spines.values():
                spine.set_edgecolor('#00FFB2')   
                spine.set_linewidth(5)

# =====================================================
# FIXED COLORBAR (NO LONGER INSIDE SUBPLOTS)
# =====================================================

fig.subplots_adjust(
    right=0.88,
    wspace=0.05,
    hspace=0.10
)

# one global colorbar (cleanest)
cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.70])

fig.colorbar(
    im,
    cax=cbar_ax
)

cbar_ax.set_ylabel("Magnitude", rotation=270, labelpad=15)


# =====================================================
# Save
# =====================================================

save_path = os.path.join(
    output_dir,
    f"decomposition_s_cleanCB_c.png"
)

plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", save_path)