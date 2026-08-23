from torch.utils import data
import argparse
from pathlib import Path
from tqdm import tqdm

import torchaudio
from textless.data.speech_encoder import SpeechEncoder

from data.dataset import mel_spectrogram
import numpy as np
import os
import traceback

import torch
import builtins

import pyworld as pw
from librosa.util import normalize
from scipy.signal import find_peaks

def get_formants_world(audio, rate=16000, frame_shift=0.02, num_candidates=6):
    """
    More accurate formant estimation using WORLD spectral envelope.
    Returns only F1 and F2 (the most informative and stable formants).
    """
    # --- Preprocess (same as get_f0)
    audio = normalize(audio) * 0.95
    frame_length_ms = 20.0
    to_pad = int(frame_length_ms / 1000 * rate) // 2
    audio = np.pad(audio, (to_pad, to_pad), "constant")
    audio = audio.astype(np.float64)

    # --- WORLD extraction
    f0, time_axis = pw.harvest(audio, rate, frame_period=frame_shift*1000)
    sp = pw.cheaptrick(audio, f0, time_axis, rate)
    num_frames = sp.shape[0]
    num_bins = sp.shape[1]
    freqs = np.linspace(0, rate/2, num_bins)

    # Only return F1 and F2
    F = np.zeros((num_frames, 2), dtype=np.float32)
    for i in range(num_frames):
        spectrum = sp[i, :]
        peaks, _ = find_peaks(spectrum, distance=5)
        if len(peaks) < 2:
            continue

        # 1. Select top-N peaks by amplitude
        sorted_peaks = peaks[np.argsort(spectrum[peaks])[::-1]]
        topN = sorted_peaks[:num_candidates]

        # 2. Sort selected peaks by frequency (lowest = F1, next = F2)
        topN_sorted_by_freq = topN[np.argsort(freqs[topN])]
        if len(topN_sorted_by_freq) >= 2:
            F[i, 0] = freqs[topN_sorted_by_freq[0]]   # F1
            F[i, 1] = freqs[topN_sorted_by_freq[1]]   # F2
    return F


torch.load_orig = torch.load
def torch_load_safe(*args, **kwargs):
    kwargs["weights_only"] = False
    return torch.load_orig(*args, **kwargs)
torch.load = torch_load_safe

parser = argparse.ArgumentParser()
parser.add_argument('--n_clusters', type=int, choices=[50, 100, 200], default=200)
parser.add_argument('--model', type=str, default='hubert-base-ls960')
parser.add_argument('--datadir', type=str, default='/vol/das-nobackup/users/jzheng/Database/datasets/l2arctic_release_v5/l2arctic_release_v5')
parser.add_argument('--outdir', type=str, default='/vol/das-nobackup/users/jzheng/PSST_1/UUVC/features/L2A')
parser.add_argument('--with_pitch_unit', action='store_true')
parser.add_argument('--VCTK', action='store_true')

args = parser.parse_args()

if args.with_pitch_unit:
    encoder = SpeechEncoder.by_name(
        dense_model_name=args.model,
        quantizer_model_name='kmeans',
        vocab_size=args.n_clusters,
        deduplicate=False
#        need_f0=False
    ).cuda()

# processed = [p.stem for p in Path(args.outdir).glob(f'*-unit.npy')]
# print('###we are here?',processed)
# exit()
transforms_16k = {
    22050: torchaudio.transforms.Resample(22050, 16000).cuda(),
    24000: torchaudio.transforms.Resample(24000, 16000).cuda(),
    48000: torchaudio.transforms.Resample(48000, 16000).cuda(),
    44100: torchaudio.transforms.Resample(44100, 16000).cuda()
}
transforms_22k = {
    24000: torchaudio.transforms.Resample(24000, 22050).cuda(),
    16000: torchaudio.transforms.Resample(16000, 22050).cuda(),
    48000: torchaudio.transforms.Resample(48000, 22050).cuda(),
    44100: torchaudio.transforms.Resample(44100, 22050).cuda()
}

wavfiles = [p for p in Path(args.datadir).rglob('*.wav')] + [p for p in Path(args.datadir).rglob('*.flac')]
if args.VCTK:
    wavfiles = [p for p in wavfiles if '_mic1' in str(p)]

for f in tqdm(wavfiles,unit='file'):
#    if f.stem + '-unit' in processed:
#        continue
    wav, sr = torchaudio.load(str(f))
    # print('we are here??',sr) #44100
    # exit() #yes
    wav = wav.cuda()
    wav_16k = wav
    if sr != 16000:
        if sr not in transforms_16k:
            continue
        wav_16k = transforms_16k[sr](wav)
    if sr != 22050:
        if sr not in transforms_22k:
            continue
        wav = transforms_22k[sr](wav)
    # print('we are here??')
    # exit()
#    try:
    mels, energy = mel_spectrogram(wav, 1025, 80, 22050, 256, 1024, 0, 8000, return_energy=True)
    mels, energy = mels.cpu().numpy()[0].T, energy.cpu().numpy()[0]
    if args.with_pitch_unit:
        encoded = encoder(wav_16k)
        units = encoded["units"].cpu().numpy()
        f0 = encoded["f0"].cpu().numpy()
        formants = get_formants_world(wav_16k.cpu().numpy()[0], rate=16000)
#    except:
#        print (f.stem)
#        traceback.print_exc()
#        continue
    
    name = f.stem
    speaker = f.parent.parent.name  
    # print(speaker,name)
    name = name.replace("arctic", speaker)
    # print(new_name)
    # exit()
    
    if args.VCTK:
        name = name.replace('_mic1', '')
    # print('###',os.path.join(args.outdir, name + '-mel.npy'))
    # exit()
    np.save(os.path.join(args.outdir, name + '-mel.npy'), mels)
    np.save(os.path.join(args.outdir, name + '-E.npy'), energy)
    torchaudio.save(os.path.join(args.outdir, name + '.wav'), wav_16k.cpu(), 16000)
    if args.with_pitch_unit:
        np.save(os.path.join(args.outdir, name + '-unit.npy'), units)
        np.save(os.path.join(args.outdir, name + '-f0.npy'), f0)
        np.save(os.path.join(args.outdir, name + '-formants.npy'), formants)
