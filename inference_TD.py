import numpy
import torch
import torch.nn as nn
import torchaudio

import argparse
from tqdm import tqdm
import json

from textless.data.speech_encoder import SpeechEncoder
from models.s2rv import Speech2Vector
from models.u2mel import Unit2Mel
from models.vp import F0Predictor, DurationPredictor, EPredictor
from models.utils import LengthRegulator

from pathlib import Path
import random
import pyloudnorm as pyln
from vocoder.vocoder import Vocoder
import os
import soundfile as sf
import shutil

import argparse
import fairseq
from torch_pitch_shift import *

import numpy as np

torch.serialization.add_safe_globals([argparse.Namespace, fairseq.data.dictionary.Dictionary])

meter = pyln.Meter(22050)

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

parser = argparse.ArgumentParser()

parser.add_argument('--metapath', type=str, default='') # modify 1/2
parser.add_argument('--result_dir', type=str, default='') # modify 2/2
parser.add_argument('--ckpt', type=str, default='/vol/tensusers8/jzheng/T8_Exp/PSST_2/UUVC/ckpt/last.ckpt')
parser.add_argument('--config', type=str, default='/vol/tensusers8/jzheng/T8_Exp/PSST_2/UUVC/ckpt/config.json')

args = parser.parse_args()

class Tester(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.RVEncoder = Speech2Vector(hp)
        self.dp = DurationPredictor(hp, hp.dp_hidden_size, hp.dp_dropout, hp.dp_layers)
        self.f0p = F0Predictor(hp, hp.f0_hidden_size, hp.f0_dropout, hp.f0_layers, n_outputs=hp.pitch_bins)
        self.vp = F0Predictor(hp, hp.voiced_hidden_size, hp.voiced_dropout, hp.voiced_layers)
        self.Ep = EPredictor(hp, hp.E_hidden_size, hp.E_dropout, hp.E_layers, hp.E_bins)
        self.u2m = Unit2Mel(hp)
        self.embedding = nn.Embedding(hp.vocab_size+1, hp.hidden_size, padding_idx=hp.vocab_size)
        bin_size = (hp.f0_max - hp.f0_min) / hp.pitch_bins
        self.f0_bins = torch.arange(hp.pitch_bins, dtype=torch.float32) * bin_size + hp.f0_var_min
        self.duration_length = LengthRegulator()
        self.vocoder = Vocoder(hp.vocoder_config_path, hp.vocoder_ckpt_path)
        self.vocoder.eval()
        self.vocoder.generator.remove_weight_norm()

    def encode(self, audio):
        PVQ, spk, L, _ = self.RVEncoder(audio)
        return {
            'a_p': PVQ,
            'a_s': spk,
            'a_r': L
        }

    def forward(self, U, PVQ, spkr, L, direct_UL=False):
        U = self.embedding(U)
        if not direct_UL:
            L = self.dp(L, U)
            L = torch.round(torch.exp(L) - 1)
            L = torch.clamp(L, min=1.0) #At least one frame is allocated(?)
            UL, _ = self.duration_length(U, L, None)
        else:
            UL = U
        f0_preds, f0_feats = self.f0p(PVQ, UL)
        voiced_preds, _ = self.vp(PVQ, UL)
        V = (torch.sigmoid(voiced_preds) > 0.5)
        f0 = torch.sigmoid(f0_preds)
        #Energy prediction
        E = torch.sigmoid(self.Ep(f0_feats))
        #MelSpec Prediction
        pred_mel_1, pred_mel_2, pred_mel_3, pred_mel_4 = self.u2m(E, f0, V, spkr, UL, None)
        pred_mels = pred_mel_1 + pred_mel_2 + pred_mel_3 + pred_mel_4
        #Vocoder, LoudNorm
        wav = self.vocoder(pred_mels).squeeze(0).squeeze(0).detach().cpu().numpy()
        # loudness = meter.integrated_loudness(wav)
        # wav = pyln.normalize.loudness(wav, loudness, -12.0)
        return wav


class Split_Tester(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.RVEncoder = Speech2Vector(hp)
        self.dp = DurationPredictor(hp, hp.dp_hidden_size, hp.dp_dropout, hp.dp_layers)
        self.f0p = F0Predictor(hp, hp.f0_hidden_size, hp.f0_dropout, hp.f0_layers, n_outputs=hp.pitch_bins)
        self.vp = F0Predictor(hp, hp.voiced_hidden_size, hp.voiced_dropout, hp.voiced_layers)
        self.Ep = EPredictor(hp, hp.E_hidden_size, hp.E_dropout, hp.E_layers, hp.E_bins)
        self.u2m = Unit2Mel(hp)
        self.embedding = nn.Embedding(hp.vocab_size+1, hp.hidden_size, padding_idx=hp.vocab_size)
        bin_size = (hp.f0_max - hp.f0_min) / hp.pitch_bins
        self.f0_bins = torch.arange(hp.pitch_bins, dtype=torch.float32) * bin_size + hp.f0_var_min
        self.duration_length = LengthRegulator()
        self.vocoder = Vocoder(hp.vocoder_config_path, hp.vocoder_ckpt_path)
        self.vocoder.eval()
        self.vocoder.generator.remove_weight_norm()

    def encode(self, audio):
        PVQ, spk, L, _ = self.RVEncoder(audio)
        return {
            'a_p': PVQ,
            'a_s': spk,
            'a_r': L
        }

    def forward(self, U, PVQ, spkr_src, spkr_target, L, direct_UL=False, alpha=0.75):
        U = self.embedding(U)
        if not direct_UL:
            L = self.dp(L, U)
            L = torch.round(torch.exp(L) - 1)
            L = torch.clamp(L, min=1.0) #At least one frame is allocated(?)
            UL, _ = self.duration_length(U, L, None)
        else:
            UL = U
        f0_preds, f0_feats = self.f0p(PVQ, UL)
        voiced_preds, _ = self.vp(PVQ, UL)
        V = (torch.sigmoid(voiced_preds) > 0.5)
        f0 = torch.sigmoid(f0_preds)
        #Energy prediction
        E = torch.sigmoid(self.Ep(f0_feats))

        split_index = int(alpha*256)
        
        spkr = torch.cat([spkr_target[:, :256-split_index], spkr_target[:, 256-split_index:]],dim=-1)
    
        pred_mel_1, pred_mel_2, pred_mel_3, pred_mel_4 = self.u2m(E, f0, V, spkr, UL, None)
        pred_mels = pred_mel_1 + pred_mel_2 + pred_mel_3 + pred_mel_4
        #Vocoder, LoudNorm
        wav = self.vocoder(pred_mels).squeeze(0).squeeze(0).detach().cpu().numpy()
        return wav



with open(args.config, 'r') as f:
    hp = Namespace()
    hp.__dict__ = json.load(f)

model = Tester(hp)
model.load_state_dict(torch.load(args.ckpt)['state_dict'], strict=False)
model.cuda()
model.eval()


split_model = Split_Tester(hp)
split_model.load_state_dict(torch.load(args.ckpt)['state_dict'], strict=False)
split_model.cuda()
split_model.eval()

encoder = SpeechEncoder.by_name(
    dense_model_name='hubert-base-ls960',
    quantizer_model_name='kmeans',
    vocab_size=hp.vocab_size,
    deduplicate=False,
    need_f0=False
).cuda()

if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)
    os.makedirs(args.result_dir + '/source', exist_ok=True)
    os.makedirs(args.result_dir + '/reference', exist_ok=True)

with open(args.metapath, 'r') as f:
    lines = f.readlines()

folders = ['rhythm', 'pitch-energy', 'prosody', 'speaker', 'imitation', 'source', 'reference', 'speaker_split_0.0', 'speaker_split_0.25', 'speaker_split_0.5', 'speaker_split_0.5_x', 'speaker_split_0.75', 'speaker_split_1.0']
for folder in folders:
    (Path(args.result_dir) / Path(folder)).mkdir(parents=True, exist_ok=True)

def load(src_wav):
    src_wav, sr = torchaudio.load(src_wav)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        src_wav = resampler(src_wav)
        sr = 16000
    assert sr == 16000, "The source and target speech are not in 16k, adjust the code here to resample"
    src_wav = src_wav.cuda()
    if src_wav.size(0) != 1:
        src_wav = src_wav.mean(0)
        src_wav = src_wav.unsqueeze(0) ###
    return src_wav #1, T


for line in tqdm(lines):
    src_wav_n, tgt_wav_n = line.split()
    shutil.copy(src_wav_n, args.result_dir+'/source')
    shutil.copy(tgt_wav_n, args.result_dir+'/reference')
    
    out_n = Path(src_wav_n).stem + '--' + Path(tgt_wav_n).stem + '.wav'
    src_wav = load(src_wav_n)
    tgt_wav = load(tgt_wav_n)
    
    transfer = dict()
    with torch.no_grad():
        src_attributes = model.encode(src_wav)
        tgt_attributes = model.encode(tgt_wav)
        UL = encoder(src_wav)['units']
        U, L = torch.unique_consecutive(UL, return_counts=True)
        U, L, UL = U.unsqueeze(0), L.unsqueeze(0), UL.unsqueeze(0)

        transfer['speaker_split_0.0'] = split_model(U=UL, PVQ=src_attributes['a_p'], spkr_src=src_attributes['a_s'], spkr_target=tgt_attributes['a_s'], L=None, direct_UL=True, alpha=0.0)
        transfer['speaker_split_0.25'] = split_model(U=UL, PVQ=src_attributes['a_p'], spkr_src=src_attributes['a_s'], spkr_target=tgt_attributes['a_s'], L=None, direct_UL=True, alpha=0.25)
        transfer['speaker_split_0.5'] = split_model(U=UL, PVQ=src_attributes['a_p'], spkr_src=src_attributes['a_s'], spkr_target=tgt_attributes['a_s'], L=None, direct_UL=True, alpha=0.50)
        transfer['speaker_split_0.75'] = split_model(U=UL, PVQ=src_attributes['a_p'], spkr_src=src_attributes['a_s'], spkr_target=tgt_attributes['a_s'], L=None, direct_UL=True, alpha=0.75)
        transfer['speaker_split_1.0'] = split_model(U=UL, PVQ=src_attributes['a_p'], spkr_src=src_attributes['a_s'], spkr_target=tgt_attributes['a_s'], L=None, direct_UL=True, alpha=1.0)
    for k in transfer.keys():
        sf.write(os.path.join(args.result_dir, k, out_n), transfer[k], 22050)
