import os
from torch.utils import data
import torch
import json
import numpy as np
import soundfile as sf
import random
from pathlib import Path
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
import torchaudio

import torch.nn.functional as F

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression(x, C=1):
    return torch.exp(x) / C

mel_basis = {}
hann_window = {}
def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False, return_energy=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    stft = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    stft = torch.sqrt(stft.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], stft)
    spec = dynamic_range_compression(spec)

    if return_energy:
        energy = torch.norm(stft, dim=1)
        return spec, energy
    return spec

class SpeechDataset(data.Dataset):
    def __init__(self, hp, metadata):
        self.hp = hp
        self.units = self.load_dataset(metadata)
        self.units = [str(x) for x in self.units]
        # ###
        # print(metadata)
        # print(self.units)
        # exit()
        # print('************Attributes (gender, age) infomation is added***************')
        # for x in self.units:
        #     print(x.split())
        #     print(x.split()[3])
        # exit()

        # self.gender = [x.split()[1] for x in self.units]
        # self.accent = [x.split()[2] for x in self.units]
        # self.emotion = [x.split()[3] for x in self.units]
        self.units = [x.split()[0] for x in self.units]
        # self.emotion = [x.split()[3] for x in self.units]

        # for x in self.units:
        #     tmp = x.split()
        #     x = tmp[0]
        #print(self.units[:4])
        #print(self.accent[:4]) # ['American', 'American', 'English', 'English']
        # print(self.gender[:4])
        #exit()
        # for x in self.units:
        #     print('x',x.split())
        #     print(x.split()[0],type(x.split()[0]))
        #     print('x[:?]',x[:65])
        #     exit()
        # ###
        self.data = [x[:-9] + '.wav' for x in self.units]
        # print('*****',self.data[:4])
        self.mels = [x[:-9] + '-mel.npy' for x in self.units]
#        self.energy = [x[:-9] + '-E-normalized.npy' for x in self.units]
        self.energy = [x[:-9] + '-E.npy' for x in self.units]
        self.f0 = [x[:-9] + '-f0.npy' for x in self.units]
        #print('*****',self.f0[:4])
        #exit()
        #Assume 32bit fp PCM, 16000 Hz
        self.lengths = [os.path.getsize(f) / (16000. * 4) for f in self.data]
        bin_size = (self.hp.f0_max - self.hp.f0_min) / self.hp.pitch_bins
        self.f0_bins = torch.arange(self.hp.pitch_bins, dtype=torch.float32) * bin_size + self.hp.f0_var_min
        bin_size = self.hp.E_max / self.hp.E_bins
        self.E_bins = torch.arange(self.hp.E_bins, dtype=torch.float32) * bin_size

        # self.gender2idx = {"M": 0, "F": 1}  
        # self.accent2idx = {
        #     "Arabic": 0,
        #     "Chinese": 1,
        #     "Hindi": 2,
        #     "Korean": 3,
        #     "Spanish": 4,
        #     "Vietnamese": 5,
        #     "English": 6,
        #     "Scottish": 7,
        #     "North_American": 8
        # }
        # self.accent2idx = {
        #     "Arabic": 0,
        #     "Chinese": 0,
        #     "Hindi": 0,
        #     "Korean": 0,
        #     "Spanish": 0,
        #     "Vietnamese": 0,
        #     "Native": 1
        # }
        # self.emotion2idx = {
        #     "neutral": 0,
        #     "happy": 1,
        #     "sad": 2,
        #     "anger": 3
        # }
        # self.emotion2idx = {
        #     "neutral": 0,
        #     "happy": 1,
        #     "sad": 1,
        #     "anger": 1
        # }

        #Print statistics:
        # l = len(self.data)
        #print (f'Total {l} examples, average length {np.mean(self.lengths)} seconds.')

    def load_dataset(self, metadata):
        units = []
        with open(metadata, 'r') as f:
            for line in f.readlines():
                units.append(line.strip())
                # print('units:',units) #['/vol/das-nobackup/users/jzheng/PSST_1/UUVC/features/VCTK/p330_100-unit.npy F American']
                # exit()
        return units

    # def smooth(self, x, win_size):
    #     org = x
    #     x = x.transpose(0, 1) #torch.Size([80, 366])
    #     # print('x.shape:', x.shape)
    #     # exit()
    #     # kernel shape: [out_channels, in_channels/groups, kernel_size]
    #     # kernel = torch.ones((x.shape[0], 1, win_size)) / win_size
    #     n = torch.arange(win_size).float() #.to(x.device)
    #     # hann = 0.5 * (1 - torch.cos(2 * torch.pi * n / (win_size - 1)))
    #     hann = 0.5 * (1 - torch.cos(2 * torch.pi * (n+0.5) / (win_size)))
    #     hann = hann / hann.sum()
    #     kernel = hann.unsqueeze(0).unsqueeze(0).expand(x.shape[0], 1, win_size) #.to(x.device)
        
    #     mel_component = F.conv1d(input=x.unsqueeze(0), weight=kernel, groups=x.shape[0], padding='same')
        
    #     mel_component = mel_component.squeeze(0).transpose(0, 1)
    #     assert mel_component.shape == org.shape
    #     return mel_component
    
    def smooth(self, x, win_size):
        org = x
        x = x.transpose(0, 1)
        T = x.shape[-1]
        N = win_size
        odd = N - 2 * (N // 2)
        
        n = torch.arange(N).float()
        hann = torch.sin(torch.pi * (n + 0.5 * (1 + odd)) / (N + odd)) ** 2
        hann = hann / ((N + 1) // 2) ** 0.5
        kernel = hann.unsqueeze(0).unsqueeze(0).expand(x.shape[0], 1, N)
        
        pad = N // 2
        x_padded = F.pad(x.unsqueeze(0), (pad, pad), mode='constant', value=0)
        mel_component = F.conv1d(x_padded, weight=kernel, groups=x.shape[0], padding=0)

        stride = (N + 1) // 2
        mel_downsample = mel_component[:, :, stride//2::stride]
        
        mel = torch.zeros_like(mel_component) 
        mel[:, :, stride//2::stride] = mel_downsample
        
        mel_padded = F.pad(mel, (pad, pad), mode='constant', value=0)
        mel_component = F.conv1d(mel_padded, weight=kernel, groups=x.shape[0], padding=0)
        
        # mel_component = mel_component.squeeze(0)
        mel_component = mel_component.squeeze(0).transpose(0, 1)
        # assert mel_component.shape == x.shape, f"{mel_component.shape} vs {x.shape}"
        return mel_component
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        audio, sampling_rate = torchaudio.load(self.data[i])
        assert sampling_rate == 16000
        audio = audio[0]
        unit = torch.LongTensor(np.load(self.units[i]))
        dedup_unit, duration = torch.unique_consecutive(unit, return_counts=True)
        mel = torch.FloatTensor(np.load(self.mels[i]))
        # print('mel shape:', mel.shape) #torch.Size([366, 80])
        
        ### Get different time-scale features in the mel-spectrogram
        mel_1 = self.smooth(mel, win_size=129) #low frequency component
        mel_2 = self.smooth(mel, win_size=43) - self.smooth(mel, win_size=129) #mid-low frequency component
        mel_3 = self.smooth(mel, win_size=17) - self.smooth(mel, win_size=43) #mid-high frequency component
        mel_4 = mel - self.smooth(mel, win_size=17) #high frequency component (# fine detail / noise)
        ###
        
        ###
        #print('I want to see the duration info:',duration)
        #print('###SSL Token###',np.load(self.units[i]))
        #print('###ASR Token###',np.load(self.units[i].replace("features", "ASR_features")))
        
        # print('###***###',self.gender[i])
        # exit()
        # gender = torch.FloatTensor([self.gender2idx[self.gender[i]]])   #([ord(self.gender[i])])
        # #print('gender-should be a scalar',gender)
        # accent = torch.FloatTensor([self.accent2idx[self.accent[i]]]) #([ord(self.accent[i][0])+ord(self.accent[i][1])+ord(self.accent[i][2])])
        # #print('accent-should be a scalar',accent)
        # #print('accent',accent)
        # emotion = torch.FloatTensor([self.emotion2idx[self.emotion[i]]])
        # ###
        
        energy, f0 = torch.FloatTensor(np.load(self.energy[i])), torch.FloatTensor(np.load(self.f0[i]))
        voiced = (f0 != 0)
        f0mean = f0[voiced].mean()
        f0[voiced] = f0[voiced] - f0mean
        f0[~voiced] = -1000
        f0 = torch.exp(-(self.f0_bins.repeat(f0.size(0), 1) - f0.unsqueeze(-1)) ** 2 / (2 * self.hp.f0_blur_sigma ** 2))
        energy = torch.exp(-(self.E_bins.repeat(energy.size(0), 1) - energy.unsqueeze(-1)) ** 2 / (2 * self.hp.E_blur_sigma ** 2))
        
        return audio, unit, dedup_unit, duration, mel, energy, f0, voiced.long(), mel_1, mel_2, mel_3, mel_4

    def seqCollate(self, batch):
        output = {
            'audio': [],
            'unit': [],
            'dedup_unit': [],
            'duration': [],
            'mel': [],
            'audio_mask': [],
            'mel_mask': [],
            'unit_mask': [],
            'dedup_unit_mask': [],
            'energy': [],
            'f0': [],
            'voiced': [],
            'mel_1': [],
            'mel_2': [],
            'mel_3': [],
            'mel_4': [],
        }
        #Get the max length of everything
        m_a, m_u, m_m, m_d = 0, 0, 0, 0
        # print('$$$',batch)
        # exit()
        for audio, unit, dedup_unit, duration, mel, _, _, _, _, _, _, _ in batch:
            if len(audio) > m_a:
                m_a = len(audio)
            if len(unit) > m_u:
                m_u = len(unit)
            if len(mel) > m_m:
                m_m = len(mel)
            if len(dedup_unit) > m_d:
                m_d = len(dedup_unit)
        #Pad each element, create mask
        for audio, unit, dedup_unit, duration, mel, E, f0, voiced, mel_1, mel_2, mel_3, mel_4 in batch:
            #Deal with audio
            audio_mask = torch.BoolTensor([False] * len(audio) + [True] * (m_a - len(audio)))
            audio = F.pad(audio, [0, m_a-len(audio)])
            #Deal with units
            unit_mask = torch.BoolTensor([False] * len(unit) + [True] * (m_u - len(unit)))
            unit = F.pad(unit, [0, m_u-len(unit)], value=self.hp.vocab_size)
            #Deal with deduplicated units
            dedup_unit_mask = torch.BoolTensor([False] * len(dedup_unit) + [True] * (m_d - len(dedup_unit)))
            dedup_unit = F.pad(dedup_unit, [0, m_d-len(dedup_unit)], value=self.hp.vocab_size)
            duration = F.pad(duration, [0, m_d-len(duration)], value=-100)
            #Deal with mels
            mel_mask = torch.BoolTensor([False] * len(mel) + [True] * (m_m - len(mel)))
            mel = F.pad(mel, [0, 0, 0, m_m-len(mel)])
            mel_1 = F.pad(mel_1, [0, 0, 0, m_m-len(mel_1)])
            mel_2 = F.pad(mel_2, [0, 0, 0, m_m-len(mel_2)])
            mel_3 = F.pad(mel_3, [0, 0, 0, m_m-len(mel_3)])
            mel_4 = F.pad(mel_4, [0, 0, 0, m_m-len(mel_4)])
            #Energy, pitch
            E = F.pad(E, [0, 0, 0, m_m-len(E)])
            f0 = F.pad(f0, [0, 0, 0, m_u-len(f0)])
            voiced = F.pad(voiced, [0, m_u-len(voiced)])
            #Aggregate
            output['audio'].append(audio)
            output['unit'].append(unit)
            output['dedup_unit'].append(dedup_unit)
            output['duration'].append(duration)
            output['mel'].append(mel)
            output['unit_mask'].append(unit_mask)
            output['dedup_unit_mask'].append(dedup_unit_mask)
            output['mel_mask'].append(mel_mask)
            output['audio_mask'].append(audio_mask)
            output['energy'].append(E)
            output['f0'].append(f0)
            output['voiced'].append(voiced)
            #print(audio.shape,mel.shape) #torch.Size([68043]) torch.Size([366, 80])
            output['mel_1'].append(mel_1)
            output['mel_2'].append(mel_2)
            output['mel_3'].append(mel_3)
            output['mel_4'].append(mel_4)
        for k in output.keys():
            output[k] = torch.stack(output[k])
        return output
