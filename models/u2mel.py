import torch.nn as nn
import torch
import torch.nn.functional as F
from .utils import ConvLNBlock, ResBlock

class Unit2Mel(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.source = Split_Generator_T(hp)
        self.filter = Split_Generator_T(hp)
        self.energy = EGenerator(hp)
        self.f0_embedding = nn.Embedding(hp.pitch_bins, hp.hidden_size)
        self.unvoiced_embedding = nn.Parameter(torch.randn(hp.hidden_size) * 0.02)
        self.spkr_unit_linear = nn.Linear(hp.hidden_size + hp.feature_size, hp.hidden_size)
        self.spkr_f0_linear = nn.Linear(hp.hidden_size + hp.feature_size, hp.hidden_size)
        self.E_embedding = nn.Embedding(hp.E_bins, hp.hidden_size)

    def encode(self, embedding, bins):
        #bins: N, T, b or N, b
        bins = bins / (bins.sum(-1, keepdim=True) + 1e-5)
        return torch.matmul(bins, embedding.weight)

    def forward(self, E, f0, voicing, spkr, x, mask, mel_length=None, spkr_r=None):
        #x: N, T, C
        #rv, spkr: N, C
        t = x.size(1)
        spkr = spkr.unsqueeze(1).expand(-1, t, -1)
        # print('1:',spkr.shape) #[16, 424, 256]
        f0 = self.encode(self.f0_embedding, f0)
        f0[~voicing] = self.unvoiced_embedding.to(f0.dtype)
        E = self.encode(self.E_embedding, E)
        ## x = self.spkr_unit_linear(torch.cat([x, spkr], 2))
        f1, f2, f3, f4 = self.filter(x, spkr, mel_length)  # swap
        # print('1:',f.shape)
        if spkr_r is not None:
            spkr_r = spkr_r.unsqueeze(1).expand(-1, t, -1)
            s1, s2, s3, s4 = self.source(f0, spkr_r, mel_length)
            # print('2:',s.shape)
            ## f0 = self.spkr_f0_linear(torch.cat([f0, spkr_r], 2))
        else:
            s1, s2, s3, s4 = self.source(f0, spkr, mel_length)

        e = self.energy(E)
        ret1, ret2, ret3, ret4 = s1 + f1, s2 + f2, s3 + f3, s4 + f4
        ret1, ret2, ret3, ret4 = ret1 + e, ret2 + e, ret3 + e, ret4 + e
        if mask is not None:
            ret1[mask], ret2[mask], ret3[mask], ret4[mask] = 0, 0, 0, 0
        return ret1, ret2, ret3, ret4
    

class Split_Generator_T(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.layers1_1 = nn.ModuleList([ConvLNBlock(hp.hidden_size + hp.feature_size//4, hp.dropout, dilation=(2*i+1)) for i in range(hp.mel_layers)])
        self.layers1_2 = nn.ModuleList([ConvLNBlock(hp.hidden_size + hp.feature_size//4, hp.dropout, dilation=(2*i+1)) for i in range(hp.mel_layers)])
        self.layers1_3 = nn.ModuleList([ConvLNBlock(hp.hidden_size + hp.feature_size//4, hp.dropout, dilation=(2*i+1)) for i in range(hp.mel_layers)])
        self.layers1_4 = nn.ModuleList([ConvLNBlock(hp.hidden_size + hp.feature_size//4, hp.dropout, dilation=(2*i+1)) for i in range(hp.mel_layers)])
        self.layers1 = [self.layers1_1, self.layers1_2, self.layers1_3, self.layers1_4]
        
        self.layers2_1 = nn.ModuleList([ConvLNBlock(hp.hidden_size + hp.feature_size//4, hp.dropout, dilation=(2*i+1)) for i in range(hp.mel_layers)])
        self.layers2_2 = nn.ModuleList([ConvLNBlock(hp.hidden_size + hp.feature_size//4, hp.dropout, dilation=(2*i+1)) for i in range(hp.mel_layers)])
        self.layers2_3 = nn.ModuleList([ConvLNBlock(hp.hidden_size + hp.feature_size//4, hp.dropout, dilation=(2*i+1)) for i in range(hp.mel_layers)])
        self.layers2_4 = nn.ModuleList([ConvLNBlock(hp.hidden_size + hp.feature_size//4, hp.dropout, dilation=(2*i+1)) for i in range(hp.mel_layers)])
        self.layers2 = [self.layers2_1, self.layers2_2, self.layers2_3, self.layers2_4]
        
        self.linear_1 = nn.Linear(hp.hidden_size + hp.feature_size//4, hp.n_mels)
        self.linear_2 = nn.Linear(hp.hidden_size + hp.feature_size//4, hp.n_mels)
        self.linear_3 = nn.Linear(hp.hidden_size + hp.feature_size//4, hp.n_mels)
        self.linear_4 = nn.Linear(hp.hidden_size + hp.feature_size//4, hp.n_mels)
        self.linears = [self.linear_1, self.linear_2, self.linear_3, self.linear_4]
        
        self.hp = hp

    def forward(self, x, spk, mel_length=None):
        #x: N, T, C
        spk_list = list(torch.chunk(spk, chunks=4, dim=-1)) #spk1, spk2, spk3, spk4
        t = x.size(1)
        if mel_length is None: #Use approximation during inference
            mel_length = int(t * self.hp.scale_factor)
        x_unit = x
        sub_mels = [] 
        
        for i, spk in enumerate(spk_list):
            # t = x.size(1)
            # print('1:',spk.shape)
            x = torch.cat([x_unit,spk],2)
            x = x.transpose(1, 2)
            for layer in self.layers1[i]:
                x = layer(x)
            x = F.interpolate(x, size=mel_length)
            for layer in self.layers2[i]:
                x = layer(x)
            x = x.transpose(1, 2) # N, T, C_i
            x = self.linears[i](x).squeeze(-1) # N, T, C_i
            sub_mels.append(x)
        
        #x = sub_mels[0] + sub_mels[1] + sub_mels[2] + sub_mels[3]
        
        return sub_mels[0], sub_mels[1], sub_mels[2], sub_mels[3]


class Generator(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.layers1 = nn.ModuleList([ConvLNBlock(hp.hidden_size, hp.dropout, dilation=(2*i+1)) for i in range(hp.mel_layers)])
        self.layers2 = nn.ModuleList([ConvLNBlock(hp.hidden_size, hp.dropout, dilation=(2*i+1)) for i in range(hp.mel_layers)])
        self.linear = nn.Linear(hp.hidden_size, hp.n_mels)
        self.hp = hp

    def forward(self, x, mel_length=None):
        #x: N, T, C
        t = x.size(1)
        x = x.transpose(1, 2)
        for layer in self.layers1:
            x = layer(x)
        if mel_length is None: #Use approximation during inference
            mel_length = int(t * self.hp.scale_factor)
        x = F.interpolate(x, size=mel_length)
        for layer in self.layers2:
            x = layer(x)
        x = x.transpose(1, 2) #N, T, C
        x = self.linear(x).squeeze(-1) #N, T, C
        return x

class Discriminator(nn.Module):
    def __init__(self, hp):
        super(Discriminator, self).__init__()
        self.hp = hp
        c_in = hp.n_mels
        c_mid = 256
        c_out = hp.hidden_size

        self.phi = nn.Sequential(
            nn.Conv1d(c_in, c_mid, kernel_size=3, stride=1, padding=1, dilation=1),
            ResBlock(c_mid, c_mid, c_mid),
            ResBlock(c_mid, c_mid, c_mid),
            ResBlock(c_mid, c_mid, c_mid),
            ResBlock(c_mid, c_mid, c_mid),
            ResBlock(c_mid, c_mid, c_mid),
#            ResBlock(c_mid, c_mid, c_mid),
#            ResBlock(c_mid, c_mid, c_mid),
#            ResBlock(c_mid, c_mid, c_mid),
        )
#        self.res = ResBlock(c_mid, c_mid, c_out)

        self.psi = nn.Conv1d(c_mid, 1, kernel_size=3, stride=1, padding=1, dilation=1)

#        self.match = nn.Sequential(
#            nn.Linear(hp.hidden_size, c_mid),
#            nn.ReLU(),
#            nn.Linear(c_mid, c_mid)
#        )

    def forward(self, mel):
        """
        Args:
            mel: mel spectrogram, torch.Tensor of shape (B x C x T)
            positive: positive speaker embedding, torch.Tensor of shape (B x d)
            negative: negative speaker embedding, torch.Tensor of shape (B x d)
        Returns:
Nsi
        """
        pred1 = self.psi(self.phi(mel))
#        pred = self.res(self.phi(mel))
#        perm = torch.randperm(mel.size(0))
#        pred2 = torch.bmm(spkr.unsqueeze(1), pred)
#        pred3 = torch.bmm(spkr[perm].unsqueeze(1), pred)
#        perm = torch.randperm(mel.size(0))
#        pred4 = torch.bmm(rv.unsqueeze(1), pred)
#        pred5 = torch.bmm(rv[perm].unsqueeze(1), pred)
#        perm = torch.randperm(mel.size(0))
#        pred6 = torch.bmm(acc.unsqueeze(1), pred)
#        pred7 = torch.bmm(acc[perm].unsqueeze(1), pred)
#        pred6 = torch.bmm(self.match(spkr).unsqueeze(1), self.match(rv).unsqueeze(2))
#        pred7 = torch.bmm(self.match(spkr[perm]).unsqueeze(1), self.match(rv).unsqueeze(2))
        result = pred1# + pred2 - pred3 + pred4 - pred5
        result = result.squeeze(1)
        return result#, (pred7 - pred6).squeeze(1).squeeze(1)

class EGenerator(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.layers1 = nn.ModuleList([ConvLNBlock(hp.hidden_size, hp.dropout, dilation=(2*i+1)) for i in range(hp.mel_layers//2)])
        self.linear = nn.Linear(hp.hidden_size, 1)
        self.hp = hp

    def forward(self, x):
        #x: N, T, C
        t = x.size(1)
        x = x.transpose(1, 2)
        for layer in self.layers1:
            x = layer(x)
        x = x.transpose(1, 2) #N, T, C
        x = self.linear(x) #N, T, C
        return x

