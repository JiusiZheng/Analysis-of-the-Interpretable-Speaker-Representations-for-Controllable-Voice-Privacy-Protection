# Analysis-of-the-Interpretable-Speaker-Representations-for-Controllable-Voice-Privacy-Protection

## 1. Environment Setup

The environment uses Python 3.10.19. Install the main dependencies as follows:

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu126

pip install pytorch-lightning transformers librosa pyloudnorm
```

Install `textlesslib`:

```bash
git clone git@github.com:facebookresearch/textlesslib.git
cd textlesslib
pip install -e .
cd ..
```

Install `fairseq`:

```bash
git clone https://github.com/pytorch/fairseq.git
cd fairseq
pip install --editable ./
cd ..
```

> **Note:** Minor compatibility issues may occur when installing `textlesslib` and `fairseq`, but they do not currently affect model training or evaluation.

## 2. Get the Vocoder

We use the pretrained [HiFi-GAN vocoder](https://github.com/jik876/hifi-gan). Download the [Universal-V1](https://drive.google.com/drive/folders/1YuOoV3lO2-Hhn1F2HJ2aQ4S0LC1JdKLd) model and place its checkpoint files in:

```text
vocoder/cp_hifigan/
```
