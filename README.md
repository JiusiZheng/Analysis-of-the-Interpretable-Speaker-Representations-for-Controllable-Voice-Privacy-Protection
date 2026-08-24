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

## 2. Data Processing
Convert the VCTK and L2-Arctic datasets into speech units with pitch information. You need to configure `--datadir` and `--outdir` by yourself.

VCTK
```
CUDA_VISIBLE_DEVICES=0 nohup python s2u-VCTK.py \
    --VCTK \
    --with_pitch_unit \
    > vctk_process.txt 2>&1 &
```
L2-Arctic
```
CUDA_VISIBLE_DEVICES=0 nohup python s2u-L2A.py \
    --with_pitch_unit \
    > l2a_process.txt 2>&1 &
```
Both commands run in the background. Processing logs are saved to `vctk_process.txt` and `l2a_process.txt`, respectively.

Modify the absolute paths in `datasets_merged/train_data_final_vctk_l2a.txt` and `datasets_merged/valid_data_final_vctk_l2a.txt`.

## 3. Get the Pre-trained Vocoder and UUVC model

We use the pretrained [HiFi-GAN vocoder](https://github.com/jik876/hifi-gan). Download the [Universal-V1](https://drive.google.com/drive/folders/1YuOoV3lO2-Hhn1F2HJ2aQ4S0LC1JdKLd) model and place its checkpoint files in:

```text
vocoder/cp_hifigan/
```

Also, place the downloaded [checkpoint of pretrained UUVC model](https://cmu.app.box.com/s/76f7kkhuns929da4kaafjqqk2x7nf2d3) in the corresponding model checkpoint directory before running training or inference.

```text
pretrained_LibriTTS360_VCTK_ESD-selected/
```
## 4.Training of the Decomposition Framework

```
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python train.py \
    --saving_path ckpt/ \
    --training_step 60000 \
    --batch_size 100 \
    --check_val_every_n_epoch 5 \
    --traintxt datasets_merged/train_data_final_vctk_l2a.txt \
    --validtxt datasets_merged/valid_data_final_vctk_l2a.txt \
    --distributed \
    > train_decomposition.log 2>&1 &
```

## 5.Inference
This is an example for synthesis of the converted speech, the format of the meta data can be found in `accent_300.txt`:
```
python inference_TD.py --result_dir ./samples --metapath META_PATH
```

## 6.Visualization of the Mel-Spectrogram
```
python time-scale_analysis.py
```

## 7.Acknowledgements

This work builds upon the open-source implementation of [UUVC](https://github.com/b04901014/UUVC). We sincerely thank the original authors for making their code publicly available. Our repository extends and adapts UUVC for the analysis of interpretable speaker representations and controllable voice privacy protection.


