# Grad-TTS

<p align="center">
    <img src="resources/reverse-diffusion.gif" alt="drawing" width="500"/>
</p>

Grad-TTS model based on Diffusion Probabilistic Modelling. For all details check out our paper accepted to ICML 2021 via [this](https://arxiv.org/abs/2105.06337) link.

## Abstract

**Demo page** with voiced abstract: [link](https://grad-tts.github.io/).

Recently, denoising diffusion probabilistic models and generative score matching have shown high potential in modelling complex data distributions while stochastic calculus has provided a unified point of view on these techniques allowing for flexible inference schemes. In this paper we introduce Grad-TTS, a novel text-to-speech model with score-based decoder producing mel-spectrograms by gradually transforming noise predicted by encoder and aligned with text input by means of Monotonic Alignment Search. The framework of stochastic differential equations helps us to generalize conventional diffusion probabilistic models to the case of reconstructing data from noise with different parameters and allows to make this reconstruction flexible by explicitly controlling trade-off between sound quality and inference speed. Subjective human evaluation shows that Grad-TTS is competitive with state-of-the-art text-to-speech approaches in terms of Mean Opinion Score.

## Installation

Firstly, install all Python package requirements:

```bash
pip install -r requirements.txt
```

Secondly, build `monotonic_align` code (Cython):

```bash
cd model/monotonic_align; python setup.py build_ext --inplace; cd ../..
```

**Note**: code is tested on Python==3.8

## Low-bit-optimizer Installation

**Requirements**
Python >= 3.7 + CUDA >= 11.0 + torch >= 1.13.0.

To install run:

```bash
git clone https://github.com/thu-ml/low-bit-optimizers.git
pip install -v -e .
```

## Choose Cleaners

* Fill "text_cleaners" in params.py
* Edit text/symbols.py
* Remove unnecessary imports from text/cleaners.py

## Inference

You can download HiFi-GAN checkpoint trained on LJSpeech* and Libri-TTS datasets (22kHz) from [here](https://drive.google.com/drive/folders/1grsfccJbmEuSBGQExQKr3cVxNV0xEOZ7?usp=sharing).

Put necessary Grad-TTS and HiFi-GAN checkpoints into `checkpts` folder in root Grad-TTS directory (note: in `inference.py` you can change default HiFi-GAN path).

1. Create text file with sentences you want to synthesize like `test.txt`.
2. For single speaker set `params.n_spks=1` and for multispeaker (Libri-TTS) inference set `params.n_spks=247`.
3. Run script `inference.py` by providing path to the text file, path to the Grad-TTS checkpoint, number of iterations to be used for reverse diffusion (default: 10) and speaker id if you want to perform multispeaker inference:
    ```bash
    python inference.py -f <your-text-file> -c <grad-tts-checkpoint> -t <number-of-timesteps> -s <speaker-id-if-multispeaker>
    ```
4. Check out folder called `outputs` for generated audios.

You can also perform *interactive inference* by running Jupyter Notebook `inference.ipynb`.

## Training

1. Make filelists of your audio data like ones included into `resources/filelists` folder. For single speaker training refer to `ljspeech` filelists and to `libri-tts` filelists for multispeaker.
2. Set experiment configuration in `params.py` file.
3. Specify your GPU device and run training script:
    ```bash
    export CUDA_VISIBLE_DEVICES=YOUR_GPU_ID
    python train.py  # if single speaker
    python train_multi_speaker.py  # if multispeaker
    ```
4. To track your training process run tensorboard server on any available port:
    ```bash
    tensorboard --logdir=YOUR_LOG_DIR
    ```
    During training all logging information and checkpoints are stored in `YOUR_LOG_DIR`, which you can specify in `params.py` before training.

## References

* HiFi-GAN model is used as vocoder, official github repository: [link](https://github.com/jik876/hifi-gan).
* Monotonic Alignment Search algorithm is used for unsupervised duration modelling, official github repository: [link](https://github.com/jaywalnut310/glow-tts).
* text/cleaners.py [ORI-Muchim/PolyLangVITS](https://github.com/ORI-Muchim/PolyLangVITS)
* [low-bit-optimizers](https://github.com/thu-ml/low-bit-optimizers)

## To-Do

* Model Quantization(float32 -> int8) ONLY INFERENCE
