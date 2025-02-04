# DSA-CycleGAN

This repository contains the code for the paper "[DSA-CycleGAN: A Domain Shift Aware CycleGAN to enhance Stain Transfer in Digital Histopathology](https://openreview.net/pdf?id=zYBYJKHEhz)" submitted to [MIDL-2025](https://2025.midl.io/).

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Introduction
A key challenge in digital and computational histopathology is inter- and intra-stain variations, which can cause deep learning models to fail, particularly when trained on one stain (domain) and applied to others.

*Note: This section is under development and will be updated soon.*

## Installation
To install the required dependencies, run:
```
pip install -r requirements.txt
```

## Dataset
The dataset should be organized as follows to load and train the models. The stains used in our case are 02 (PAS), 03 (Jones H&E), 16 (CD68), 32 (Sirius Red), and 39 (CD34). As specified in the paper, PAS is considered the source stain, and the rest are target stains.

```
├── data
│   ├── 02
│   │   ├── patches
│   │   │   ├── colour
│   │   │   │   ├── train
│   │   │   │   │   ├── images
│   │   │   │   │   │   ├── images
│   │   │   │   │   │   │   ├── 0.png
│   │   │   │   │   │   │   ├── 1.png

│   ├── 03
│   │   ├── patches
│   │   │   ├── colour
│   │   │   │   ├── train
│   │   │   │   │   ├── images
│   │   │   │   │   │   ├── images
│   │   │   │   │   │   │   ├── 0.png
│   │   │   │   │   │   │   ├── 1.png
```

## Training
The scripts to train each respective model used in this work are provided in the `scripts` directory.
Particularly, to train our proposed DSA-CycleGAN model, run:
```
sh scripts/train_CycleGAN_with_DSL.sh
```
To train the Original CycleGAN model, run:
```
sh scripts/train_CycleGAN_original.sh
```

To train CycleGAN with Extra Channels, run:
```
sh scripts/train_CycleGAN_with_Extra_Channels.sh
```

To train CycleGAN with Self-supervision, run:
```
sh scripts/train_CycleGAN_with_Self_Supervision.sh
```

To train CycleGAN with Gaussian Noise, run:
```
sh scripts/train_CycleGAN_with_Gaussian_Noise.sh
```

## Results
*Note: This section is under development and will be updated soon.*

## Acknowledgements
*Note: This section is under development and will be updated soon.*
