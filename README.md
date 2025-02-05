# DSA-CycleGAN

This repository contains the code for the paper "[DSA-CycleGAN: A Domain Shift Aware CycleGAN to enhance Stain Transfer in Digital Histopathology](https://openreview.net/pdf?id=zYBYJKHEhz)" submitted to [MIDL-2025](https://2025.midl.io/).

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Acknowledgements](#acknowledgements)

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

## Acknowledgements
This work is funded by Funded by ANR HistoGraph (ANR-23-CE45-0038) and the ArtIC project &ldquo;Artificial 
Intelligence for Care&rdquo; (grant ANR-20-THIA-0006-01), co funded by *Région Grand Est*, Inria Nancy - 
Grand Est, IHU Strasbourg, University of Strasbourg & University of Haute-Alsace. We acknowledge 
the ERACoSysMed & e:Med initiatives by BMBF, SysMIFTA (managed by PTJ, FKZ 031L-0085A; ANR, grant 
ANR-15-CMED-0004), Prof. Cédric Wemmert, and Prof. Friedrich Feuerhake and team at MHH for the 
high-quality images & annotations: specifically N. Kroenke, for excellent technical assistance, 
N. Schaadt, for image management and quality control, and V. Volk & J. Schmitz. for annotations 
under the supervision of domain experts.
