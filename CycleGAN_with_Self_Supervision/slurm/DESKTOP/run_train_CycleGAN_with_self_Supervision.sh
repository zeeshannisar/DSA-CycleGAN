#! /bin/bash
source_domain=02
target_domain=32
num_epochs=50
save_interval=500
data_path="${HOME}/phd/data/Nephrectomies_random"

pretrained_model_path="${HOME}/phd/saved_models/UNet/Baseline/${source_domain}/models"
pretrained_model_label="rgb_${source_domain}_rep1_250"
lambda_segmentation=1.0

output_path="${HOME}/phd/saved_models/I2I_translation_models/DSA_CycleGAN/CycleGAN_with_Self_Supervision/lambda_segmentation_${lambda_segmentation}"
repetition=rep1

cd "${HOME}/phd/code/GitHub/personal_implementations/DSA-CycleGAN/CycleGAN_with_Self_Supervision/code"

## echo running command so you can see what is going to be executed
echo python3 train_CycleGAN_with_Self_Supervision.py -sd ${source_domain} -td ${target_domain} -ne ${num_epochs} -si ${save_interval} -pmp ${pretrained_model_path} -pml ${pretrained_model_label} -l ${lambda_segmentation} -op ${output_path} -dp ${data_path} -rep ${repetition}
## run command
python3 train_CycleGAN_with_Self_Supervision.py -sd ${source_domain} -td ${target_domain} -ne ${num_epochs} -si ${save_interval} -pmp ${pretrained_model_path} -pml ${pretrained_model_label} -l ${lambda_segmentation} -dp ${data_path} -op ${output_path} -rep ${repetition}

cd ..
