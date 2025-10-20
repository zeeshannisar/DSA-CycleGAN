#! /bin/bash
# Histopathological stains used in our study are 02 (PAS), 03 (Jones H&E), 16 (CD68), 32 (Sirius Red), 39 (CD34)
source_domain=02
num_epochs=50
save_interval=500
data_path="../data"

pretrained_model_path="../saved_models/UNet/Baseline/${source_domain}/models"
pretrained_model_label="rgb_${source_domain}_rep1_250"


cd "../CycleGAN_with_DSL/code"

## run command
for lambda_segmentation in 1.0 5.0 10.0;
do
  for target_domain in 03 16 32 39;
  do
    for repetition in rep1 rep2 rep3;
    do
      output_path="${HOME}/phd/saved_models/I2I_translation_models/DSA_CycleGAN/CycleGAN_with_Self_Supervision/lambda_segmentation_${lambda_segmentation}"
      python3 train_CycleGAN_with_Self_Supervision.py -sd ${source_domain} -td ${target_domain} -ne ${num_epochs} -si ${save_interval} -pmp ${pretrained_model_path} -pml ${pretrained_model_label} -l ${lambda_segmentation} -dp ${data_path} -op ${output_path} -rep ${repetition}
    done
  done
done

