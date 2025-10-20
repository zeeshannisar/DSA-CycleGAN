#! /bin/bash
# Histopathological stains used in our study are 02 (PAS), 03 (Jones H&E), 16 (CD68), 32 (Sirius Red), 39 (CD34)

source_domain=02
num_epochs=50
save_interval=500
data_path="../data/"
output_path="../saved_models/I2I_translation_models/DSA_CycleGAN/CycleGAN_original"

cd "../CycleGAN_original/code"

## run command
for target_domain in 03 16 32 39;
do
  for repetition in rep1 rep2 rep3;
  do
    python3 train_CycleGAN_original.py -sd ${source_domain} -td ${target_domain} -ne ${num_epochs} -si ${save_interval} -dp ${data_path} -op ${output_path} -rep ${repetition}
  done
done