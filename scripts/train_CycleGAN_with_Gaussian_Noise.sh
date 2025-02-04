#! /bin/bash
# Histopathological stains used in our study are 02 (PAS), 03 (Jones H&E), 16 (CD68), 32 (Sirius Red), 39 (CD34)
source_domain=02
num_epochs=50
save_interval=500
data_path="../data"
repetition=rep1

cd "../CycleGAN_with_Gaussian_Noise/code"
## run command
for noise_std_dev in 0.0125 0.025 0.05 0.075 0.1 0.3 0.5 0.9;
do
  for repetition in rep1 rep2 rep3;
  do
    for target_domain in 03 16 32 39;
    do
      output_path="../saved_models/I2I_translation_models/DSA_CycleGAN/CycleGAN_with_Gaussian_Noise/${noise_std_dev}"
      python3 train_CycleGAN_with_Gaussian_Noise.py -sd ${source_domain} -td ${target_domain} -ne ${num_epochs} -si ${save_interval} -stddev ${noise_std_dev} -dp ${data_path} -op ${output_path} -rep ${repetition}
    done
  done
done

