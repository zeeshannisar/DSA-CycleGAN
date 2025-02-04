#! /bin/bash
source_domain=02
target_domain=32
num_epochs=50
save_interval=500
extra_channel=True
extra_channel_mode=RGB
data_path="${HOME}/phd/data/Nephrectomies_random"
output_path="${HOME}/phd/saved_models/I2I_translation_models/DSA_CycleGAN/CycleGAN_with_Extra_Channels"
repetition=rep1

cd "${HOME}/phd/code/GitHub/personal_implementations/DSA-CycleGAN/CycleGAN_with_Extra_Channels/code"

## echo running command so you can see what is going to be executed
echo python3 train_CycleGAN_with_Extra_Channels.py -sd ${source_domain} -td ${target_domain} -ne ${num_epochs} -si ${save_interval} -dp ${data_path} -op ${output_path} -rep ${repetition}
## run command
python3 train_CycleGAN_with_Extra_Channels.py -sd ${source_domain} -td ${target_domain} -ne ${num_epochs} -si ${save_interval} -ec ${extra_channel} -ecm ${extra_channel_mode} -dp ${data_path} -op ${output_path} -rep ${repetition}

cd ..
