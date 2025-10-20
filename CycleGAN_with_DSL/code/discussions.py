import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import numpy
import tqdm

import tensorflow as tf

from code.model import CycleGAN as cyclegan
from train_cyclegan_plus_extrachannel import add_extra_channel_to_input, split_extra_channel_from_input
from helper import utils


def transform_extra_channel(model, patch, extra_channel_mode, direction):
    patch = (patch[numpy.newaxis, ...] / 127.5) - 1
    patch_w_extra_channel = add_extra_channel_to_input(patch, extra_channel_mode)
    if direction == "source2target":
        translation_w_Noise = model.G_AB.predict(patch_w_extra_channel)
        # reconstruction_w_Noise = (model.G_AB.predict(patch_w_extra_channel[numpy.newaxis, ...]) + 1) * 127.5
        reconstruction_w_Noise = model.G_BA.predict(translation_w_Noise)
        identity_w_Noise = model.G_BA.predict(patch_w_extra_channel)
    elif direction == "target2source":
        translation_w_Noise = model.G_BA.predict(patch_w_extra_channel)
        reconstruction_w_Noise = model.G_AB.predict(translation_w_Noise)
        identity_w_Noise = model.G_AB.predict(patch_w_extra_channel)

    return patch_w_extra_channel, translation_w_Noise, reconstruction_w_Noise, identity_w_Noise


def save_extra_channels_diagram(patch_w_metaINFO, trans_w_Noise, recons_w_Noise, id_w_Noise, filename, direction,
                                extra_channel_mode, source_stain, target_stain):
    if direction == "source2target":
        save_mode = os.path.join(source_stain + "_to_" + target_stain, extra_channel_mode)
    elif direction == "target2source":
        save_mode = os.path.join(target_stain + "_to_" + source_stain, extra_channel_mode)

    save_dir = os.path.join("/home/nisar/phd/saved_models/UNet/SSL/unet/percentN_equally_randomly/qualitative_outputs",
                            "translations/cyclegan_plus_extrachannel", direction, save_mode)
    os.makedirs(save_dir, exist_ok=True)

    patch, extra_channel = split_extra_channel_from_input(patch_w_metaINFO)
    trans, trans_noise = split_extra_channel_from_input(trans_w_Noise)
    recons, recons_noise = split_extra_channel_from_input(recons_w_Noise)
    id, id_noise = split_extra_channel_from_input(id_w_Noise)

    utils.save_image((patch[0, :, :, :] + 1) * 127.5, os.path.join(save_dir, filename))
    utils.save_image((extra_channel[0, :, :, :] + 1) * 127.5, os.path.join(save_dir, "extra_channel.png"))

    utils.save_image((trans[0, :, :, :] + 1) * 127.5,
                     os.path.join(save_dir, filename.rsplit('.', 1)[0] + f"_translated.png"))
    utils.save_image((trans_noise[0, :, :, :] + 1) * 127.5,
                     os.path.join(save_dir, filename.rsplit('.', 1)[0] + f"_translated_noise.png"))

    utils.save_image((recons[0, :, :, :] + 1) * 127.5,
                     os.path.join(save_dir, filename.rsplit('.', 1)[0] + f"_reconstructed.png"))
    utils.save_image((recons_noise[0, :, :, :] + 1) * 127.5,
                     os.path.join(save_dir, filename.rsplit('.', 1)[0] + f"_reconstructed_noise.png"))

    utils.save_image((id[0, :, :, :] + 1) * 127.5,
                     os.path.join(save_dir, filename.rsplit('.', 1)[0] + f"_identity.png"))
    utils.save_image((id_noise[0, :, :, :] + 1) * 127.5,
                     os.path.join(save_dir, filename.rsplit('.', 1)[0] + f"_identity_noise.png"))



def translate(images_dir, model, direction, extra_channel_mode, source_stain, target_stain):
    # Read images from directory
    for filename in tqdm.tqdm(os.listdir(images_dir)):
        patch = utils.read_image(os.path.join(images_dir, filename)).astype(numpy.float32)
        # patch = utils.image_colour_convert(patch, "rgb")
        # Taking the central part of the image to make the shape 508, 508, 3
        # patch = patch[2:510, 2:510, :]
        patch_w_metaINFO, trans_w_Noise, recons_w_Noise, id_w_Noise = transform_extra_channel(model, patch,
                                                                                             extra_channel_mode,
                                                                                             direction)
        save_extra_channels_diagram(patch_w_metaINFO, trans_w_Noise, recons_w_Noise, id_w_Noise,
                                    filename, direction, extra_channel_mode, source_stain, target_stain)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Save results for Discussions Section.')

    parser.add_argument('-id', '--images_dir', type=str)
    parser.add_argument("-i2i_ms", "--image2image_model_strategy", type=str)
    parser.add_argument("-i2i_mp", "--image2image_model_path", type=str)
    parser.add_argument("-i2i_md", "--image2image_model_direction", type=str)
    parser.add_argument("-i2i_me", "--image2image_model_at_epoch", type=int, default=50)
    parser.add_argument("-ec", "--extra_channel", default=True, type=lambda x: True if str(x).lower() in ['true', '1', 'yes'] else False)
    parser.add_argument("-ecm", "--extra_channel_mode", default="rgb", type=lambda x: str(x).lower())
    parser.add_argument('-ss', '--source_stain', type=str, default='02')
    parser.add_argument('-ts', '--target_stain', type=str, default='32')

    args = parser.parse_args()
    args.image2image_model_direction = "source2target"
    args.image2image_model_strategy = "cyclegan_plus_extrachannel"
    if args.extra_channel:
        args.image2image_model_path = os.path.join("/home/nisar/phd/saved_models/I2I_translation_models",
                                                   "self_supervised_cycleGAN/cyclegan_plus_extrachannel",
                                                   f"{args.extra_channel_mode}", f"target_{args.target_stain}", "rep1")
    if args.image2image_model_direction == "source2target":
        args.images_dir = os.path.join("/home/nisar/phd/saved_models/UNet/SSL/unet/percentN_equally_randomly",
                                       "qualitative_outputs", f"{args.source_stain}", "images")
    else:
        args.images_dir = os.path.join("/home/nisar/phd/saved_models/UNet/SSL/unet/percentN_equally_randomly",
                                       "qualitative_outputs", f"{args.target_stain}", "images")

    if args.image2image_model_strategy.lower() == "cyclegan_plus_extrachannel":
        image2image_model = cyclegan(norm="instance", extra_channel=True, extra_channel_mode=args.extra_channel_mode)
    else:
        image2image_model = cyclegan(norm="instance")

    checkpoint = tf.train.Checkpoint(G_AB=image2image_model.G_AB, G_BA=image2image_model.G_BA, D_A=image2image_model.D_A,
                                     D_B=image2image_model.D_B, G_optimizer=image2image_model.G_optimizer,
                                     D_optimizer=image2image_model.D_optimizer)
    manager = tf.train.CheckpointManager(checkpoint,
                                         directory=os.path.join(args.image2image_model_path, "models", "TF_checkpoints"),
                                         max_to_keep=1, checkpoint_name='latest_ckpt', step_counter=None,
                                         checkpoint_interval=None, init_fn=None)
    checkpoint.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        latest_model_at_epoch = int(manager.latest_checkpoint.rsplit("/", 1)[1].rsplit("-", 1)[1])
        if latest_model_at_epoch == args.image2image_model_at_epoch:
            translation_model = image2image_model

            # if args.image2image_model_direction == "source2target":
            #     print(f"\nLoading unpaired image to image model to translate from Source-->Target. The model is "
            #           f"trained for {latest_model_at_epoch} epochs.")
            #     translation_model = image2image_model.G_AB
            # elif args.image2image_model_direction == "target2source":
            #     print(f"\nLoading unpaired image to image model to to translate from Target-->Source. The model is "
            #           f"trained for {latest_model_at_epoch} epochs.")
            #     translation_model = image2image_model.G_BA
            # else:
            #     raise ValueError('\nTranslation direction is invalid. It should be one of ["src2tgt", "tgt2src"].')
        else:
            raise ValueError(f"\nTraining is done for only {latest_model_at_epoch} epoch and is not completed "
                             f"for upto {args.image2image_model_at_epoch} epochs.")
    else:
        raise ValueError(f"No trained models are available.")

    # print(translation_model.summary())
    translate(args.images_dir, translation_model, args.image2image_model_direction, args.extra_channel_mode,
              args.source_stain, args.target_stain)
