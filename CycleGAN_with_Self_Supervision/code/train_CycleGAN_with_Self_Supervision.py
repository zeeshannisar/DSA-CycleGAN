import os

import matplotlib
matplotlib.use('Agg')
import numpy as np
import glob
import os
import datetime
import h5py
import json
import argparse

import tensorflow as tf
import matplotlib.pyplot as plt

from model import CycleGAN
from Image_Pool_model import ImagePool
from dataloader.tf_dataloader_API import TFDataLoader
from callbacks import SaveHistory, update_train_history
from select_gpu import pick_gpu_lowest_memory
import tensorflow.keras.backend as K


def SaveTrainingHistory(logpath, historyfile):
    """
        Save the history of the model
    """

    with open(historyfile, 'r') as f:
        history = json.load(f)

    if 'lr' in history:
        history['lr'] = [float(v) for v in history['lr']]

    # Summarise loss history
    plt.clf()
    plt.plot(history['d_loss'])
    plt.plot(history['g_loss'])
    plt.plot(history['adv_loss'])
    plt.plot(history['cyc_loss'])
    plt.plot(history['id_loss'])
    plt.plot(history['seg_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.legend(['dis_loss', 'gen_loss', 'adv_loss', 'cyc_loss', 'id_loss', 'seg_loss'], loc='best')
    plt.gcf().savefig(os.path.join(logpath, 'loss_history.png'))

    # Write history to json file
    with open(os.path.join(logpath, 'history.json'), 'w') as fp:
        json.dump(history, fp, indent=True)


def TryRestoreExistingCkpt(G_AB, G_BA, D_A, D_B, G_optimizer, D_optimizer, savePath, start_epoch):
    os.makedirs(savePath, exist_ok=True)

    checkpoint = tf.train.Checkpoint(G_AB=G_AB, G_BA=G_BA, D_A=D_A, D_B=D_B,
                                     G_optimizer=G_optimizer, D_optimizer=D_optimizer)

    manager = tf.train.CheckpointManager(checkpoint, directory=os.path.join(savePath, "TF_checkpoints"),
                                         max_to_keep=1, checkpoint_name='latest_ckpt', step_counter=None,
                                         checkpoint_interval=None, init_fn=None)

    checkpoint.restore(manager.latest_checkpoint).expect_partial()

    if manager.latest_checkpoint:
        print(f"Training is resumed at Epoch: {start_epoch} with checkpoint: {manager.latest_checkpoint}")
    else:
        print("Training is started from scratch...")

    return checkpoint, manager


def SaveLatestModelWeights(G_AB, G_BA, D_A, D_B, currentEpoch, savePath):
    os.makedirs(savePath, exist_ok=True)
    for file in glob.glob(os.path.join(savePath, "*.h5")):
        os.remove(file)

    G_AB.save_weights(os.path.join(savePath, f"g_AB_latest_epoch_{currentEpoch:03d}.h5"))
    G_BA.save_weights(os.path.join(savePath, f"g_BA_latest_epoch_{currentEpoch:03d}.h5"))
    D_A.save_weights(os.path.join(savePath, f"d_A_latest_epoch_{currentEpoch:03d}.h5"))
    D_B.save_weights(os.path.join(savePath, f"d_B_latest_epoch_{currentEpoch:03d}.h5"))


def saveModelArchitecture(path, model):
    """
    Function to save model architecture and parameters

    :param path: folder to save model
    :param model: model to be saved
    :return:
    """
    if (not os.path.isdir(path)):
        os.makedirs(path)

    # summary of model
    with open(path + '/model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # saving architecture, model can be recreated from this file
    json_str = model.to_json()
    with open(path + '/model_json.json', "w") as f:
        json.dump(json.loads(json_str), f, indent=4)


def LR_decay(epoch, same_lr, reach_zero):
    return 1.0 - max(0, epoch + 1 - same_lr) / float(reach_zero)

def set_LR(optimizer, epoch, epochs):
    current_LR = tf.keras.backend.get_value(optimizer.lr)
    tf.keras.backend.set_value(optimizer.lr, current_LR * LR_decay(epoch, epochs / 2, epochs / 2))


def load_pretrained_model(args):
    model = tf.keras.models.load_model(
        os.path.join(args.pretrained_model_path, "unet_best." + args.pretrained_model_label + ".hdf5"))

    model.trainable = False
    model._name = "unet_model"

    # Read normalisation statistics
    with h5py.File(os.path.join(args.pretrained_model_path,
                                "normalisation_stats." + args.pretrained_model_label + ".hdf5"), "r") as f:
        mean = f["stats"][0]
        stddev = f["stats"][1]

    return {"model": model,
            "name": model._name,
            "mean": mean,
            "stddev": stddev,
            "loss_name": "cyclegan_with_segmentation_loss"}


def standardise_sample(sample):
    sample_min = tf.reduce_min(sample, axis=[0, 1], keepdims=True)  # Min value across height and width
    sample_max = tf.reduce_max(sample, axis=[0, 1], keepdims=True)  # Max value across height and width
    standardized_sample = (sample - sample_min) / (sample_max - sample_min + K.epsilon())
    return standardized_sample


def normalise_sample(sample, mean, stddev):
    sample -= mean
    sample /= stddev + K.epsilon()
    return sample


def extract_features(image, model, mean, stddev):
    image = (image + 1) * 127.5  # [0-255]
    image = standardise_sample(image)
    image = normalise_sample(image, mean, stddev)
    return model(image, training=False)

def segmentation_loss(y_true, y_pred, model):
    y_true = extract_features(y_true, model["model"], model["mean"], model["stddev"])
    y_pred = extract_features(y_pred, model["model"], model["mean"], model["stddev"])
    return tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)

def disc_loss_func(y_true, y_pred):
    y_true_loss = tf.keras.losses.MeanSquaredError()(y_true, tf.ones_like(input=y_true, dtype=y_true.dtype))
    y_pred_loss = tf.keras.losses.MeanSquaredError()(y_pred, tf.zeros_like(input=y_pred, dtype=y_pred.dtype))
    return 0.5 * (y_true_loss + y_pred_loss)

def gen_loss_func(y_pred):
    return tf.keras.losses.MeanSquaredError()(y_pred, tf.ones_like(input=y_pred, dtype=y_pred.dtype))

def cyc_loss_func(y_true, y_pred):
    return tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)

def id_loss_func(y_true, y_pred):
    return tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)


class Training:
    def __init__(self, args):
        source_dataset = TFDataLoader(dataDir=os.path.join(args.data_path, args.source_domain, 'patches', 'colour'),
                                      mode='train', imageSize=(args.image_height, args.image_width, args.image_channel),
                                      batchSize=args.batch_size).LoadDataset()

        target_dataset = TFDataLoader(dataDir=os.path.join(args.data_path, args.target_domain, 'patches', 'colour'),
                                      mode='train', imageSize=(args.image_height, args.image_width, args.image_channel),
                                      batchSize=args.batch_size).LoadDataset()

        self.dataset = tf.data.Dataset.zip((source_dataset, target_dataset))
        self.__len__dataset__ = max(source_dataset.cardinality().numpy(), target_dataset.cardinality().numpy())

        # Initialise model
        print(f'Initialising cycleGAN-model with normalization: {args.normalisation}')
        self.model = CycleGAN(args.normalisation)
        self.output_path = os.path.join(args.output_path, f'target_{args.target_domain}', arguments.repetition)
        os.makedirs(self.output_path, exist_ok=True)
        print(f'output_path: {self.output_path}')

        # Save model architecture
        saveModelArchitecture(os.path.join(self.output_path, "modelsArchitectures/generators/G_AB"), self.model.G_AB)
        saveModelArchitecture(os.path.join(self.output_path, "modelsArchitectures/generators/G_BA"), self.model.G_BA)
        saveModelArchitecture(os.path.join(self.output_path, "modelsArchitectures/discriminators/D_A"), self.model.D_A)
        saveModelArchitecture(os.path.join(self.output_path, "modelsArchitectures/discriminators/D_B"), self.model.D_B)

        # the buffer stores 50 generated images
        self.fake_A_pool = ImagePool(50)
        self.fake_B_pool = ImagePool(50)

        self.model_weights = {"lambda_adv": self.model.lambda_adv,
                              "lambda_cycle": self.model.lambda_cycle,
                              "lambda_id": self.model.lambda_id,
                              "lambda_segmentation": arguments.lambda_segmentation}

        # Initialize pretrained model
        self.pretrained_model = load_pretrained_model(args)
        # self.pretrained_model["model"].summary()

    def train(self, NumEpochs, batch_size=1, sample_interval=10):
        os.makedirs(os.path.join(self.output_path, "graphs"), exist_ok=True)

        if os.path.isfile(os.path.join(self.output_path, "graphs", "history.json")):
            history = SaveHistory(os.path.join(self.output_path, "graphs", "history.json"), read_existing=True)
            StartEpoch = int(len(history.on_train_begin()['epoch']))
        else:
            history = SaveHistory(os.path.join(self.output_path, "graphs", "history.json"), read_existing=False)
            StartEpoch = 0

        if StartEpoch >= NumEpochs // 2:
            print('Reduced the influence of cycle-consistency to 5, identity to 2.5')
            self.model.lambda_cycle = self.model.lambda_cycle // 2
            self.model.lambda_id = self.model.lambda_id // 2
            self.model_weights.update({"lambda_cycle": self.model.lambda_cycle, "lambda_id": self.model.lambda_id})

        @tf.function
        def train_G(imgs_A, imgs_B):
            # persistent is set to True because the tape is used more than once to calculate the gradients.
            with tf.GradientTape() as tape:
                # Generator G_AB translates A -> B
                fake_B = self.model.G_AB(imgs_A, training=True)
                cyc_A = self.model.G_BA(fake_B, training=True)
                # Generator G_BA translates B -> A.
                fake_A = self.model.G_BA(imgs_B, training=True)
                cyc_B = self.model.G_AB(fake_A, training=True)
                # same_x and same_y are used for identity loss.
                id_A = self.model.G_BA(imgs_A, training=True)
                id_B = self.model.G_AB(imgs_B, training=True)
                # discriminator output
                disc_fake_B = self.model.D_B(fake_B, training=True)
                disc_fake_A = self.model.D_A(fake_A, training=True)

                # adversarial loss
                adv_loss_AB = gen_loss_func(disc_fake_B)
                adv_loss_BA = gen_loss_func(disc_fake_A)
                adv_loss = adv_loss_AB + adv_loss_BA

                # cycle loss
                cyc_loss_ABA = self.model_weights["lambda_cycle"] * cyc_loss_func(imgs_A, cyc_A)
                cyc_loss_BAB = self.model_weights["lambda_cycle"] * cyc_loss_func(imgs_B, cyc_B)
                cyc_loss = cyc_loss_ABA + cyc_loss_BAB

                # identity loss
                id_loss_AA = self.model_weights["lambda_id"] * id_loss_func(imgs_A, id_A)
                id_loss_BB = self.model_weights["lambda_id"] * id_loss_func(imgs_B, id_B)
                id_loss = id_loss_AA + id_loss_BB

                # segmentation loss
                seg_loss_AcycA = segmentation_loss(imgs_A, cyc_A, self.pretrained_model)
                seg_loss_AidA = segmentation_loss(imgs_A, id_A, self.pretrained_model)
                seg_loss = self.model_weights["lambda_segmentation"] * (seg_loss_AcycA + seg_loss_AidA)


                # adversarial loss + cycle loss + identity loss + segmentation_loss
                gen_loss = adv_loss + cyc_loss + id_loss + seg_loss

                losses = {"gen_loss": gen_loss, "adv_loss": adv_loss, "cyc_loss": cyc_loss, "id_loss": id_loss,
                          "seg_loss": seg_loss}

            # Calculate the gradients for generator
            G_grads = tape.gradient(gen_loss, self.model.G_AB.trainable_variables + self.model.G_BA.trainable_variables)
            self.model.G_optimizer.apply_gradients(zip(G_grads, self.model.G_AB.trainable_variables + self.model.G_BA.trainable_variables))
            return fake_B, fake_A, losses

        @tf.function
        def train_D(imgs_A, imgs_B, fake_B, fake_A):
            with tf.GradientTape() as tape:
                disc_real_A = self.model.D_A(imgs_A, training=True)
                disc_fake_A = self.model.D_A(fake_A, training=True)

                disc_real_B = self.model.D_B(imgs_B, training=True)
                disc_fake_B = self.model.D_B(fake_B, training=True)

                # calculate Discriminator loss
                disc_loss_A = disc_loss_func(disc_real_A, disc_fake_A)
                disc_loss_B = disc_loss_func(disc_real_B, disc_fake_B)
                disc_loss = disc_loss_A + disc_loss_B

            # Calculate the gradients for discriminator
            D_grads = tape.gradient(disc_loss, self.model.D_A.trainable_variables + self.model.D_B.trainable_variables)
            self.model.D_optimizer.apply_gradients(zip(D_grads, self.model.D_A.trainable_variables + self.model.D_B.trainable_variables))
            return disc_loss


        def train_on_batch(image_A, image_B):
            fake_B, fake_A, g_loss_per_batch_tmp = train_G(image_A, image_B)
            # Image pool utilization
            fake_B = self.fake_B_pool.query(fake_B)
            fake_A = self.fake_A_pool.query(fake_A)
            d_loss_per_batch_tmp = train_D(image_A, image_B, fake_B, fake_A)
            return d_loss_per_batch_tmp, g_loss_per_batch_tmp

        ckpt, manager = TryRestoreExistingCkpt(self.model.G_AB, self.model.G_BA, self.model.D_A, self.model.D_B,
                                               self.model.G_optimizer, self.model.D_optimizer,
                                               savePath=os.path.join(self.output_path, "models"), start_epoch=StartEpoch)

        if StartEpoch >= NumEpochs:
            print(f'Model has already trained upto {NumEpochs} epochs...')
        else:
            for currentEpoch in range(StartEpoch, NumEpochs):
                set_LR(optimizer=self.model.G_optimizer, epoch=currentEpoch, epochs=NumEpochs)
                set_LR(optimizer=self.model.D_optimizer, epoch=currentEpoch, epochs=NumEpochs)

                print(f'\nEpoch {currentEpoch+1}/{NumEpochs}, '
                      f'gen_LR: {self.model.G_optimizer.lr.numpy():.6f}, '
                      f'disc_LR: {self.model.D_optimizer.lr.numpy():.6f}')

                progbar = tf.keras.utils.Progbar(target=self.__len__dataset__,
                                                 stateful_metrics=['d_loss', 'g_loss', 'adv_loss', 'cyc_loss',
                                                                   'id_loss', 'seg_loss'],
                                                 verbose=1)
                d_loss_total, g_loss_total, adv_loss_total, cyc_loss_total, id_loss_total, seg_loss_total = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


                if currentEpoch == NumEpochs // 2:
                    print('Reduced the influence of cycle-consistency to 5, identity to 2.5')
                    self.model.lambda_cycle = self.model.lambda_cycle // 2
                    self.model.lambda_id = self.model.lambda_id // 2

                    self.model_weights.update({"lambda_cycle": self.model.lambda_cycle,
                                               "lambda_id": self.model.lambda_id,
                                               "lambda_segmentation": arguments.lambda_segmentation})

                for step, (image_A, image_B) in enumerate(self.dataset):
                    assert image_A.shape[0] == batch_size
                    assert image_B.shape[0] == batch_size

                    d_loss_per_batch, g_loss_per_batch = train_on_batch(image_A, image_B)

                    d_loss_total += d_loss_per_batch.numpy()
                    g_loss_total += g_loss_per_batch["gen_loss"].numpy()
                    adv_loss_total += g_loss_per_batch["adv_loss"].numpy()
                    cyc_loss_total += g_loss_per_batch["cyc_loss"].numpy()
                    id_loss_total += g_loss_per_batch["id_loss"].numpy()
                    seg_loss_total += g_loss_per_batch["seg_loss"].numpy()


                    # Update Progbar
                    progbar.update(step + 1, values=[('d_loss', d_loss_total / (step + 1)),
                                                     ('g_loss', g_loss_total / (step + 1)),
                                                     ('adv_loss', adv_loss_total / (step + 1)),
                                                     ('cyc_loss', cyc_loss_total / (step + 1)),
                                                     ('id_loss', id_loss_total / (step + 1)),
                                                     ('seg_loss', seg_loss_total / (step + 1))])

                    if step % sample_interval == 0:
                        self.sample_images(image_A, image_B, currentEpoch, step,
                                           saveDir=os.path.join(self.output_path, "train_outputs"))

                d_loss_avg = d_loss_total / self.__len__dataset__
                g_loss_avg = g_loss_total / self.__len__dataset__
                adv_loss_avg = adv_loss_total / self.__len__dataset__
                cyc_loss_avg = cyc_loss_total / self.__len__dataset__
                id_loss_avg = id_loss_total / self.__len__dataset__
                seg_loss_avg = seg_loss_total / self.__len__dataset__

                update_train_history(history, currentEpoch+1, d_loss_avg, g_loss_avg, adv_loss_avg,
                                     cyc_loss_avg, id_loss_avg, seg_loss_avg)

                print(f"saving latest checkpoint at {manager.save()}")

            SaveTrainingHistory(os.path.join(self.output_path, "graphs"),
                                historyfile=os.path.join(self.output_path, "graphs", "history.json"))

    def sample_images(self, img_A, img_B, epoch, step, saveDir):
        os.makedirs(saveDir, exist_ok=True)
        r, c = 2, 4
        # Translate images to the other domain
        fake_B = self.model.G_AB.predict(img_A)
        fake_A = self.model.G_BA.predict(img_B)
        # Translate back to original domain
        cyc_A = self.model.G_BA.predict(fake_B)
        cyc_B = self.model.G_AB.predict(fake_A)

        id_A = self.model.G_BA.predict(img_A)
        id_B = self.model.G_AB.predict(img_B)

        img_A = img_A[0, :][np.newaxis, ...]
        img_B = img_B[0, :][np.newaxis, ...]
        fake_A = fake_A[0, :][np.newaxis, ...]
        fake_B = fake_B[0, :][np.newaxis, ...]
        cyc_A = cyc_A[0, :][np.newaxis, ...]
        cyc_B = cyc_B[0, :][np.newaxis, ...]
        id_A = id_A[0, :][np.newaxis, ...]
        id_B = id_B[0, :][np.newaxis, ...]

        gen_imgs = np.concatenate([img_A, fake_B, cyc_A, id_A, img_B, fake_A, cyc_B, id_B])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['original', 'translated', 'cycle', "identity"]
        fig, axs = plt.subplots(r, c, figsize=(12, 6))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(saveDir + '/image' + str(epoch) + '_' + str(step) + '.png')
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train cycleGAN with self-supervision.")

    parser.add_argument("-sd", "--source_domain", type=str, default="02")
    parser.add_argument("-td", "--target_domain", type=str, default="03")
    parser.add_argument("-ih", "--image_height", type=int, default=512)
    parser.add_argument("-iw", "--image_width", type=int, default=512)
    parser.add_argument("-ic", "--image_channel", type=int, default=3)
    parser.add_argument("-bs", "--batch_size", type=int, default=1)
    parser.add_argument("-norm", "--normalisation", type=str, default="instance")
    parser.add_argument("-ne", "--num_epochs", type=int, default=50)
    parser.add_argument("-si", "--save_interval", type=int, default=1000)
    parser.add_argument("-pmp", "--pretrained_model_path", type=str)
    parser.add_argument("-pml", "--pretrained_model_label", type=str)
    parser.add_argument("-l", "--lambda_segmentation", type=float, default=1.0)

    parser.add_argument("-dp", "--data_path", type=str, default=os.path.join(os.path.expanduser('~'), "data"))
    parser.add_argument("-op", "--output_path", type=str)
    parser.add_argument("-rep", "--repetition", type=str, default="rep1")

    arguments = parser.parse_args()
    print('input args:\n', json.dumps(vars(arguments), indent=4, separators=(',', ':')))  # pretty print args

    os.environ["CUDA_VISIBLE_DEVICES"] = str(pick_gpu_lowest_memory())

    print("Selected GPU : " + os.environ["CUDA_VISIBLE_DEVICES"])
    print("\n")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    start = datetime.datetime.now()

    cyclegan_with_dsl_model = Training(arguments)
    cyclegan_with_dsl_model.train(arguments.num_epochs, arguments.batch_size, arguments.save_interval)

    print('\nTraining Time: {}'.format(datetime.datetime.now() - start))
