import matplotlib
from model import CycleGAN_with_Gaussian_Noise, additive_gaussian_noise_layer
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import numpy as np
import os
import datetime
from load_data import make_generators_from_directory
import tensorflow as tf
import json
import argparse
from Image_Pool_model import ImagePool
from callbacks import SaveHistory, update_train_history


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

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.legend(['dis_loss', 'gen_loss', 'adv_loss', 'cyc_loss', 'id_loss'], loc='best')
    plt.gcf().savefig(os.path.join(logpath, 'loss_history.png'))

    # Write history to json file
    with open(os.path.join(logpath, 'history.json'), 'w') as fp:
        json.dump(history, fp, indent=True)

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


def learning_rate_decay(epoch, same_lr, reach_zero):
    return 1.0 - max(0, epoch + 1 - same_lr) / float(reach_zero)


class Training:
    def __init__(self, args):

        self.model = CycleGAN_with_Gaussian_Noise(args.normalisation, args.noise_std_dev)

        # Configure data loader
        source = args.source_domain # 02
        target = args.target_domain # 03
        batchSize = args.batch_size # 1

        self.data_path = args.data_path
        self.noise_std_dev = args.noise_std_dev

        self.output_path = os.path.join(args.output_path, f'target_{args.target_domain}', args.repetition)
        os.makedirs(self.output_path, exist_ok=True)

        source_train_flow = make_generators_from_directory(f"{args.data_path}/{source}/patches/colour",
                                                                              patchSize=512, batch_size=batchSize)
        target_train_flow = make_generators_from_directory(f"{args.data_path}/{target}/patches/colour",
                                                                              patchSize=512, batch_size=batchSize)

        self.source_data = source_train_flow
        self.target_data = target_train_flow

        saveModelArchitecture(self.output_path + '/modelsArchitectures/generators/gen_AB', self.model.g_AB)
        saveModelArchitecture(self.output_path + '/modelsArchitectures/generators/gen_BA', self.model.g_BA)
        saveModelArchitecture(self.output_path + '/modelsArchitectures/discriminators/d_A', self.model.d_A)
        saveModelArchitecture(self.output_path + '/modelsArchitectures/discriminators/d_B', self.model.d_B)
        saveModelArchitecture(self.output_path + '/modelsArchitectures/combined', self.model.combined)

    def train(self, epochs, batch_size=1, sample_interval=10):
        os.makedirs(os.path.join(self.output_path, "graphs"), exist_ok=True)
        history = SaveHistory(os.path.join(self.output_path, "graphs", "history.json"), read_existing=False)

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.model.disc_patch)
        fake = np.zeros((batch_size,) + self.model.disc_patch)

        fake_A_pool = ImagePool(50)  # the buffer stores 50 generated images
        fake_B_pool = ImagePool(50)
        for epoch in range(epochs):
            keras.backend.set_value(self.model.optimizer.lr,
                                    keras.backend.get_value(self.model.optimizer.lr) * learning_rate_decay(epoch, epochs / 2, epochs / 2))
            print(f'\nEpoch {epoch+1}/{epochs}, ' 
                  f'LR: {keras.backend.get_value(self.model.optimizer.lr)}')

            if epoch == epochs // 2:
                print('Reduced the influence of cycle-consistency to 5, identity to 2.5')
                self.model.lambda_cycle = self.model.lambda_cycle // 2
                self.model.lambda_id = self.model.lambda_id // 2

                self.model.combined.compile(loss=['mse', 'mse','mae', 'mae','mae', 'mae'],
                                            loss_weights=[1, 1, self.model.lambda_cycle, self.model.lambda_cycle,
                                                          self.model.lambda_id, self.model.lambda_id],
                                            optimizer=self.model.optimizer)

            progbar = tf.keras.utils.Progbar(target=len(self.source_data),
                                             stateful_metrics=['d_loss', 'g_loss', 'adv_loss', 'cyc_loss',
                                                               'id_loss', 'ds_loss'],
                                             verbose=1)
            d_loss_total, g_loss_total, adv_loss_total, cyc_loss_total, id_loss_total = 0.0, 0.0, 0.0, 0.0, 0.0

            for batch_i in range(len(self.source_data)):
                imgs_A, _ = self.source_data.next()
                imgs_B, _ = self.target_data.next()

                if (imgs_A.shape[0] != batch_size) or (imgs_B.shape[0] != batch_size):
                    continue

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_B = self.model.g_AB.predict(imgs_A)
                fake_A = self.model.g_BA.predict(imgs_B)

                fake_B = fake_B_pool.query(fake_B)
                fake_A = fake_A_pool.query(fake_A)

                # Train the discriminators (original images = real / translated = Fake)
                dA_input = np.vstack([imgs_A, fake_A])
                dA_output = np.vstack([valid, fake])
                dA_loss = self.model.d_A.train_on_batch(dA_input, dA_output)

                dB_input = np.vstack([imgs_B, fake_B])
                dB_output = np.vstack([valid, fake])
                dB_loss = self.model.d_B.train_on_batch(dB_input, dB_output)

                # Total disciminator loss

                d_loss = np.add(dA_loss, dB_loss)

                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.model.combined.train_on_batch([imgs_A, imgs_B],
                                                            [valid, valid, imgs_A, imgs_B, imgs_A, imgs_B])

                d_loss_total += d_loss[0]
                g_loss_total += g_loss[0]
                adv_loss_total += np.mean(g_loss[1:3])
                cyc_loss_total += np.mean(g_loss[3:5])
                id_loss_total += np.mean(g_loss[5:7])
                # Update Progbar
                progbar.update(batch_i + 1, values=[('d_loss', d_loss_total / (batch_i + 1)),
                                                    ('g_loss', g_loss_total / (batch_i + 1)),
                                                    ('adv_loss', adv_loss_total / (batch_i + 1)),
                                                    ('cyc_loss', cyc_loss_total / (batch_i + 1)),
                                                    ('id_loss', id_loss_total / (batch_i + 1))])

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(imgs_A, imgs_B, epoch, batch_i, self.output_path)

            d_loss_avg = d_loss_total / len(self.source_data)
            g_loss_avg = g_loss_total / len(self.source_data)
            adv_loss_avg = adv_loss_total / len(self.source_data)
            cyc_loss_avg = cyc_loss_total / len(self.source_data)
            id_loss_avg = id_loss_total / len(self.source_data)

            update_train_history(history, epoch + 1, d_loss_avg, g_loss_avg, adv_loss_avg,
                                 cyc_loss_avg, id_loss_avg)

        print(f"Saving model at epoch: {epoch}")
        filePath = self.output_path + '/models/' + str(epoch)
        os.makedirs(filePath, exist_ok=True)
        self.model.combined.save_weights(filePath + '/combined.h5')
        print('\n')


    def sample_images(self, img_A, img_B, epoch, step, saveDir):
        os.makedirs(saveDir, exist_ok=True)

        r, c = 2, 4
        # Translate images to the other domain
        fake_B = self.model.g_AB.predict(img_A)
        fake_A = self.model.g_BA.predict(img_B)
        # Translate images back to original domain
        fake_B_noisy = additive_gaussian_noise_layer(x=fake_B, std_dev=self.noise_std_dev)
        cyc_A = self.model.g_BA(fake_B_noisy)
        fake_A_noisy = additive_gaussian_noise_layer(x=fake_A, std_dev=self.noise_std_dev)
        cyc_B = self.model.g_AB(fake_A_noisy)

        id_A = self.model.g_BA.predict(img_A)
        id_B = self.model.g_AB.predict(img_B)

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
    parser = argparse.ArgumentParser(description='Train CycleGAN + Gaussian Noise)')

    parser.add_argument("-sd", "--source_domain", type=str, default="02")
    parser.add_argument("-td", "--target_domain", type=str, default="03")
    parser.add_argument("-ih", "--image_height", type=int, default=512)
    parser.add_argument("-iw", "--image_width", type=int, default=512)
    parser.add_argument("-ic", "--image_channel", type=int, default=3)
    parser.add_argument("-bs", "--batch_size", type=int, default=1)
    parser.add_argument("-norm", "--normalisation", type=str, default="instance")
    parser.add_argument("-ne", "--num_epochs", type=int, default=50)
    parser.add_argument("-si", "--save_interval", type=int, default=1000)
    parser.add_argument("-stddev", "--noise_std_dev", type=float, default=0.0125,
                        help="[0.0125, 0.025, 0.050, 0.075, 0.1, 0.2, 0.3, 0.5 0.9]")
    parser.add_argument("-dp", "--data_path", type=str, default=os.path.join(os.path.expanduser('~'), "data"))
    parser.add_argument("-op", "--output_path", type=str)
    parser.add_argument("-rep", "--repetition", type=str, default="rep1")

    arguments = parser.parse_args()
    print('input args:\n', json.dumps(vars(arguments), indent=4, separators=(',', ':')))  # pretty print args

    start = datetime.datetime.now()

    cyclegan_with_gaussian_noise_model = Training(arguments)
    cyclegan_with_gaussian_noise_model.train(arguments.num_epochs, arguments.batch_size, arguments.save_interval)

    print('\nTraining Time: {}'.format(datetime.datetime.now() - start))
