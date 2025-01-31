import json
import numpy as np
import tensorflow as tf


class SaveHistory(tf.keras.callbacks.Callback):

    def __init__(self, history_path, read_existing=False):

        self.history_path = history_path

        if read_existing:
            print('Loading existing .json history')
            with open(history_path, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {}

    def on_train_begin(self):
        return self.history

    def on_epoch_end(self, logs={}):
        for k in logs:
            if k in self.history:
                self.history[k].append(logs[k])
            else:
                self.history[k] = [logs[k]]

        with open(self.history_path, 'w') as f:
            json.dump(self.history, f, indent=4, cls=NumpyFloatValuesEncoder)


class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def update_train_history(history_file, epoch, d_loss, g_loss, adv_loss, cyc_loss, id_loss, ds_loss):
    history_file.on_epoch_end(logs={"epoch": epoch, "d_loss": d_loss, "g_loss": g_loss, "adv_loss": adv_loss,
                                    "cyc_loss": cyc_loss, "id_loss": id_loss, "ds_loss": ds_loss})
