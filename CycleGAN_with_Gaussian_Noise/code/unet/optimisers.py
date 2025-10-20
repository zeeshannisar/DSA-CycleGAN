from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras import backend as K
import numpy as np


class Adam_accumulate(Optimizer):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, accum_iters=20, **kwargs):
        super(Adam_accumulate, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0)
        self.lr = K.variable(lr)
        self.beta_1 = K.variable(beta_1)
        self.beta_2 = K.variable(beta_2)
        self.accum_iters = K.variable(accum_iters)

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [(self.iterations, self.iterations + 1)]

        t = self.iterations + 1
        lr_t = self.lr * K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t))

        ms = [K.variable(np.zeros(K.get_value(p).shape)) for p in params]
        vs = [K.variable(np.zeros(K.get_value(p).shape)) for p in params]
        gs = [K.variable(np.zeros(K.get_value(p).shape)) for p in params]
        self.weights = ms + vs

        for p, g, m, v, gg in zip(params, grads, ms, vs, gs):

            flag = K.equal(self.iterations % self.accum_iters, 0)
            flag = K.cast(flag, dtype='float32')

            gg_t = (1 - flag) * (gg + g)
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * (gg + flag * g) / self.accum_iters
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square((gg + flag * g) / self.accum_iters)
            p_t = p - flag * lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append((m, flag * m_t + (1 - flag) * m))
            self.updates.append((v, flag * v_t + (1 - flag) * v))
            self.updates.append((gg, gg_t))

            new_p = p_t
            # apply constraints
            if getattr(p, 'constraint', None) is not None:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append((p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'epsilon': self.epsilon}
        base_config = super(Adam_accumulate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))