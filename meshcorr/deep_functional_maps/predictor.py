## Standard Lib Imports ##
import os
import sys
import math
import datetime

import numpy as np

## tensorflow ##
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf


cur_dir = os.path.dirname(__file__)
sys.path.insert(1, cur_dir)

import operations as ops
import data_loading as dl

'''====================================================================================='''
'''                                       Training                                      '''
'''====================================================================================='''


class dfmPredictorBase:
    def __init__(self, num, type, training):
        self.type = type
        self.func = ops.correspondenceMatrix(num, training = training)
        weight_path = os.path.join(cur_dir, 'checkpoints', type)
        try:
            self.func.load_weights(weight_path)
        except:
            print("weights not fund.")

    def __call__(self, src, dst):
        raise NotImplementedError("base __call__ called")


class dfmPredictor(dfmPredictorBase):
    def __init__(self, num, type):
        super().__init__(num, type, False)

    def __call__(self, src, dst):
        s = [m.scalars['signatures'][np.newaxis].astype(np.float32) for m in (src, dst)]
        et = [np.transpose(m.mass @ m.eigen[1])[np.newaxis].astype(np.float32) for m in (src, dst)]
        C =  self.func(s, et)
        return C.numpy()[0]


class dfmEnsemblePredictor(dfmPredictorBase):
    def __init__(self, num, type):
        super().__init__(num, type, False)

    def __call__(self, sources, targets):
        assert(len(sources) == len(targets))
        s_x = np.stack([m.scalars['signatures'] for m in sources], axis=0).astype(np.float32)
        s_y = np.stack([m.scalars['signatures'] for m in targets], axis=0).astype(np.float32)
        et_x = np.stack([np.transpose(m.mass @ m.eigen[1]) for m in sources], axis=0).astype(np.float32)
        et_y = np.stack([np.transpose(m.mass @ m.eigen[1]) for m in targets], axis=0).astype(np.float32)
        C   =  self.func([s_x,s_y],[et_x,et_y])
        return C.numpy()


def extract_variables(mesh):
    s = mesh.scalars['signatures']
    g = mesh.g
    e = mesh.eigen[1]
    et = np.transpose(mesh.mass @ mesh.eigen[1])
    vars = [x[np.newaxis].astype(np.float32) for x in (e, et,s,g)]
    return vars


class dfmPredictorwithOptimisation(dfmPredictorBase):

    def __init__(self, num, type, lr = 1e-3, max_steps = 100):
        super().__init__(num, type, True)
        self.optimiser = tf.keras.optimizers.Adam(learning_rate = lr)
        self.max_steps = max_steps

    def train_step(self, x, y):
        e_x, et_x, s_x, g_x = x
        e_y, et_y, s_y, g_y = y
        with tf.GradientTape() as tape:
            C =  self.func([s_x, s_y], [et_x, et_y])
            P = ops.softCorrespondenceEnsemble(C, et_x, e_y)
            loss = ops.geodesicErrorEnsemble(P, g_x, g_y)
            grads = tape.gradient(loss, self.func.trainable_variables)
            self.optimiser.apply_gradients(zip(grads, self.func.trainable_variables))
        result = [x.numpy() for x in [loss,C]]
        return result

    def optimise(self, x, y):
        l_opt = np.inf
        C_opt: np.ndarray
        for n in range(self.max_steps):
            loss,C = self.train_step(x,y)
            if (loss < l_opt):
                l_opt = loss
                C_opt = C
        return C[0]

    def __call__(self, src, dst):
        x,y = [extract_variables(m) for m in (src,dst)]
        C = self.optimise(x,y)
        return C



if __name__ == "__main__":
    pass
