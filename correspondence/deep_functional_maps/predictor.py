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
        self.func = ops.residualNet(7, num, training = training)
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

    def __call__(self, mesh):
        s = mesh.scalars['signatures'][np.newaxis].astype(np.float32)
        mesh.scalars['signatures'] = self.func(s)[0].numpy()
        return


if __name__ == "__main__":
    pass
