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

from data_loading import generate_TFRecord
from training import ensembleTrainer
from prediction import dfmPredictor



def process_directory(data_dir, config, mesh_type):
    
    ## Parameter extraction
    num_signatures = sum(config["number_{}".format(s)] for s in ["hks", "wks", "gaussian"])
    num_vertices = int(0.95 * config["number_vertices"]) ## because decimate may go a few under
    number_epochs = config["deep_functional_maps"]["epochs"]
    lr, bs = [config["deep_functional_maps"][k] for k in ["learning_rate", "batch_size"]]
    
    ## Data preparation
    generate_TFRecord(data_dir, num_vertices, mesh_type)

    ## Training
    fin = os.path.join(cur_dir, "data", "{}.tfrecords".format(mesh_type))
    trainer = ensembleTrainer(tf_record_file=fin, num_sigs=num_signatures, lr=lr, bs=bs)
    trainer.train(number_epochs, mesh_type)
    
    ## Optimise signature functions
    predictor = dfmPredictor(mesh_type, num_signatures)
    dir_sigs = os.path.join(data_dir, "signatures")
    files = os.listdir(dir_sigs)
    for fn in files:
        fpath = os.path.join(dir_sigs, fn)
        x_raw = np.load(fpath)
        x_new = predictor(x_raw)
        np.save(fpath, x_new)
    
    return
                


if __name__ == "__main__":
    pass
