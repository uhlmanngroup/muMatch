import os

import numpy as np

from .data_loading import generate_TFRecord
from .prediction import dfmPredictor
from .training import ensembleTrainer


def process_directory(data_dir, config, mesh_type):

    # Parameter extraction
    num_signatures = sum(
        config[f"number_{s}"] for s in ["hks", "wks", "gaussian"]
    )
    num_vertices = int(
        0.95 * config["number_vertices"]
    )  # because decimate may go a few under
    number_epochs = config["deep_functional_maps"]["epochs"]
    lr, bs = [
        config["deep_functional_maps"][k]
        for k in ["learning_rate", "batch_size"]
    ]

    # Data preparation
    generate_TFRecord(data_dir, num_vertices, mesh_type)

    # Training
    cur_dir = os.path.dirname(
        __file__
    )  # TODO: Data needs to be stored outside of python package.
    fin = os.path.join(cur_dir, "data", f"{mesh_type}.tfrecords")
    trainer = ensembleTrainer(
        tf_record_file=fin, num_sigs=num_signatures, lr=lr, bs=bs
    )
    trainer.train(number_epochs, mesh_type)

    # Optimise signature functions
    predictor = dfmPredictor(mesh_type, num_signatures)
    dir_sigs = os.path.join(data_dir, "signatures")
    files = os.listdir(dir_sigs)
    for fn in files:
        fpath = os.path.join(dir_sigs, fn)
        x_raw = np.load(fpath)
        x_new = predictor(x_raw)
        np.save(fpath, x_new)

    return
