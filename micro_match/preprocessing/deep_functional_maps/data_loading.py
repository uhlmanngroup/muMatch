import os
import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf


"""====================================================================================="""
"""                                       Training                                      """
"""====================================================================================="""


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def pruneIndices(number, target):
    return np.linspace(0, number, target, endpoint=False).astype(np.int)


def stripFileType(s):
    s = s.split(".npy")[0]
    return s


def generate_TFRecord(dir_in, N, output_name):
    tfrecords_filename = os.path.join(dir_in, f"{output_name}.tfrecords")
    writer = tf.io.TFRecordWriter(tfrecords_filename)
    names = [
        os.path.splitext(f)[0]
        for f in os.listdir(os.path.join(dir_in, "signatures"))
    ]

    for fn in names:
        # try:
        s = np.load(os.path.join(dir_in, "signatures", fn + ".npy")).astype(
            np.float32
        )
        eigen = np.load(os.path.join(dir_in, "eigen", fn + ".npz"))
        e, e_t = [eigen[k].astype(np.float32) for k in ["evecs", "evecs_t"]]
        g = np.load(
            os.path.join(dir_in, "geodesic_matrices", fn + ".npy")
        ).astype(np.float32)
        i = pruneIndices(e.shape[0], N)
        g /= np.mean(g)

        feature = {
            "evecs": _bytes_feature(e[i].tobytes()),
            "evecs_t": _bytes_feature(e_t[:, i].tobytes()),
            "metric": _bytes_feature(g[i][:, i].tobytes()),
            "sigs": _bytes_feature(s[i].tobytes()),
            "N_eigs": _int64_feature(e.shape[-1]),
            "N_vert": _int64_feature(N),
            "N_sigs": _int64_feature(s.shape[-1]),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()
    return


if __name__ == "__main__":
    pass
