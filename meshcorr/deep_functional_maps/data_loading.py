

## Standard Lib Imports ##
import os
import sys
import math
import datetime
import igl
import numpy as np
import json

## tensorflow ##
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf

import operations as ops


'''====================================================================================='''
'''                                       Training                                      '''
'''====================================================================================='''



def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def pruneIndices(number, target):
    return np.linspace(0, number, target, endpoint = False).astype(np.int)


def stripFileType(s):
    s = s.split('.npy')[0]
    return s


#targs = ["11_DUSP6_11.5_WT_RFL", "14_DUSP6_10.5_WT_LFL", "10_DUSP6_10.5_WT_LFL", "11_DUSP6_10.5_WT_RFL",
#         "10_DUSP6_11.5_WT_LFL",  "8_DUSP6_11.5_WT_RFL", "9_DUSP6_10.5_WT_RFL", "9_DUSP6_10.5_WT_LFL",
#         "14_DUSP6_10.5_WT_RFL",  "9_DUSP6_11.5_WT_RFL", "8_DUSP6_11.5_WT_LFL", "11_DUSP6_10.5_WT_LFL",
#         "10_DUSP6_11.5_WT_RFL"]

#targs = ["9_DUSP6_11.5_WT_RHL", "11_DUSP6_11.5_WT_LHL", "8_DUSP6_11.5_WT_LHL", "12_DUSP6_10.5_WT_LHL", "10_DUSP6_10.5_WT_LHL",
#         "13_DUSP6_10.5_WT_RHL", "14_DUSP6_10.5_WT_LHL", "8_DUSP6_11.5_WT_RHL", "10_DUSP6_10.5_WT_RHL", "9_DUSP6_11.5_WT_LHL"]

#targs = ["6_DUSP6_10.5_WT_LFL", "8_DUSP6_10.5_WT_RFL", "5_DUSP6_10.5_WT_RFL", "6_DUSP6_10.5_WT_RFL",
#         "7_DUSP6_10.5_WT_RFL", "5_DUSP6_10.5_WT_LFL", "7_DUSP6_10.5_WT_LFL"]


def generate_TFRecord(dir, target_dir, type):

    tfrecords_filename = type + ".tfrecords"
    writer = tf.io.TFRecordWriter(tfrecords_filename)

    fjson = os.path.join(dir, "configs", "limbs.json")
    with open(fjson) as file:
        config = json.load(file)

    target =  config["N"]

    files = [f.split('.npy')[0] for f in os.listdir(dir + '/partitions')]
    targs = [f.split('.stl')[0] for f in os.listdir(target_dir)]
    names = [x for x in files if x in targs]

    for fn in names:
        #try:
        s   =  np.load(os.path.join(dir, 'descriptors', fn + '.npy')).astype(np.float32)
        e   =  np.load(os.path.join(dir, 'eigenvectors', fn + '.npy')).astype(np.float32)
        e_t =  np.load(os.path.join(dir, 'transposed_eigenvectors', fn + '.npy')).astype(np.float32)
        g   =  np.load(os.path.join(dir, 'geodesic_matrices', fn + '.npy')).astype(np.float32)
        i   =  pruneIndices(e.shape[0], target)
        g  /=  np.mean(g)

        partitions = np.load(os.path.join(dir, 'partitions', fn + '.npy'))[..., np.newaxis].astype(np.float32)
        s = np.concatenate([s] + [x * s for x in partitions], axis = -1)

        print(fn, s.shape, e.shape, e_t.shape, g.shape, target)

        feature = { 'evecs'  : _bytes_feature(    e[  i].tobytes() ),
                    'evecs_t': _bytes_feature(  e_t[:,i].tobytes() ),
                    'metric' : _bytes_feature( g[i][:,i].tobytes() ),
                    'sigs'   : _bytes_feature(      s[i].tobytes() ),
                    'N_eigs' : _int64_feature(         e.shape[-1] ),
                    'N_vert' : _int64_feature(              target ),
                    'N_sigs' : _int64_feature(         s.shape[-1] )}

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        #except:
        #    pass

    writer.close()
    return



if __name__ == "__main__":
    dir_in  =  '/home/jamesklatzow/Documents/EBI/Preprocess/data/limbs'
    raw_dir = '/home/jamesklatzow/Documents/EBI/Datasets/limb_source_data/Limbs/'

    for condition in ['WT', 'MUT']:
        for stage in ['Early', 'Mid', 'Late']:
            for position in ['Fore', 'Hind']:
                tar_dir = os.path.join(raw_dir, stage, position + 'limb', condition)
                variety = (stage + '_' + position + '_' + condition + '_limbs').lower()
                print(variety)
                generate_TFRecord(dir_in, tar_dir, variety)
