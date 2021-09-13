# The µMatch Pipeline

"""
## Introduction
This script demonstrates how to run the µMatch pipeline from start to end on a pair of meshes.
"""
import os
import sys
cur_dir = os.getcwd()

"""
## Parameter file
If you wish to experiment with any of the default parameters (e.g., number of mesh vertices), open the parameters.yml file and adjust the relevant parameter values. This file is read in the following code block.
"""
import yaml
config: dict
with open("parameters.yml", "r") as file:
    config = yaml.safe_load(file)

"""
## Preprocessing

* raw_dir should point to the folder containing the meshes that we wish to process.
* data_dir should point to where we want the processed data to be stored for later use.
"""
from preprocessing.preprocess import batch_preprocess

raw_dir  = os.path.join(cur_dir, "example_data", "raw")
data_dir = os.path.join(cur_dir, "example_data", "processed_data")

batch_preprocess(raw_dir, data_dir, config["preprocessing"])

"""
## Mesh correspondence

Here we compute the actual correspondences between input meshes. A functor matching_functional is created and a match between a pair of meshes is computed by calling it with their respective filenames. There are two ways of doing this:

* A minimal way is to identify a template mesh and compute the correspondences between this and all other meshes (this is what is done in the code block below).
* A more advanced way is to compute all pairwise correspondences. This will take much longer, but then improvement schemes can be used subsequently to increase the qualtity of the correspondences.
"""
from correspondence.match import Match

data_dir = os.path.join(cur_dir, "example_data", "processed_data")
match_dir = os.path.join(cur_dir, "example_data", "match_results")

matching_functional = Match(dir_in=data_dir, dir_out=match_dir, config=config["correspondence"], display_result=False)
matching_functional("Q02", "Q03")
matching_functional("Q04", "Q03")

"""
## Shape Analysis

This is an example of the kind of analysis one can do once correspondences have been established. 

* The meshes are aligned to yield a set of aligned point clouds (whose points have also been placed in a one-to-one correspondence). 
* A procrustes analysis is done to retrieve the differences (deviations) at each point from the average mesh.
* Finally, these deviations are fed into a function to extract the principal components of the deviations and the resulting two first PC are plotted.
"""
import shape_analysis.shape_alignment as align
from shape_analysis.statistical_shape_analysis import collection_deviation, clustering_analysis

names, point_clouds = align.process_directory(data_dir, match_dir, display=True)
deviations = collection_deviation(point_clouds, iterations = 10)
classes = ['Specie Q04', 'Specie Q02', 'Specie Q03']
clustering_analysis(classes, deviations, variant = "TEETH dataset")

