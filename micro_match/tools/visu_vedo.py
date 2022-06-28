#! /usr/bin/env python

import argparse
import os

import vedo as vp

__doc__ = "Visualise meshes put in correspondence"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, prog="visualise_interactive"
    )

    parser.add_argument("m1", help="Source mesh")
    parser.add_argument("m2", help="Target mesh")

    args = parser.parse_args()

    m1 = vp.load(args.m1).normalize().lineWidth(0.1)
    m2 = vp.load(args.m2).normalize().lineWidth(0.1)

    # Color mesh using mesh indices
    scals = list(range(m1.NPoints()))
    m1.addPointArray(scals, "vertices_indices")
    scals = list(range(m2.NPoints()))
    m2.addPointArray(scals, "vertices_indices")

    base_name_m1 = os.path.basename(args.m1)
    base_name_m2 = os.path.basename(args.m2)

    plt = vp.Plotter(
        shape=[1, 2], size="full", title=f"{base_name_m1} vs. {base_name_m2}"
    )
    print("Press [Space] to continue")
    plt.show(m1, at=0)
    plt.show(m2, at=1, interactive=True)
    plt.close()
