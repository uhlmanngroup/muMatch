#! /usr/bin/env python

from setuptools import setup

VERSION = "1.0"
AUTHOR = "James Klatzow, Virginie Uhlmann"
AUTHOR_EMAIL = "uhlmann@ebi.ac.uk"


setup(
    name="microMatch",
    version=VERSION,
    description="3D shape correspondence for microscopy data",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    packages=[
        "mumatch",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved",
        "Programming Language :: C",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: " "Implementation :: CPython",
    ],
    url="https://github.com/uhlmanngroup/muMatch",
    python_requires=">=3.6",
)
