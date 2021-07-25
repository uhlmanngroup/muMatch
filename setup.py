#! /usr/bin/env python

from setuptools import setup

VERSION = "0.1.0"
AUTHOR = "James Klatzow"
AUTHOR_EMAIL = "jklatzow@ebi.ac.uk"


setup(
    name="meshcorr",
    version=VERSION,
    description="Library to include correspondence techniques for a variety of mesh types.",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    packages=[
        "meshcorr",
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
    url="https://gitlab.ebi.ac.uk/jklatzow/mesh_correspondence_library",
    python_requires=">=3.6",
)
