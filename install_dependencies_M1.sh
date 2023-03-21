#!/bin/bash

##########################################################################################
#
# This Python repository requires `pyscf`,
#   which is a bit of a pain to install on Mac M1 processors.
#
# This script runs through the steps to set up an Anaconda environment with `pyscf`,
#   `openfermionpyscf` (the other main dependency of this module),
#   and `ipython` (which is the interface through which I typically use this module).
#
# Pre-requisites:
#   - You are using a Mac M1 processor.
#   - You have `anaconda` installed already.
#   - You don't already have a conda environment called `pyscf`.
#
# If you aren't using an M1 processor, you can skip a few steps, pointed out below.
# I don't think anything bad happens if you just run the script,
#   but you will be using unnecessarily older versions of code.
#
##########################################################################################

# CREATE THE CONDA ENVIRONMENT
conda create -n pyscf python=3.7    # CREATE THE ENVIRONMENT, INITIALIZING WITH PYTHON 3.7

    # NOTE: On the M1 processor, the Python version MUST be specified to 3.7.
    #       Moreover, it MUST be specified WHEN you create the environment,
    #           because (unlike specifcying a version for any other package)
    #           `conda` uses the python version for additional system configuration.
    #       If you're not on an M1 processor, you don't need that extra configuration,
    #           so you can set up your environment any way you like.

conda activate pyscf                # ACTIVATE THE NEW ENVIRONMENT

# INSTALL `pyscf`
conda install -c pyscf pyscf        # INSTALL `pyscf` FROM THE `pyscf` CHANNEL

    # NOTE: Because we have specified Python 3.7,
    #           this line will automatically install pyscf 1.6.3,
    #           which is the last version compatible with M1 processors.

conda install h5py=2.10.0           # DOWNGRADE A DEPENDENCY THAT DOESN'T WORK WITH

    # NOTE: `h5py` is automatically installed when you install `pyscf`,
    #           but the `pyscf` package metadata doesn't realize
    #           the newest version of `h5py` is incompatible with pyscf 1.6.3.
    #       This line is only necessary for M1 processors.

# INSTALL ADDITIONAL DEPENDENCIES FOR THIS REPOSITORY
pip install openfermionpyscf        # Google's pyscf interface for openfermion.
conda install ipython               # A better python shell. Not strictly necessary.