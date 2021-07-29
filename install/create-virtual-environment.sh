#!/bin/bash 
# Copyright (c) 2021 The Center for Biomolecular NMR Data Processing and Analysis
# License: https://opensource.org/licenses/MIT
PYTHONVERS=3.8

ORIGIN=$(dirname $(readlink -f $0))
PYTHON=python$PYTHONVERS
VENV=$ORIGIN/../venv
MODELS=$ORIGIN/../models
$PYTHON -m venv $VENV 
$VENV/bin/pip install -U pip
$VENV/bin/pip install -r $ORIGIN/requirements.txt
ln -s $ORIGIN/../bins/mkdssp $VENV/bin/
ln -s $ORIGIN/../bins/mTM-align/mTM-align $VENV/bin/
ln -s $ORIGIN/../bins/ncbi-blast-2.9.0+/bin/blastp $VENV/bin/
ln -s $ORIGIN/../bins/ncbi-blast-2.9.0+/bin/makeblastdb $VENV/bin/
if [ ! -d $MODELS ]; then
	mkdir $MODELS
	echo "Installing February 14, 2020 models from datadryad"
	URL=https://datadryad.org/stash/downloads/file_stream/242856 
	curl --location $URL |tar -xz -o -C $MODELS 
fi
