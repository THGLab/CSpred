#!/usr/bin/env python
# Script for downloading all the single-chain pdb files for the training and add hydrogens to the structures

# Author: Jie Li
# Date created: Oct 13, 2019

import os
import sys
sys.path.append("../")
import toolbox
import pandas as pd
import multiprocessing
import math

WORKER = 8
DOWNLOAD_DIR = "pdbs"

all_pdbs = pd.read_csv("all_training.csv")

# Make sure there is a folder to store all PDBS
if not os.path.exists(DOWNLOAD_DIR):
    os.mkdir(DOWNLOAD_DIR)

def download_range(idxes):
    '''
    Download PDB files for the given indices in all training PDBS
    '''
    for i in idxes:
        pdb_id = all_pdbs.loc[i , "PDB_ID"]
        chain = all_pdbs.loc[i , "chain_ID"]
        toolbox.download_pdb(pdb_id , chain , "pdbs" , True)

pool = multiprocessing.Pool(WORKER)
chunk_length = math.ceil(len(all_pdbs) / WORKER)
indices = [range(n * chunk_length , (n + 1) * chunk_length) for n in range(WORKER - 1)] + [range((WORKER - 1) * chunk_length , len(all_pdbs))]
pool.map(download_range , indices)

# Check downloaded file numbers
if len(os.listdir(DOWNLOAD_DIR)) == len(all_pdbs):
    print("All PDB files created!")
else:
    print("Number check failed!")
