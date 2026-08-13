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
from Bio.SeqUtils import IUPACData

WORKER = 8
DOWNLOAD_DIR = "pdbs/"
AMINO_ACIDS = [aa.upper() for aa in IUPACData.protein_letters_3to1]

WHITELIST = AMINO_ACIDS + ["HOH"]   # Download structure with protein and water
#WHITELIST = AMINO_ACIDS   # Download only protein, no water no ligands
#WHITELIST = None   # Download everything, including protein, water and ligands

all_training_pdbs = pd.read_csv("all_training.csv")
all_testing_pdbs = pd.read_csv("all_testing.csv")
all_pdbs = pd.concat([all_training_pdbs , all_testing_pdbs] , ignore_index=True)


# Make sure there is a folder to store PDBS
if not os.path.exists(DOWNLOAD_DIR):
    os.mkdir(DOWNLOAD_DIR)
if not os.path.exists(DOWNLOAD_DIR + "train/"):
    os.mkdir(DOWNLOAD_DIR + "train/")
if not os.path.exists(DOWNLOAD_DIR + "test/"):
    os.mkdir(DOWNLOAD_DIR + "test/")

def download_range(idxes):
    '''
    Download PDB files for the given indices in all training PDBS
    '''
    for i in idxes:
        pdb_id = all_pdbs.loc[i , "PDB_ID"]
        chain = all_pdbs.loc[i , "chain_ID"]
        # Do not download if file already exists
        if not os.path.exists(DOWNLOAD_DIR + pdb_id + chain + ".pdb"):
            toolbox.download_pdb(pdb_id , chain , DOWNLOAD_DIR , True, WHITELIST)

pool = multiprocessing.Pool(WORKER)
chunk_length = math.ceil(len(all_pdbs) / WORKER)
indices = [range(n * chunk_length , (n + 1) * chunk_length) for n in range(WORKER - 1)] + [range((WORKER - 1) * chunk_length , len(all_pdbs))]
pool.map(download_range , indices)

# Move train PDBs and test PDBs into separate folders and check downloaded file numbers
for i in range(len(all_training_pdbs)):
    pdb_id = all_training_pdbs.loc[i , "PDB_ID"]
    chain = all_training_pdbs.loc[i , "chain_ID"]
    os.rename(DOWNLOAD_DIR + pdb_id + chain + ".pdb" , DOWNLOAD_DIR + "train/" + pdb_id + chain + ".pdb")
if len(os.listdir(DOWNLOAD_DIR + "train/")) == len(all_training_pdbs):
    print("All train PDB files created!")
else:
    print("Train PDB number check failed!")

for i in range(len(all_testing_pdbs)):
    pdb_id = all_testing_pdbs.loc[i , "PDB_ID"]
    chain = all_testing_pdbs.loc[i , "chain_ID"]
    os.rename(DOWNLOAD_DIR + pdb_id + chain + ".pdb" , DOWNLOAD_DIR + "test/" + pdb_id + chain + ".pdb")
if len(os.listdir(DOWNLOAD_DIR + "test/")) == len(all_testing_pdbs):
    print("All test PDB files created!")
else:
    print("Test PDB number check failed!")
