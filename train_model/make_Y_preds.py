#/usr/bin/env python
# Script for making UCBShift-Y predictions on the training data using testing mode

# Author: Jie Li
# Date created: Oct 14, 2019

import sys
import os
sys.path.append("../")
import ucbshifty

PDB_FOLDER = "pdbs/"
PRED_FOLDER = "Y_preds/"
if not os.path.exists(PRED_FOLDER):
    os.mkdir(PRED_FOLDER)

for pdb in os.listdir(PDB_FOLDER + "train/"):
    result = ucbshifty.main(PDB_FOLDER + "train/" + pdb, strict=1, secondary=False, exclude=True) # Exact matches will be excluded in order to reflect more realistic scenarios 
    result.to_csv(PRED_FOLDER + pdb.replace(".pdb", ".csv"), index=None)

print("Finished generating predictions for training.")
