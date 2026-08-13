#!/usr/bin/env python
# This is the script for evaluating the performance of the UCBShift model on the testing dataset. It uses the testing mode of UCBShift-Y so that no sequence matches with >99% identity will be considered, which reflects some more realistic situation.

# Author: Jie Li
# Date created: Oct 16, 2019

import os
import pandas as pd
import sys
sys.path.append("../")
import toolbox
import numpy as np
from spartap_features import rc_ala
pd.options.mode.chained_assignment = None  # default='warn'


PRED_PATH = "UCBShift_preds/"
PDB_PATH = "pdbs/test/"
ANALYSIS_PATH = "analysis/"
SHIFTS_PATH = "test_targets/"
PRED_SCRIPT = "../CSpred.py"


def evaluate_final_pred(pdbid, pred_path, real_path):
    '''
    Compare predictions made by UCBShift to real target values

    args:
        pdbid = 5-digit PDB ID that is analyzed
        pred_path = Directory path to all the prediction files
        real_path = Directory path to all the real target files

    returns:
        List of length 3. Each element of the list is a tuple of length 2, recording all the predictions and errors, and statistical analysis of the predictions for this specific proteins. The three elements in the list correspond to X model, Y model and UCBShift model.
    '''
    pred = pd.read_csv(pred_path + pdbid + ".csv")
    real = pd.read_csv(real_path + pdbid + ".csv")
    real.rename(columns={"RES_NUM": "RESNUM"}, inplace=True)
    merged = pd.merge(pred, real, on="RESNUM")
    invalid = merged.RESNAME_y != merged.RESNAME_x
    merged.loc[invalid, toolbox.ATOMS] = np.nan
    # Exclude crazy outliers

    hydrogenatoms = ['HB', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE21', 'HE22', 'HG', 'HG1', 'HG12', 'HG13', 'HG2', 'HG3', 'HZ','HE3', 'HZ3' ,'HH2', 'HZ2', 'HA', 'H']
    carbonatoms = ['CG','CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CG1', 'CG2', 'CZ', 'CE3', 'CZ3', 'CH2', 'CA', 'CB', 'C', 'CZ2']



    for atom in carbonatoms:
        for aa in range(len(toolbox.AMINOACIDS)):
            if not np.isnan(rc_ala[atom][aa]):
                selected_rows = merged.loc[merged['RESNAME_x'] == toolbox.AMINOACIDS[aa]]
                selected_rows.loc[(selected_rows[atom ] > rc_ala[atom][aa]+12), atom] = np.nan
                selected_rows.loc[(selected_rows[atom] < rc_ala[atom][aa]-12), atom] = np.nan
                merged.loc[merged['RESNAME_x'] == toolbox.AMINOACIDS[aa], atom] = selected_rows.loc[selected_rows['RESNAME_x'] == toolbox.AMINOACIDS[aa], atom].values

                


    for atom in hydrogenatoms:
        for aa in range(len(toolbox.AMINOACIDS)):
            if not np.isnan(rc_ala[atom][aa]):
                selected_rows = merged.loc[merged['RESNAME_x'] == toolbox.AMINOACIDS[aa]]
                selected_rows.loc[(selected_rows[atom] > rc_ala[atom][aa]+6), atom] = np.nan
                selected_rows.loc[(selected_rows[atom] < rc_ala[atom][aa]-6), atom] = np.nan
                merged.loc[merged['RESNAME_x'] == toolbox.AMINOACIDS[aa], atom] = selected_rows.loc[selected_rows['RESNAME_x'] == toolbox.AMINOACIDS[aa], atom].values




    merged["PDBID"] = pdbid
    results = []
    for model in ["X", "Y", "UCBShift"]:

        # Prepare the model analysis dataframe for the specific model
        model_analysis = merged[["PDBID", "RESNUM", "RESNAME_x"] + [atom + "_" + model for atom in toolbox.ATOMS] + toolbox.ATOMS]
        rename_cols = {atom + "_" + model: atom + "_PRED" for atom in toolbox.ATOMS}
        rename_cols.update({"RESNAME_x": "RESNAME"})
        model_analysis.rename(columns=rename_cols, inplace=True)
        model_err = dict()
        for atom in toolbox.ATOMS:
            err = model_analysis[atom + "_PRED"] - model_analysis[atom]
            model_analysis[atom + "_DIFF"] = err
            model_err[atom + "_RMSE"] = toolbox.rmse(err)
            model_err[atom + "_ERR_MIN"] = np.abs(err).min()
            model_err[atom + "_ERR_MAX"] = np.abs(err).max()

            
        results.append((model_analysis, {pdbid: model_err}))

    return results

# Make sure the folder for storing prediction files and analysis is empty
for PATH in [PRED_PATH, ANALYSIS_PATH]:
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    else:
        for file in os.listdir(PATH):
            os.remove(PATH + file)

# Prepare input files
with open("inputs", "w") as f:
    for file in os.listdir(PDB_PATH):
        f.write(PDB_PATH + file + "\n")

# Make all predictions by executing the script with batch mode and test mode
os.system("python " + PRED_SCRIPT + " inputs -b -t -o " + PRED_PATH)

print("All predictions have been made. Now analyzing...")




preds = dict()
errors = dict()
for model in ["X", "Y", "UCBShift"]:
    preds[model] = []
    errors[model] = dict()
for file in os.listdir(PRED_PATH):
    pdbid = file.replace(".csv", "")
    for model, result in zip(["X", "Y", "UCBShift"], evaluate_final_pred(pdbid, PRED_PATH, SHIFTS_PATH)):
        preds[model].append(result[0])
        errors[model].update(result[1])

for model in ["X", "Y", "UCBShift"]:
    print("Model:", model)
    all_preds = pd.concat(preds[model])
    
    for atom in ['C','CA','CB','CG','CD1','CG2','CD','CG1','CD2','CE','CE1','CE2','CZ','CZ2','CH2','CE3','CZ3','H','HA','HB2','HB3','HG2','HB','HD2','HG3','HD1','HD3','HE2','HG','HE1','HE3','HG12','HG13','HG1','HD21','HD22','HE','HE21','HE22','HZ','HZ2','HH2','HZ3','N','ND2','NE2','NE1']:
        print(atom + "_RMSE:", toolbox.rmse(all_preds[atom + "_DIFF"]))
        valid = all_preds[atom + "_DIFF"].notnull()
        print(atom + "_CORR:", np.corrcoef(all_preds[valid][atom + "_PRED"], all_preds[valid][atom])[0,1])
        print("\n")
        

    print("-" * 25, end="\n")
    all_preds.to_csv(ANALYSIS_PATH + model + "_preds.csv", index=None)
    all_err = pd.DataFrame.from_dict(errors[model], orient="index")
    all_err.to_csv(ANALYSIS_PATH + model + "_errors.csv")

print("All analyses have finished. Results are saved at " + ANALYSIS_PATH)
