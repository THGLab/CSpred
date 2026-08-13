#!/usr/bin/env python3
# Script for making predictions using both the X module and the Y module

# Author: Jie Li
# Date created: Sep 20, 2019

from spartap_features import PDB_SPARTAp_DataReader
from data_prep_functions import *
import ucbshifty
import joblib
import toolbox
import os
import pandas as pd
import argparse
import multiprocessing
# import warnings
import sys
if sys.version_info.major < 3 or sys.version_info.major == 3 and sys.version_info.minor < 5:
    raise ValueError("Python >= 3.5 required")


# Suppress Setting With Copy warnings
pd.options.mode.chained_assignment = None

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
ML_MODEL_PATH = SCRIPT_PATH + "/models/"

def build_input(pdb_file_name, pH=5, rcfeats=True, hse=True, hbrad=[5.0] * 3):
    '''
    Function for building dataframe for the specified pdb file.
    Returns a pandas dataframe for a specific single-chain pdb file.

    args:
        pdb_file_name = The path to the PDB file for prediction (str)
        pH = pH value to be considered
        rcfeats = Include a column for random coil chemical shifts (Bool)
        hse = Include a feature column for the half-sphere exposure (Bool)
        hbrad = Max length of hydrogen bonds for HA,HN and O (list of float)
    '''
    # Build feature reader and read all features
    feature_reader = PDB_SPARTAp_DataReader()
    pdb_data = feature_reader.df_from_file_3res(pdb_file_name,rcshifts=rcfeats, hse=hse, first_chain_only=False, sequence_columns=0,hbrad=hbrad)
    pdb_data["pH"]=pH
    return pdb_data

def data_preprocessing(data):
    '''
    Function for executing all the preprocessing steps based on the original extracted features, including fixing HA2/HA3 ring current ambiguity, adding hydrophobicity, powering features, drop unnecessary columns, etc.
    '''
    data = data.copy()
    data = data[sorted(data.columns)]
    data = ha23ambigfix(data, mode=0)
    Add_res_spec_feats(data, include_onehot=False)
    data = feat_pwr(data, hbondd_cols + cos_cols, [2])
    data = feat_pwr(data, hbondd_cols, [-1,-2,-3])
    dropped_cols = dssp_pp_cols + dssp_energy_cols + ['FILE_ID_x','FILE_ID_y','Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM',"RES", 'CHAIN', 'RESNAME_ip1', 'RESNAME_im1', 'BMRB_RES_NUM', 'RCI_S2', 'MATCHED_BMRB',"identifier"]+ rcoil_cols
    data = data.drop(set(dropped_cols) & set(data.columns), axis=1)
    return data



def prepare_data_for_atom(data,atom):
    '''
    Function to generat features data for a given atom type: meaning that the irrelevant values are removed from the dataset

    args:
        data - the dataset that contains all the features (pandas.DataFrame)
        atom - the atom to keep ring currents

    returns:
        pandas.DataFrame containing the cleaned feature set
    '''
    
    dat = data.copy()
    
    column_names = dat.columns.tolist()
    new_column_names = [name.replace('.1', '') if name.endswith('.1') else name for name in column_names]
    dat.columns = new_column_names

    
    ring_col = atom + '_RC'
    rem1 = ring_cols.copy()
    rem1.remove(ring_col)
    rem2 = [rm_atom + "_RING" for rm_atom in ['C', 'CA', 'CB', 'N', 'HA', 'HA2', 'HA3', 'H', '1H', '1HA', '2HA','CG','CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CG1', 'CG2', 'CZ','HB', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE21', 'HE22', 'HE3', 'HG', 'HG1', 'HG12', 'HG13', 'HG2', 'HG3', 'HZ','CE3','CZ3','HZ3','CH2','HH2','CZ2','HZ2', 'HB1', 'HD11', 'HD12', 'HD13', 'HD23', 'HG11', 'HZ1', 'HG21', 'HG22', 'HG23','ND2','NE1','NE2']]
    rem3 = [rm_atom + "_EFIELD" for rm_atom in ['HA2', 'HA3', 'HA', 'H', 'HB', 'HB1', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE21', 'HE22', 'HG', 'HG1', 'HG11', 'HG12', 'HG13', 'HG2', 'HG3', 'HZ','HH11','HH12', 'HD11', 'HD12', 'HD13', 'HH21','HH22', 'HD23', 'HE3','HZ3','HH2','HZ2', 'HZ1', 'HG21', 'HG22', 'HG23', 'HZ1', 'HZ2', 'HZ3',  'ND2','NE1','NE2','N'] if rm_atom != atom]
    rem4 = [rm_atom + "_dHA" for rm_atom in ['HB', 'HB1', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE21', 'HE22', 'HG', 'HG1', 'HG12', 'HG13', 'HG2', 'HG3', 'HZ', 'HD11', 'HD12', 'HD13',  'HD23', 'HE3','HZ3','HH2','HZ2', 'HZ1', 'HG21', 'HG22', 'HG23', 'HG'] if rm_atom != atom]
    rem5 = [rm_atom + "_COS_H" for rm_atom in ['HB', 'HB1', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE21', 'HE22', 'HG', 'HG1', 'HG12', 'HG13', 'HG2', 'HG3', 'HZ', 'HD11', 'HD12', 'HD13',  'HD23', 'HE3','HZ3','HH2','HZ2', 'HZ1', 'HG21', 'HG22', 'HG23', 'HG'] if rm_atom != atom]
    rem6 = [rm_atom + "_COS_A" for rm_atom in ['HB', 'HB1', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE21', 'HE22', 'HG', 'HG1', 'HG12', 'HG13', 'HG2', 'HG3', 'HZ','HD11', 'HD12', 'HD13', 'HD23', 'HE3','HZ3','HH2','HZ2', 'HZ1', 'HG21', 'HG22', 'HG23', 'HG'] if rm_atom != atom]
    rem7 = [rm_atom + "_EXISTS" for rm_atom in ['HB', 'HB1', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE21', 'HE22', 'HG', 'HG1', 'HG12', 'HG13', 'HG2', 'HG3', 'HZ','HD11', 'HD12', 'HD13','HD23', 'HE3','HZ3','HH2','HZ2', 'HZ1', 'HG21', 'HG22', 'HG23', 'HG'] if rm_atom != atom]
    rem8 = [rm_atom + "_ENERGY" for rm_atom in ['HB', 'HB1', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE21', 'HE22', 'HG', 'HG1', 'HG12', 'HG13', 'HG2', 'HG3', 'HZ', 'HD11', 'HD12', 'HD13',  'HD23', 'HE3','HZ3','HH2','HZ2', 'HZ1', 'HG21', 'HG22', 'HG23', 'HG'] if rm_atom != atom]
    dat = dat.drop(rem1 + rem2 + rem3 + rem4 + rem5 + rem6 + rem7 + rem8, axis=1, errors='ignore')

    
    hbondd_sidechain_cols = [i+j for i in ['HB', 'HB1', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE21', 'HE22', 'HG', 'HG1', 'HG12', 'HG13', 'HG2', 'HG3', 'HZ','HH11','HH12', 'HD11', 'HD12', 'HD13', 'HH21','HH22', 'HD23', 'HE3','HZ3','HH2','HZ2', 'HZ1', 'HG21', 'HG22', 'HG23']  for j in ['_dHA', '_COS_H', '_COS_A']]
    hbondd_sidechain_cols = [element for element in hbondd_sidechain_cols if element.startswith(atom + '_')]
    #add polynomial transformation of side chain hbonds
    dat = dat.loc[:, ~dat.columns.duplicated()]
    dat=feat_pwr(dat,hbondd_sidechain_cols,[-1,-2,-3])
    dat=feat_pwr(dat,hbondd_sidechain_cols,[2])    
    
    dat[ring_col] = dat[ring_col].fillna(value=0)

     
    return dat



def calc_sing_pdb(pdb_file_name,pH=5,TP=True,TP_pred=None,ML=True,test=False):
    '''
    Function for calculating chemical shifts for a single PDB file using X module / Y module / both

    args:
        pdb_file_name = The path to the PDB file for prediction (str)
        pH = pH value to be considered
        TP = Whether or not use TP module (Bool)
        TP_pred = predicted shifts dataframe from Y module. If None, generate this prediction within this function (pandas.DataFrame / None)
        ML = Whether or not use ML module (Bool)
        test = Whether or not use test mode (Exclude mode for SHIFTY++, Bool)
    '''
    if not os.path.isdir(ML_MODEL_PATH):
        raise ValueError("models not installed in {}".format(ML_MODEL_PATH))
    if pH < 2 or pH > 12:
        print("Warning! Predictions for proteins in extreme pH conditions are likely to be erroneous. Take prediction results at your own risk!")
    preds = pd.DataFrame()
    if TP:
        if TP_pred is None:
            print("Calculating UCBShift-Y predictions ...")
            # generate hash string from pdb file name
            hashed_file_name = str(hash(pdb_file_name) % ((sys.maxsize + 1) * 2)) + '/'
            TP_pred = ucbshifty.main(pdb_file_name, 1, exclude=test, custom_working_dir=hashed_file_name)
        if not ML:
            # Prepare data when only doing TP prediction
            preds = TP_pred[["RESNUM","RESNAME"]]
            for atom in toolbox.ATOMS:
                if atom+"_RC" in TP_pred.columns:
                    rc = TP_pred[atom+"_RC"]
                else:
                    rc = 0
                preds[atom+"_Y"] = TP_pred[atom] + rc
    if ML:
        print("Generating features ...")
        feats = build_input(pdb_file_name, pH)
        
        feats.rename(index=str, columns=sparta_rename_map, inplace=True) # Rename columns so that random coil columns can be correctly recognized
        
        resnames = feats["RESNAME"]
        resnums = feats["RES_NUM"]
        rcoils = feats[rcoil_cols]
        feats = data_preprocessing(feats)

        result = {"RESNUM":resnums, "RESNAME":resnames}
        for atom in toolbox.ATOMS:
            
            print("Calculating UCBShift-X predictions for %s ..." % atom)
           
           # Predictions for each atom
            atom_feats = prepare_data_for_atom(feats, atom)
            
            r0 = joblib.load(ML_MODEL_PATH + "%s_R0.sav" % atom)

            atom_feats.fillna(0, inplace=True)
            r0_pred = r0.predict(atom_feats.values)

            feats_r1 = atom_feats.copy()
            feats_r1["R0_PRED"] = r0_pred
            r1 = joblib.load(ML_MODEL_PATH + "%s_R1.sav" % atom)
           

            r1_pred = r1.predict(feats_r1.values)
            # Write ML predictions
            result[atom+"_X"] = r1_pred + rcoils["RCOIL_"+atom]

            if TP:
                print("Calculating UCBShift predictions for %s ..." % atom)
                feats_r2 = atom_feats.copy()
                feats_r2["RESNAME"] = resnames
                feats_r2["RESNUM"] = resnums
                tp_atom = TP_pred[["RESNAME", "RESNUM", atom, atom+"_BEST_REF_SCORE", atom+"_BEST_REF_COV", atom+"_BEST_REF_MATCH"]]
                feats_r2 = pd.merge(feats_r2, tp_atom, on="RESNUM", suffixes=("","_TP"), how="left")
                # Write TP predictions
                result[atom+"_Y"] = feats_r2[atom].values
                result[atom+"_BEST_REF_SCORE"] = feats_r2[atom+"_BEST_REF_SCORE"].values
                result[atom+"_BEST_REF_COV"] = feats_r2[atom+"_BEST_REF_COV"].values
                result[atom+"_BEST_REF_MATCH"] = feats_r2[atom+"_BEST_REF_MATCH"].values
                valid = (feats_r2.RESNAME == feats_r2.RESNAME_TP) & (feats_r2[atom].notnull())
                # Subtract random coils to make secondary TP shifts
                feats_r2[atom] -= rcoils["RCOIL_"+atom].values
                feats_r2["R0_PRED"] = r0_pred
                valid_feats_r2 = feats_r2.drop(["RESNAME","RESNUM","RESNAME_TP"], axis=1)[valid]
                r2_pred = r1_pred.copy()
                if len(valid_feats_r2):
                    r2 = joblib.load(ML_MODEL_PATH + "%s_R2.sav" % atom)
                    valid_feats_r2 = valid_feats_r2.fillna(0)
                    
                    r2_pred_valid = r2.predict(valid_feats_r2.values)
                    r2_pred[valid] = r2_pred_valid
                # Write final predictions
                result[atom+"_UCBShift"] = r2_pred + rcoils["RCOIL_"+atom]
        preds = pd.DataFrame(result)
    return preds

        

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='This program is an NMR chemical shift predictor for protein chemical shifts (including H, Hα, C\', Cα, Cβ and N) in aqueous solution. It has two sub-modules: one is the machine learning (X) module. It uses ensemble tree-based methods to predict chemical shifts from the features extracted from the PDB files. The second sub-module is the transfer prediction (Y) module that predicts chemical shifts by "transfering" shifts from similar proteins in the database to the query protein through structure and sequence alignments. Finally, the two parts are combined to give the UCBShift predictions.')
    args.add_argument("input", help="The query PDB file or list of PDB files for which the shifts are calculated")
    args.add_argument("--batch", "-b", action="store_true", help="If toggled, input accepts a text file specifying all the PDB files need to be calculated (Each line is a PDB file name. If pH values are specified, followed with a space)")
    args.add_argument("--output", "-o", help="Filename of generated output file. A file [shifts.csv] is generated by default. If in batch mode, you should specify the path for storing all the output files. Each output file has the same name as the input PDB file name.", default="shifts.csv")
    args.add_argument("--worker", "-w", type=int, help="Number of CPU cores to use for parallel prediction in batch mode.", default=4)
    args.add_argument("--shifty_only", "-y", "-Y", action="store_true", help="Only use UCBShift-Y (transfer prediction) module. Equivalent to executing UCBShift-Y directly with default settings")
    args.add_argument("--shiftx_only", "-x", "-X", action="store_true", help="Only use UCBShift-X (machine learning) module. No alignment results will be utilized or calculated")
    args.add_argument("--pH", "-pH", "-ph", type=float, help="pH value to be considered. Default is 5", default=5)
    args.add_argument("--test", "-t", action="store_true", help="If toggled, use test mode for UCBShift-Y prediction")
    args.add_argument("--models", help="Alternate location for models directory")
    args = args.parse_args()
    if args.models:
        if not os.path.isdir(args.models):
            raise ValueError("Directory {} specified by models does not exists".format(args.models))
        ML_MODEL_PATH = args.models
 
    if not args.batch:
        preds = calc_sing_pdb(args.input, args.pH, TP=not args.shiftx_only, ML=not args.shifty_only, test=args.test)
        preds.to_csv(args.output, index=None)
    else:
        inputs = []
        with open(args.input) as f:
            for line in f:
                line_content = line.split()
                if len(line_content) == 1:
                    # No pH values explicitly specified. Use the global pH values
                    line_content.append(args.pH)
                else:
                    line_content[-1] = float(line_content[-1])
                inputs.append(line_content)
        # Decide saving folder
        if args.output == "shifts.csv":
            # No specific output path specified. Store all files in the current folder
            SAVE_PREFIX = ""
        else:
            SAVE_PREFIX = args.output
            if SAVE_PREFIX[-1] != "/":
                SAVE_PREFIX = SAVE_PREFIX + "/"

        for idx, item in enumerate(inputs):
            preds = calc_sing_pdb(item[0], item[1], TP=not args.shiftx_only, ML=not args.shifty_only, test=args.test)
            preds.to_csv(SAVE_PREFIX + os.path.basename(item[0]).replace(".pdb", ".csv"), index=None)
            print("Finished prediction for %s (%d/%d)" % (item[0], idx + 1, len(inputs)))    
    
    print("Complete!")
   

