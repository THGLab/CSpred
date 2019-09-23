#!/usr/bin/env python
# Script for making predictions using both the TP module and the ML module

# Author: Jie Li
# Date created: Sep 20, 2019

from build_df.spartap_features import PDB_SPARTAp_DataReader
from data_prep_functions import *
import shiftypp
import joblib
# from stacking_estimator import StackingEstimator
# from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
# from sklearn.pipeline import make_pipeline, make_union
import toolbox
import os
import pandas as pd
import argparse
# import warnings

# Suppress Setting With Copy warnings
pd.options.mode.chained_assignment = None

SCRIPT_PATH=os.path.dirname(os.path.realpath(__file__))
ML_MODEL_PATH=SCRIPT_PATH+"/pipelines/"

def build_input(pdb_file_name,pH=5,rcfeats=True, hse=True, hbrad=[5.0]*3):
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
    data=data.copy()
    data=data[sorted(data.columns)]
    data=ha23ambigfix(data, mode=0)
    Add_res_spec_feats(data,include_onehot=False)
    data=feat_pwr(data,hbondd_cols+cos_cols,[2])
    data=feat_pwr(data,hbondd_cols,[-1,-2,-3])
    dropped_cols=dssp_pp_cols+dssp_energy_cols+['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM',"RES", 'CHAIN', 'RESNAME_ip1', 'RESNAME_im1', 'BMRB_RES_NUM', 'CG', 'RCI_S2', 'MATCHED_BMRB',"identifier"]+rcoil_cols
    data=data.drop(set(dropped_cols)&set(data.columns),axis=1)
    return data

def prepare_data_for_atom(data,atom):
    '''
    Function to generate features data for a given atom type: meaning that the irrelevant ring current values are removed from features

    args:
        data - the dataset that contains all the features (pandas.DataFrame)
        atom - the atom to keep ring currents

    returns:
        pandas.DataFrame containing the cleaned feature set
    '''
    dat=data.copy()
    ring_col = atom + '_RC'
    rem1 = ring_cols.copy()
    rem1.remove(ring_col)
    dat = dat.drop(rem1, axis=1)
    dat[ring_col] = dat[ring_col].fillna(value=0)
    return dat

def calc_sing_pdb(pdb_file_name,pH=5,TP=True,ML=True,test=False):
    '''
    Function for calculating chemical shifts for a single PDB file using TP module / ML module / both

    args:
        pdb_file_name = The path to the PDB file for prediction (str)
        pH = pH value to be considered
        TP = Whether or not use TP module (Bool)
        ML = Whether or not use ML module (Bool)
        test = Whether or not use test mode (Exclude mode for SHIFTY++, Bool)
    '''
    if pH<2 or pH>12:
        print("Warning! Predictions for proteins in extreme pH conditions are likely to be erroneous. Take prediction results at your own risk!")
    preds=pd.DataFrame()
    if TP:
        print("Calculating TP predictions ...")
        TP_pred=shiftypp.main(pdb_file_name,1,secondary=True,exclude=test)
        if not ML:
            # Prepare data when only doing TP prediction
            preds=TP_pred[["RESNUM","RESNAME"]]
            for atom in toolbox.ATOMS:
                if atom+"_RC" in TP_pred.columns:
                    rc=TP_pred[atom+"_RC"]
                else:
                    rc=0
                preds[atom+"_TP"]=TP_pred[atom]+rc
    if ML:
        print("Generating features ...")
        feats=build_input(pdb_file_name,pH)
        feats.rename(index=str, columns=sparta_rename_map,inplace=True) # Rename columns so that random coil columns can be correctly recognized
        resnames=feats["RESNAME"]
        resnums=feats["RES_NUM"]
        rcoils=feats[rcoil_cols]
        feats=data_preprocessing(feats)

        result={"RESNUM":resnums,"RESNAME":resnames}
        print("Calculating ML predictions ...")
        for atom in toolbox.ATOMS:
            # Predictions for each atom
            atom_feats=prepare_data_for_atom(feats,atom)
            r0=joblib.load(ML_MODEL_PATH+"%s_model1.sav"%atom)
            r0_pred=r0.predict(atom_feats.values)

            feats_r1=atom_feats.copy()
            feats_r1["R0_PRED"]=r0_pred
            r1=joblib.load(ML_MODEL_PATH+"%s_model2_ny.sav"%atom)
            r1_pred=r1.predict(feats_r1.values)
            # Write ML predictions
            result[atom+"_ML"]=r1_pred+rcoils["RCOIL_"+atom]

            if TP:
                print("Calculating Combined predictions ...")
                feats_r2=atom_feats.copy()
                feats_r2["RESNAME"]=resnames
                feats_r2["RESNUM"]=resnums
                tp_atom=TP_pred[["RESNAME","RESNUM",atom,"MAX_IDENTITY","AVG_IDENTITY"]]
                feats_r2=pd.merge(feats_r2,tp_atom,on="RESNUM",suffixes=("","_TP"),how="left")
                # Write TP predictions
                result[atom+"_TP"]=feats_r2[atom].values+rcoils["RCOIL_"+atom].values
                valid=(feats_r2.RESNAME==feats_r2.RESNAME_TP) & (feats_r2[atom].notnull())
                feats_r2["R0_PRED"]=r0_pred
                valid_feats_r2=feats_r2.drop(["RESNAME","RESNUM","RESNAME_TP"],axis=1)[valid]
                r2_pred=r1_pred.copy()
                if len(valid_feats_r2):
                    r2=joblib.load(ML_MODEL_PATH+"%s_model2_wy.sav"%atom)
                    r2_pred_valid=r2.predict(valid_feats_r2.values)
                    r2_pred[valid]=r2_pred_valid
                # Write final predictions
                result[atom+"_FINAL"]=r2_pred+rcoils["RCOIL_"+atom]
        preds=pd.DataFrame(result)
    return preds

        

if __name__=="__main__":
    args=argparse.ArgumentParser(description='This program is an NMR chemical shift predictor for protein backbone chemical shifts in aqueous solution. It has two sub-modules: one is the transfer prediction (TP) module that predicts chemical shifts by "transfering" shifts from similar proteins in the database to the query protein through structure and sequence alignments. It uses the SHIFTY++ programs to calculate TP results. The second sub-module is the machine learning (ML) module. It uses ensemble tree-based methods to predict chemical shifts from the features extracted from the PDB files. ')
    args.add_argument("input",help="The query PDB file or list of PDB files for which the shifts are calculated")
    args.add_argument("--batch","-b",action="store_true",help="If toggled, input accepts a text file specifying all the PDB files need to be calculated (Each line is a PDB file name. If pH values are specified, followed with a space)")
    args.add_argument("--output", "-o",help="Filename of generated output file. A file [shifts.csv] is generated by default",default="shifts.csv")
    args.add_argument("--tp_only","-tp",action="store_true",help="Only use the transfer prediction module. Equivalent to executing the SHIFTY++ program directly with default settings")
    args.add_argument("--ml_only","-ml",action="store_true",help="Only use the machine learning module. No alignment results will be utilized or calculated")
    args.add_argument("--pH","-pH","-ph",type=float,help="pH value to be considered. Default is 5",default=5)
    args.add_argument("--test","-t",action="store_true",help="If toggled, using test mode for TP prediction")
    args=args.parse_args()
 
    if not args.batch:
        preds=calc_sing_pdb(args.input,args.pH,not args.ml_only,not args.tp_only,args.test)
        preds.to_csv(args.output,index=None)
        print("Complete!")
   
