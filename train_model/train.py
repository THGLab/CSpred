#!/usr/bin/env python
# This is the script for training the machine learning part of UCBShift. The first level predictions in subsequent features are the "out-of-sample" test predictions by K-fold cross validation of the training data.

# Author: Jie Li
# Date created: Sep 14, 2019

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split,KFold
import joblib
import multiprocessing
import os
import sys
sys.path.append("../")
import toolbox
from data_prep_functions import *



K = 10
PARALLEL_JOBS = 16
DEBUG = False
rmse = lambda x: np.sqrt(np.mean(np.square(x)))

MODEL_SAVE_PATH = "../models/"
DATASET_PATH = "../datasets/"
Y_PRED_PATH = "Y_preds/"


def prep_feat_target(data,atom,task_type,filter_outlier=False,notnull=True):
    '''
    Accepts a complete dataframe (with all features and targets) and prepare data for training

    args:
        data - the dataframe containing single-atom features and targets (pandas.DataFrame)
        atom - the atom for which features are extracted (str)
        task_type - one of "train" or "test" (str)
        filter_outlier - whether or not filter examples with outlier targets (exceed average by 5 standard deviations) (bool)
        notnull - whether filter out examples with null targets (bool)
    '''
    if notnull:
        data=data[data[atom].notnull()]
    if filter_outlier:
        mean=data[atom].mean()
        std=data[atom].std()
        filtered=(data[atom]>mean+5*std)|(data[atom]<mean-5*std)
        data=data[np.logical_not(filtered)]
        print("%d residues filtered because they exceeded 5 standard deviations"%np.sum(filtered))

        
    
    # filter outlayers
    hydrogenatoms = ['HB', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE21', 'HE22', 'HG', 'HG1', 'HG12', 'HG13', 'HG2', 'HG3', 'HZ', 'HE3', 'HZ3' ,'HH2', 'HZ2', 'HA', 'H'] 
    carbonatoms = ['CG','CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CG1', 'CG2', 'CZ', 'CE3', 'CZ3', 'CH2', 'CA', 'CB', 'C', 'CZ2']
    nitrogenatoms = ['N','ND2','NE1','NE2'] 
    

    if atom  in hydrogenatoms:
        for aa in range(len(toolbox.AMINOACIDS)):
            rows_before_drop = data.shape[0] 
            data.drop(data[((data[atom] < -6) | (data[atom] > 6)) & (data['RESNAME'] == toolbox.AMINOACIDS[aa])].index, inplace=True)
            rows_after_drop = data.shape[0] 
            rows_dropped = rows_before_drop - rows_after_drop
            print(f"{rows_dropped} residues filtered for aa={toolbox.AMINOACIDS[aa]} and atom={atom}.")

                
    elif atom in carbonatoms:
        for aa in range(len(toolbox.AMINOACIDS)):
            rows_before_drop = data.shape[0]
            data.drop(data[((data[atom] < -12) | (data[atom] > 12)) & (data['RESNAME'] == toolbox.AMINOACIDS[aa])].index, inplace=True)
            rows_after_drop = data.shape[0] 
            rows_dropped = rows_before_drop - rows_after_drop
            print(f"{rows_dropped} residues filtered for aa={toolbox.AMINOACIDS[aa]} and atom={atom}.")

 
    elif atom in nitrogenatoms:
        for aa in range(len(toolbox.AMINOACIDS)):
            if toolbox.AMINOACIDS[aa] in ['HIS']:
                data.drop(data[(data['RESNAME'] == toolbox.AMINOACIDS[aa])].index, inplace=True)

   

    
    data.fillna(0,inplace=True)
    data=combine_shift(data,atom,Y_PRED_PATH) 
    # Subtract random coils for SHIFTY predictions
    data["SHIFTY_" + atom] = data["SHIFTY_" + atom] - data["RCOIL_" + atom]
    features = data.drop([atom,"RESNAME","FILE_ID","RESNAME","RES_NUM","RCOIL_" + atom], axis=1)
    
    print("Shape of features:",features.shape)
    targets = data[atom]
    meta=data[["FILE_ID","RESNAME","RES_NUM"]]
    return features,targets,meta
    

def data_preprocessing(data):
    '''
    Function for executing all the preprocessing steps based on the original extracted features, including fixing HA2/HA3 ring current ambiguity, adding hydrophobicity, powering features, drop unnecessary columns, etc.

    returns:
        data - the dataframe after preprocessing (pandas.DataFrame)
    '''
    data=data.copy()
      
    # ASN/GLN amiguity
    condition = (data['HD22'] - data['HD21']) >= 0.4
    condition1 = (data['HE22'] - data['HE21']) >= 0.42
    data.loc[condition, ['HD21', 'HD22']] = data.loc[condition, ['HD22', 'HD21']].values
    data.loc[condition1, ['HE21', 'HE22']] = data.loc[condition1, ['HE22', 'HE21']].values


    
    data = data.rename(index=str, columns=sparta_rename_map) 
    data=data[sorted(data.columns)]
    data=ha23ambigfix(data, mode=0)
    data=dihedral_purifier(data, drop_cols=True)
    data=dssp_purifier(data)
    data=diff_targets(data,rings=False,coils=True,drop_cols=False)
    Add_res_spec_feats(data,include_onehot=False)
    data=feat_pwr(data,hbondd_cols+cos_cols,[2])
    data=feat_pwr(data,hbondd_cols,[-1,-2,-3])
    dropped_cols=dssp_pp_cols+dssp_energy_cols+['RES_NUM_x', 'RES_NUM_y','FILE_ID_x','FILE_ID_y','Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1',  'PDB_FILE_NAME',"RES", 'CHAIN', 'RESNAME_ip1', 'RESNAME_im1', 'BMRB_RES_NUM', 'RCI_S2', 'MATCHED_BMRB',"identifier"]+["RESNAME_i%s%d"%(a,b) for a in ['+','-'] for b in range(1,21)]
    data=data.drop(set(dropped_cols)&set(data.columns),axis=1)        
    

    return data


def prepare_data_for_atom(data,atom):
    '''
    Function to generate features data for a given atom type: meaning that the irrelevant ring current values, chemical shifts targets and random coil values for other atom types are removed from the dataset

    args:
        data - the dataset that contains all the features (pandas.DataFrame)
        atom - the atom to keep ring currents

    returns:
        pandas.DataFrame containing the cleaned feature set
    '''
    
    dat=data.copy()
    
    column_names = dat.columns.tolist()
    new_column_names = [name.replace('.1', '') if name.endswith('.1') else name for name in column_names]
    dat.columns = new_column_names

    
    ring_col = atom + '_RC'
    rem1 = ring_cols.copy()
    rem1.remove(ring_col)
    
    rem2 = [rm_atom + "_RING" for rm_atom in ['C', 'CA', 'CB', 'N', 'HA', 'HA2', 'HA3', 'H', '1H', '1HA', '2HA','CG','CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CG1', 'CG2', 'CZ','HB', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE21', 'HE22', 'HE3', 'HG', 'HG1', 'HG12', 'HG13', 'HG2', 'HG3', 'HZ','CE3','CZ3','HZ3','CH2','HH2','CZ2','HZ2', 'HB1', 'HD11', 'HD12', 'HD13', 'HD23', 'HG11', 'HZ1', 'HG21', 'HG22', 'HG23', 'ND2','NE1','NE2']]
    rem3 = ["RCOIL_" + rm_atom for rm_atom in toolbox.ATOMS if rm_atom != atom]
    rem4 = [rm_atom + "_EFIELD" for rm_atom in ['HA2', 'HA3', 'HA', 'H', 'HB', 'HB1', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE21', 'HE22', 'HG', 'HG1', 'HG11', 'HG12', 'HG13', 'HG2', 'HG3', 'HZ','HD11', 'HD12', 'HD13', 'HD23', 'HE3','HZ3','HH2','HZ2', 'HZ1', 'HG21', 'HG22', 'HG23', 'HZ1', 'HZ2', 'HZ3', 'ND2','NE1','NE2','N'] if rm_atom != atom]
    rem5 = [rm_atom + "_dHA" for rm_atom in ['HB', 'HB1', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE21', 'HE22', 'HG', 'HG1', 'HG12', 'HG13', 'HG2', 'HG3', 'HZ', 'HD11', 'HD12', 'HD13', 'HD23', 'HE3','HZ3','HH2','HZ2', 'HZ1', 'HG21', 'HG22', 'HG23', 'HG'] if rm_atom != atom]
    rem6 = [rm_atom + "_COS_H" for rm_atom in ['HB', 'HB1', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE21', 'HE22', 'HG', 'HG1', 'HG12', 'HG13', 'HG2', 'HG3', 'HZ', 'HD11', 'HD12', 'HD13', 'HD23', 'HE3','HZ3','HH2','HZ2', 'HZ1', 'HG21', 'HG22', 'HG23', 'HG'] if rm_atom != atom]
    rem7 = [rm_atom + "_COS_A" for rm_atom in ['HB', 'HB1', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE21', 'HE22', 'HG', 'HG1', 'HG12', 'HG13', 'HG2', 'HG3', 'HZ', 'HD11', 'HD12', 'HD13', 'HD23', 'HE3','HZ3','HH2','HZ2', 'HZ1', 'HG21', 'HG22', 'HG23', 'HG'] if rm_atom != atom]
    rem8 = [rm_atom + "_EXISTS" for rm_atom in ['HB', 'HB1', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE21', 'HE22', 'HG', 'HG1', 'HG12', 'HG13', 'HG2', 'HG3', 'HZ','HD11', 'HD12', 'HD13', 'HD23', 'HE3','HZ3','HH2','HZ2', 'HZ1', 'HG21', 'HG22', 'HG23', 'HG'] if rm_atom != atom]
    rem9 = [rm_atom + "_ENERGY" for rm_atom in ['HB', 'HB1', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE21', 'HE22', 'HG', 'HG1', 'HG12', 'HG13', 'HG2', 'HG3', 'HZ','HD11', 'HD12', 'HD13', 'HD23', 'HE3','HZ3','HH2','HZ2', 'HZ1', 'HG21', 'HG22', 'HG23', 'HG'] if rm_atom != atom]
    
    rem10 = toolbox.ATOMS.copy() 
    rem10.remove(atom)
    
    dat = dat.drop(rem1 + rem2 + rem3 + rem4 + rem5 + rem6 + rem7 + rem8 + rem9 + rem10, axis=1, errors='ignore')
    
    hbondd_sidechain_cols = [i+j for i in ['HB', 'HB1', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE21', 'HE22', 'HG', 'HG1', 'HG12', 'HG13', 'HG2', 'HG3', 'HZ','HD11', 'HD12', 'HD13', 'HD23', 'HE3','HZ3','HH2','HZ2', 'HZ1', 'HG21', 'HG22', 'HG23']  for j in ['_dHA', '_COS_H', '_COS_A']]
    hbondd_sidechain_cols = [element for element in hbondd_sidechain_cols if element.startswith(atom + '_')]
    #add polynomial transformation of side chain hbonds
    dat = dat.loc[:, ~dat.columns.duplicated()]
    dat=feat_pwr(dat,hbondd_sidechain_cols,[-1,-2,-3])
    dat=feat_pwr(dat,hbondd_sidechain_cols,[2])

    # filter if too little data
    little = {
        "HD2":["ASP"]}
    
    if atom in little:
        resnames = little[atom]  # Get the list of corresponding residue names for the selected atom
        dat.loc[dat['RESNAME'].isin(resnames), atom] = np.nan

    dat[ring_col] = dat[ring_col].fillna(value=0)

    dat = dat.loc[:, ~dat.columns.duplicated()]
     
    return dat





def evaluate(preds,targets,metas):
    '''
    Function to evaluate the performance of a model, given the predictions and the associated targets

    args:
        preds = all the predictions (numpy.array of shape (n,))
        targets = all the target values (numpy.array of shape (n,))
        metas = all the metadata about the data (pandas.DataFrame with len n)
    '''
    print("Evaluating ...")
    valid = targets.notnull().values.ravel()
    pred_valid=preds[valid]
    targ_valid=targets[valid].values.ravel()
    err = rmse(pred_valid-targ_valid)
    corr = np.corrcoef(pred_valid,targ_valid)[0,1]
    print("Error:%.3f\nCorr:%.3f"%(err,corr))

def combine_shift(df,atom,shift_pred_path):
    '''
    Function for combining features and SHIFTY++ predictions based on metadata (RESNUM)

    args:
        df = dataframe for all the features (pandas.DataFrame)
        atom =  the atom type for which shifts are combined into features (str)
        shift_pred_path = path to all the shifts (all shifts for a single PDB should be in separate .csv files)
    '''
    print("Combining features with SHIFTY++ predictions")
    new_df_singles=[]
    for pdbid in set(df["FILE_ID"]):
        pdb_idx=df["FILE_ID"]==pdbid
        pdb_df=df[pdb_idx].copy()
        shift_pred_file=[file for file in os.listdir(shift_pred_path) if str(pdbid) in file]
        if not len(shift_pred_file)==1:
            # Only combine SHIFTY++ predictions when there is exactly one match
            print("Unexpected number of shift files for %s:%d"%(pdbid,len(shift_pred_file)))
            pdb_df["SHIFTY_"+atom]=np.nan
            pdb_df["BEST_REF_SCORE_"+atom]=0
            pdb_df["BEST_REF_COV_"+atom]=0
            pdb_df["BEST_REF_MATCH_"+atom]=0
            new_df_singles.append(pdb_df)
            continue
        else:
            shifts=pd.read_csv(shift_pred_path+shift_pred_file[0])
        shift_single_df=shifts[["RESNAME"]].copy()
        shift_single_df["RES_NUM"]=shifts.RESNUM
        shift_single_df["SHIFTY_"+atom]=shifts[atom]
        if atom+"_BEST_REF_SCORE" in shifts.columns:
            shift_single_df["BEST_REF_SCORE_"+atom]=shifts[atom+"_BEST_REF_SCORE"]
            shift_single_df["BEST_REF_COV_"+atom]=shifts[atom+"_BEST_REF_COV"]
            shift_single_df["BEST_REF_MATCH_"+atom]=shifts[atom+"_BEST_REF_MATCH"]
        else:
            shift_single_df["BEST_REF_SCORE_"+atom]=0
            shift_single_df["BEST_REF_COV_"+atom]=0
            shift_single_df["BEST_REF_MATCH_"+atom]=0
        merged_df=pd.merge(pdb_df,shift_single_df,on="RES_NUM",how="left",suffixes=("","1"))
        if not (merged_df["RESNAME"]==merged_df["RESNAME1"]).all():
            merged_df[(merged_df["RESNAME"]!=merged_df["RESNAME1"])]["SHIFTY_"+atom]=np.nan
        merged_df.drop("RESNAME1",axis=1,inplace=True)
        new_df_singles.append(merged_df)
    new_df=pd.concat(new_df_singles,ignore_index=True)
    return new_df


def train_with_test(features,targets,train_idx,test_idx):
    '''
    Function that trains an ExtraTreeRegressor based on a subset of the dataset specified by the train indices, and returns the test performance specified by the test indices. Used for generating "out-of-sample" first level predictions in parallel

    args:
        features = all the features for the data (pandas.DataFrame)
        targets = all the targets for the data (pandas.Series)
        train_idx = indices for all the training data (list)
        test_idx = indices for all the testing data (list)
    '''
    predictor = ExtraTreesRegressor(bootstrap=False, max_features=0.3, min_samples_leaf=3, min_samples_split=15, n_estimators=500)
    train_feats=features.values[train_idx,:]
    train_targets=targets.iloc[train_idx].values.ravel()
    test_feats=features.values[test_idx,:]

    predictor.fit(train_feats,train_targets)
    first_pred=predictor.predict(test_feats).ravel()
    return first_pred

def train_for_atom(atom, dataset):
    '''
    Function for training machine learning models for a single atom

    args:
        atom = the atom that the models are trained for (str)
        dataset = the dataframe containing all the training data for all atoms (pandas.DataFrame)
        rcoil_atom = random coil values for the specified atom type (pandas.Series)
    '''
    print("  ======  Training model for:",atom, "  ======  ")
    single_atom_data = prepare_data_for_atom(train_data, atom)
    #print(single_atom_data.columns.tolist())
    features,targets,metas = prep_feat_target(single_atom_data,atom,"train",filter_outlier=False,notnull=True)
    
    
    kf=KFold(n_splits=K,shuffle=True)
    # Prepare parameters for Kfold in a list and do "out-of-sample" training and testing on training dataset for K folds
    print("Training R0 to provide OOB predictions as features for R1 and R2...")
    params=[]
    for train_idx,test_idx in kf.split(range(len(features))):
        params.append([features.drop(["SHIFTY_"+atom,"BEST_REF_SCORE_"+atom,"BEST_REF_COV_"+atom,"BEST_REF_MATCH_"+atom],axis=1),targets,train_idx,test_idx])
    pool=multiprocessing.Pool(processes=K)
    first_preds=pool.starmap(train_with_test,params)
    # first_preds=train_with_test(*params[0])

    # Combine results from K parallel execusions into a single list
    all_test_idx=[]
    all_first_preds=[]
    for i in range(K):
        all_test_idx.extend(params[i][-1])
        all_first_preds.extend(first_preds[i])
    first_preds=pd.Series(all_first_preds,index=all_test_idx)
    features["FIRST_PRED"]=first_preds
    evaluate(first_preds.sort_index(),targets,metas)

    # Retrain the model on all training data
    print("Retraining R0 with all data...")
    R0=ExtraTreesRegressor(bootstrap=False, max_features=0.3, min_samples_leaf=3, min_samples_split=15, n_estimators=500,n_jobs = PARALLEL_JOBS)
    R0_x=features.drop(["SHIFTY_"+atom,"BEST_REF_SCORE_"+atom,"BEST_REF_COV_"+atom,"BEST_REF_MATCH_"+atom,"FIRST_PRED"],axis=1).values
    R0_y=targets.values.ravel()
    R0.fit(R0_x,R0_y)


    # Save first level model (R0)
    if not DEBUG:
        joblib.dump(R0,MODEL_SAVE_PATH+"%s_R0.sav"%atom)



    # Train and save second level model  (R1)
    print("Training UCBShift-X with %d examples..."%len(features))
    R1=RandomForestRegressor(bootstrap=False, max_features=0.5, min_samples_leaf=7, min_samples_split=12, n_estimators=500,n_jobs = PARALLEL_JOBS)
    R1_x=features.drop(["SHIFTY_"+atom,"BEST_REF_SCORE_"+atom,"BEST_REF_COV_"+atom,"BEST_REF_MATCH_"+atom],axis=1).values
    R1_y=targets.values.ravel()
    R1.fit(R1_x,R1_y)
    if not DEBUG:
        joblib.dump(R1,MODEL_SAVE_PATH+"%s_R1.sav"%atom)




    # Train and save second level model with UCBShift-Y predictions (R2)
    R2=RandomForestRegressor(bootstrap=False, max_features=0.5, min_samples_leaf=7, min_samples_split=12, n_estimators=500,n_jobs = PARALLEL_JOBS)
    not_null_idx=features["SHIFTY_"+atom].notnull()

    print("Training combined UCBShift model with X and Y parts with %d examples..."%np.sum(not_null_idx))
    
    R2_x=features[not_null_idx].values
    R2_y=targets[not_null_idx].values.ravel()
    R2.fit(R2_x,R2_y)
    if not DEBUG:
        joblib.dump(R2,MODEL_SAVE_PATH+"%s_R2.sav"%atom)

    print("Finish for",atom)






if __name__=="__main__":
    if not os.path.exists(MODEL_SAVE_PATH):
        os.mkdir(MODEL_SAVE_PATH)
    print("  ======  Reading all datasets  ======  ")
    train_data = pd.concat([pd.read_csv(DATASET_PATH+single_df) for single_df in os.listdir(DATASET_PATH)],ignore_index=True)
    train_data = data_preprocessing(train_data)
    
    for atom in toolbox.ATOMS:
        train_for_atom(atom,train_data)
    
    print("All done!")


