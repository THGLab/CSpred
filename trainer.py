#!/usr/bin/env python
# This is the script for training the machine learning part of the chemical shift predictor. The first level predictions in subsequent features are the "out-of-sample" test predictions by K-fold cross validation of the training data.

# Author: Jie Li
# Date created: Sep 14, 2019

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split,KFold
from sklearn.pipeline import make_pipeline, make_union
from stacking_estimator import StackingEstimator
from sklearn.externals import joblib
import multiprocessing
import os
import sys
import toolbox

K=10
DEBUG=False
prefix="datasets/pipelines/"
pred_save_path="../combination/"
# pred_save_path=None
rmse = lambda x: np.sqrt(np.mean(np.square(x)))


# Definition of training, testing and validation data
trains=toolbox.load_pkl("../chains/new_train_ids.pkl")
vals=toolbox.load_pkl("../chains/new_val_ids.pkl")
tests=toolbox.load_pkl("../chains/test.pkl")

def prep_data(add,atom,task_type,filter_outlier=False,notnull=True):
    '''
    Read all data files from the given address and prepare data for training

    args:
        add - a single .csv file address for reading data, or a list of file addresses to be read in (str/list(str))
        atom - the atom for which features are extracted (str)
        task_type - one of [train] or [test] (str)
        filter_outlier - whether or not filter examples with outlier targets (exceed average by 5 standard deviations) (bool)
        notnull - whether filter out examples with null targets (bool)
    '''
    print("Preparing data for",task_type)

    if type(add) is list:
        data=pd.concat([pd.read_csv(single_add) for single_add in add])
    else:    
        data=pd.read_csv(add)
    # data=data[:300]
    if notnull:
        data=data[data[atom].notnull()]
    if filter_outlier:
        mean=data[atom].mean()
        std=data[atom].std()
        filtered=(data[atom]>mean+5*std)|(data[atom]<mean-5*std)
        data=data[np.logical_not(filtered)]
        print("%d residues filtered because they exceeded 5 standard deviations"%np.sum(filtered))
    data.fillna(0,inplace=True)
    if task_type=="train":
        data=combine_shift(data,atom,"/data/jerry/NMR/combination/shifty++_preds_train+val/") # Specify shifts prediction files for training
    else:
        data=combine_shift(data,atom,"/data/jerry/NMR/combination/shifty++_preds_test/") # Specify shifts prediction files for testing
    features = data.drop([atom,"FILE_ID","RESNAME","RES_NUM"], axis=1)
    print("Shape of features:",features.shape)
    targets = data[atom]
    meta=data[["FILE_ID","RESNAME","RES_NUM"]]
    return features,targets,meta
    
def evaluate(preds,targets,metas,save_add):
    '''
    Function to evaluate the performance of a model, given the predictions and the associated targets

    args:
        preds = all the predictions (numpy.array of shape (n,))
        targets = all the target values (numpy.array of shape (n,))
        metas = all the metadata about the data (pandas.DataFrame with len n)
        save_add = the address for saving the result .csv file
    '''
    print("Evaluating ...")
    valid = targets.notnull().values.ravel()
    pred_valid=preds[valid]
    targ_valid=targets[valid].values.ravel()
    err = rmse(pred_valid-targ_valid)
    corr = np.corrcoef(pred_valid,targ_valid)[0,1]
    print("Error:%.3f\nCorr:%.3f"%(err,corr))
    output=metas.copy()
    output["PRED"]=preds
    output["REAL"]=targets
    output.to_csv(save_add,index=None)



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
        shift_pred_file=[file for file in os.listdir(shift_pred_path) if pdbid in file]
        if not len(shift_pred_file)==1:
            # Only combine SHIFTY++ predictions when there is exactly one match
            print("Unexpected number of shift files for %s:%d"%(pdbid,len(shift_pred_file)))
            pdb_df["SHIFTY_"+atom]=np.nan
            pdb_df["MAX_IDENTITY"]=0
            pdb_df["AVG_IDENTITY"]=0
            new_df_singles.append(pdb_df)
            continue
        else:
            shifts=pd.read_csv(shift_pred_path+shift_pred_file[0])
        shift_single_df=shifts[["RESNAME"]].copy()
        shift_single_df["RES_NUM"]=shifts.RESNUM
        shift_single_df["SHIFTY_"+atom]=shifts[atom]
        shift_single_df["MAX_IDENTITY"]=shifts.MAX_IDENTITY
        shift_single_df["AVG_IDENTITY"]=shifts.AVG_IDENTITY
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

def train_for_atom(atom,dataset_path,pred_save_path):
    '''
    Function for training machine learning models for a single atom

    args:
        atom = the atom that the models are trained for (str)
        dataset_path = the path to which datasets can be found (expected to have three .csv files under the path, for train/validation/test)
        pred_save_path = the path for saving all the predictions for analysis
    '''
    print("  ======  Training model for:",atom,"under folder",dataset_path,"  ======  ")
    features,targets,metas = prep_data([dataset_path+"train_['%s'].csv"%atom,dataset_path+"val_['%s'].csv"%atom],atom,"train",filter_outlier=True,notnull=True)
    features_test,targets_test,metas_test = prep_data(dataset_path+"test_['%s'].csv"%atom,atom,"test",filter_outlier=False,notnull=False)
    kf=KFold(n_splits=K)
    # Prepare parameters for Kfold in a list and do "out-of-sample" training and testing on training dataset for K folds
    print("Training R0...")
    params=[]
    for train_idx,test_idx in kf.split(range(len(features))):
        params.append([features.drop(["SHIFTY_"+atom,"MAX_IDENTITY","AVG_IDENTITY"],axis=1),targets,train_idx,test_idx])
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
    evaluate(first_preds,targets,metas,pred_save_path+"first_pred_%s.csv"%atom)

    # Retrain the model on all training data
    model1=ExtraTreesRegressor(bootstrap=False, max_features=0.3, min_samples_leaf=3, min_samples_split=15, n_estimators=500)
    model1.fit(features.drop(["SHIFTY_"+atom,"MAX_IDENTITY","AVG_IDENTITY","FIRST_PRED"],axis=1),targets.values.ravel())

    # Write first predictions for the test dataset to the features of test
    features_test["FIRST_PRED"]=model1.predict(features_test.drop(["SHIFTY_"+atom,"MAX_IDENTITY","AVG_IDENTITY"],axis=1))

    # Save first level model (R0)
    if not DEBUG:
        joblib.dump(model1,"pipelines/%s_model1.sav"%atom)

    # Train and save second level model  (R1)
    print("Training second level model without SHIFTY++ with %d examples..."%len(features))
    model_2=RandomForestRegressor(bootstrap=False, max_features=0.5, min_samples_leaf=7, min_samples_split=12, n_estimators=500)
    model_2.fit(features.drop(["SHIFTY_"+atom,"MAX_IDENTITY","AVG_IDENTITY"],axis=1),targets.values.ravel())
    pred_2=model_2.predict(features_test.drop(["SHIFTY_"+atom,"MAX_IDENTITY","AVG_IDENTITY"],axis=1)).ravel()
    evaluate(pred_2,targets_test,metas_test,pred_save_path+"second_pred_%s_nosy.csv"%atom)
    if not DEBUG:
        joblib.dump(model_2,"pipelines/%s_model2_ny.sav"%atom)

    # Train and save second level model with SHIFTY++ predictions (R2)
    model_21=RandomForestRegressor(bootstrap=False, max_features=0.5, min_samples_leaf=7, min_samples_split=12, n_estimators=500)
    not_null_idx=features["SHIFTY_"+atom].notnull()
    not_null_idx_test=features_test["SHIFTY_"+atom].notnull()

    print("Training second level model with SHIFTY++ with %d examples..."%np.sum(not_null_idx))
    model_21.fit(features[not_null_idx],targets[not_null_idx].values.ravel())
    pred_21=pred_2.copy()
    pred_21[not_null_idx_test]=model_21.predict(features_test[not_null_idx_test])
    evaluate(pred_21,targets_test,metas_test,pred_save_path+"second_pred_%s_withsy.csv"%atom)
    if not DEBUG:
        joblib.dump(model_21,"pipelines/%s_model2_wy.sav"%atom)

    print("Finish for",atom)

if __name__=="__main__":
    for atom in toolbox.ATOMS:
        train_for_atom(atom,prefix+atom+"/",pred_save_path)
    print("All done!")
