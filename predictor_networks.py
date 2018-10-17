#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:51:30 2018

@author: bennett
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:51:30 2018

@author: bennett
"""

import pandas as pd
import numpy as np
import math
import random
import sklearn as skl
import sklearn.model_selection
import keras
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
from Bio.SeqUtils import IUPACData
from lsuv_init import LSUVinit

atom_names = ['HA', 'H', 'CA', 'CB', 'C', 'N']
sparta_results = [0.25, 0.49, 0.94, 1.14, 1.09, 2.45] # Advertised performance of SPARTA+

# For easier access, define the names of different feature columns
col_phipsi = ['PSI_'+i for i in ['COS_i-1', 'SIN_i-1']]
col_phipsi += [i+j for i in ['PHI_', 'PSI_'] for j in ['COS_i', 'SIN_i']]
col_phipsi += ['PHI_'+i for i in ['COS_i+1', 'SIN_i+1']]
col_chi = [i+j+k for k in ['_i-1', '_i', '_i+1'] for i in ['CHI1_', 'CHI2_'] for j in ['COS', 'SIN', 'EXISTS']]
col_hbprev = ['O_'+i+'_i-1' for i in ['d_HA', '_COS_H', '_COS_A', '_EXISTS']]
col_hbond = [i+j+'_i' for i in ['Ha_', 'HN_', 'O_'] for j in ['d_HA', '_COS_H', '_COS_A', '_EXISTS']]
col_hbnext = ['HN_'+i+'_i+1' for i in ['d_HA', '_COS_H', '_COS_A', '_EXISTS']]
col_s2 = ['S2'+i for i in ['_i-1', '_i', '_i+1']]
struc_cols = col_phipsi + col_chi + col_hbprev + col_hbond + col_hbnext + col_s2
blosum_names = ['BLOSUM62_NUM_'+list(IUPACData.protein_letters_3to1.keys())[i].upper() for i in range(20)]
col_blosum = [blosum_names[i]+j for j in ['_i-1', '_i', '_i+1'] for i in range(20)]
seq_cols = col_blosum
bin_seq_cols = ['BINSEQREP_'+ list(IUPACData.protein_letters_3to1.keys())[i].upper() + j for j in ['_i-1', '_i', '_i+1'] for i in range(20)]
rcoil_cols = ['RCOIL_' + atom for atom in atom_names]
ring_cols = [atom + '_RC' for atom in atom_names]
all_cols = struc_cols + seq_cols + rcoil_cols + ring_cols
all_cols_bin = struc_cols + bin_seq_cols + rcoil_cols + ring_cols

# Define the names for the _i+1 and _i-1 columns to allow dropping them for RNNs
im1_cols = ['PSI_'+i for i in ['COS_i-1', 'SIN_i-1']]
ip1_cols = ['PHI_'+i for i in ['COS_i+1', 'SIN_i+1']]
im1_cols += [i+j+'_i-1' for i in ['CHI1_', 'CHI2_'] for j in ['COS', 'SIN', 'EXISTS']]
ip1_cols += [i+j+'_i+1' for i in ['CHI1_', 'CHI2_'] for j in ['COS', 'SIN', 'EXISTS']]
im1_cols += col_hbprev
ip1_cols += col_hbnext
im1_cols += ['S2_i-1']
ip1_cols += ['S2_i+1']
im1_cols_bin = im1_cols.copy()
ip1_cols_bin = ip1_cols.copy()
im1_cols += ['BLOSUM62_NUM_'+list(IUPACData.protein_letters_3to1.keys())[i].upper() + '_i-1' for i in range(20)]
ip1_cols += ['BLOSUM62_NUM_'+list(IUPACData.protein_letters_3to1.keys())[i].upper() + '_i+1' for i in range(20)]
im1_cols_bin += ['BINSEQREP_'+ list(IUPACData.protein_letters_3to1.keys())[i].upper() + '_i-1' for i in range(20)]
ip1_cols_bin += ['BINSEQREP_'+ list(IUPACData.protein_letters_3to1.keys())[i].upper() + '_i+1' for i in range(20)]

# Load data
train_path = '/home/bennett/Documents/Git_Collaborations_THG/office_home/training_data.pkl'
test_path = '/home/bennett/Documents/Git_Collaborations_THG/office_home/test_data.pkl'
train_data_df = pd.read_pickle(train_path)
test_data_df = pd.read_pickle(test_path)
train_data = train_data_df.drop(['FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM'], axis=1)
test_data = test_data_df.drop(['FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM'], axis=1)

train_path2 = '/Users/kcbennett/Documents/Git_Collaborations_THG/shiftpred/shiftx2_training_df.pkl'
test_path2 = '/Users/kcbennett/Documents/Git_Collaborations_THG/shiftpred/shiftx2_test_df.pkl'
train_df = pd.read_pickle(train_path2)
test_df = pd.read_pickle(test_path2)
train_data = train_df.drop(['FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM'], axis=1)
test_data = test_df.drop(['FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM'], axis=1)

# Load new clean_bmrb data
clean_bmrb_traindf = pd.read_csv('/Users/kcbennett/Documents/data/ShiftX2/bmrb_clean/train_bmrb_clean.csv')
clean_bmrb_testdf = pd.read_csv('/Users/kcbennett/Documents/data/ShiftX2/bmrb_clean/test_bmrb_clean.csv')
clean_bmrb_traindf = pd.read_csv('/Users/kcbennett/Documents/data/ShiftX2/bmrb_clean/train_shiftx2_clean_rings.csv')
clean_bmrb_testdf = pd.read_csv('/Users/kcbennett/Documents/data/ShiftX2/bmrb_clean/test_shiftx2_clean_rings.csv')
clean_bmrb_traindf = pd.read_csv('/home/bennett/Documents/DataSets/bmrb_clean/clean_rings/train_shiftx2_clean_rings.csv')
clean_bmrb_testdf = pd.read_csv('/home/bennett/Documents/DataSets/bmrb_clean/clean_rings//test_shiftx2_clean_rings.csv')

# Change blosum to binary sequence representation
bin_clean_bmrb_traindf = blosum_to_binary(clean_bmrb_traindf)
bin_clean_bmrb_testdf = blosum_to_binary(clean_bmrb_testdf)
bin_clean_bmrb_traindf_data = bin_clean_bmrb_traindf.drop(['FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM', 'CHAIN'], axis=1)
bin_clean_bmrb_testdf_data = bin_clean_bmrb_testdf.drop(['FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM', 'CHAIN'], axis=1)
# Normalize bmrb data without adding rcoil and ring columns
mean_train, std_train, clean_bmrb_traindf_white = whitener(all_cols, clean_bmrb_traindf_data)
clean_bmrb_testdf_white = clean_bmrb_testdf_data.copy()
clean_bmrb_testdf_white[all_cols] = (clean_bmrb_testdf_data[all_cols] - mean_train) / std_train
mean_bin, std_bin, bin_clean_bmrb_traindf_white = whitener(struc_cols, bin_clean_bmrb_traindf_data)
bin_clean_bmrb_testdf_white = bin_clean_bmrb_testdf_data.copy()
bin_clean_bmrb_testdf_white = (bin_clean_bmrb_testdf_data[struc_cols] - mean_bin) / std_bin

# The following is to prepare a bunch of diferent DataFrames depending on normalization, sequence representation, differencing rcoils, etc
traindf_fcoils = add_rand_coils(clean_bmrb_traindf)
testdf_fcoils = add_rand_coils(clean_bmrb_testdf)
traindf_bin_fcoils = blosum_to_binary(traindf_fcoils)
testdf_bin_fcoils = blosum_to_binary(testdf_fcoils)
clean_bmrb_traindf_data = clean_bmrb_traindf.drop(['Unnamed: 0', 'FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM', 'CHAIN'], axis=1)
clean_bmrb_testdf_data = clean_bmrb_testdf.drop(['Unnamed: 0', 'FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM', 'CHAIN'], axis=1)
combined_clean_bmrb = pd.concat([clean_bmrb_traindf, clean_bmrb_testdf], ignore_index=True)
combined_clean_bmrb_data = pd.concat([clean_bmrb_traindf_data, clean_bmrb_testdf_data], ignore_index=True)
traindf_fcoils_data = traindf_fcoils.drop(['Unnamed: 0', 'FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM', 'CHAIN'], axis=1)
traindf_bin_fcoils_data = traindf_bin_fcoils.drop(['Unnamed: 0', 'FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM', 'CHAIN'], axis=1)
testdf_bin_fcoils_data = testdf_bin_fcoils.drop(['Unnamed: 0', 'FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM', 'CHAIN'], axis=1)
testdf_fcoils_data = testdf_fcoils.drop(['Unnamed: 0', 'FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM', 'CHAIN'], axis=1)
fcoil_means, fcoil_stds, traindf_fcoils_normdata = whitener(all_cols, traindf_fcoils_data)
_, _, testdf_fcoils_normdata = whitener(all_cols, testdf_fcoils_data, means=fcoil_means, stds=fcoil_stds, test_data=True)
bin_fcoil_means, bin_fcoil_stds, traindf_bin_fcoils_normdata = whitener(struc_cols+ring_cols+rcoil_cols, traindf_bin_fcoils_data)
_, _, testdf_bin_fcoils_normdata = whitener(struc_cols+ring_cols+rcoil_cols, testdf_bin_fcoils_data, means=bin_fcoil_means, stds=bin_fcoil_stds, test_data=True)
traindf_coildiff = diff_targets(traindf_fcoils, rings=False, coils=True)
testdf_coildiff = diff_targets(testdf_fcoils, rings=False, coils=True)
coildiff_means, coildiff_stds, traindf_coildiff_norm = whitener(struc_cols + seq_cols + ring_cols, traindf_coildiff)
_, _, testdf_coildiff_norm = whitener(struc_cols + seq_cols + ring_cols, testdf_coildiff, means=coildiff_means, stds=coildiff_stds, test_data=True)
traindf_bin_coildiff = diff_targets(traindf_bin_fcoils, rings=False, coils=True)
testdf_bin_coildiff = diff_targets(testdf_bin_fcoils, rings=False, coils=True)
traindf_coildiff_data = diff_targets(traindf_fcoils_data, rings=False, coils=True)
testdf_coildiff_data = diff_targets(testdf_fcoils_data, rings=False, coils=True)
cdiff_means, cdiff_stds, traindf_coildiff_normdata = whitener(struc_cols+seq_cols+ring_cols, traindf_coildiff_data)
_, _, testdf_coildiff_normdata = whitener(struc_cols+seq_cols+ring_cols, testdf_coildiff_data, means=cdiff_means, stds=cdiff_stds, test_data=True)
traindf_bin_coildiff_data = diff_targets(traindf_bin_fcoils_data, rings=False, coils=True)
testdf_bin_coildiff_data = diff_targets(testdf_bin_fcoils_data, rings=False, coils=True)
bin_cdiff_means, bin_cdiff_stds, traindf_bin_coildiff_normdata = whitener(struc_cols+ring_cols, traindf_bin_coildiff_data)
_, _, testdf_bin_coildiff_normdata = whitener(struc_cols+ring_cols, testdf_bin_coildiff_data, means=bin_cdiff_means, stds=bin_cdiff_stds, test_data=True)

# Can make lists of train and test DataFrames
traindfs = [traindf_fcoils_normdata, traindf_bin_fcoils_data, traindf_bin_fcoils_normdata, traindf_coildiff_data, traindf_coildiff_normdata, traindf_bin_coildiff_data, traindf_bin_coildiff_normdata]
testdfs = [testdf_fcoils_normdata, testdf_bin_fcoils_data, testdf_bin_fcoils_normdata, testdf_coildiff_data, testdf_coildiff_normdata, testdf_bin_coildiff_data, testdf_bin_coildiff_normdata]

# Can pickle DataFrames so we don't have to rebuild anything
traindf_fcoils_data.to_pickle('/home/bennett/Documents/DataSets/bmrb_clean/clean_rings/traindf_fcoils_data.pkl')
traindf_fcoils_normdata.to_pickle('/home/bennett/Documents/DataSets/bmrb_clean/clean_rings/traindf_fcoils_normdata.pkl')
traindf_bin_fcoils_data.to_pickle('/home/bennett/Documents/DataSets/bmrb_clean/clean_rings/traindf_bin_fcoils_data.pkl')
traindf_bin_fcoils_normdata.to_pickle('/home/bennett/Documents/DataSets/bmrb_clean/clean_rings/traindf_bin_fcoils_normdata.pkl')
traindf_coildiff_data.to_pickle('/home/bennett/Documents/DataSets/bmrb_clean/clean_rings/traindf_coildiff_data.pkl')
traindf_coildiff_normdata.to_pickle('/home/bennett/Documents/DataSets/bmrb_clean/clean_rings/traindf_coildiff_normdata.pkl')
traindf_bin_coildiff_data.to_pickle('/home/bennett/Documents/DataSets/bmrb_clean/clean_rings/traindf_bin_coildiff_data.pkl')
traindf_bin_coildiff_normdata.to_pickle('/home/bennett/Documents/DataSets/bmrb_clean/clean_rings/traindf_bin_coildiff_normdata.pkl')

testdf_fcoils_data.to_pickle('/home/bennett/Documents/DataSets/bmrb_clean/clean_rings/testdf_fcoils_data.pkl')
testdf_fcoils_normdata.to_pickle('/home/bennett/Documents/DataSets/bmrb_clean/clean_rings/testdf_fcoils_normdata.pkl')
testdf_bin_fcoils_data.to_pickle('/home/bennett/Documents/DataSets/bmrb_clean/clean_rings/testdf_bin_fcoils_data.pkl')
testdf_bin_fcoils_normdata.to_pickle('/home/bennett/Documents/DataSets/bmrb_clean/clean_rings/testdf_bin_fcoils_normdata.pkl')
testdf_coildiff_data.to_pickle('/home/bennett/Documents/DataSets/bmrb_clean/clean_rings/testdf_coildiff_data.pkl')
testdf_coildiff_normdata.to_pickle('/home/bennett/Documents/DataSets/bmrb_clean/clean_rings/testdf_coildiff_normdata.pkl')
testdf_bin_coildiff_data.to_pickle('/home/bennett/Documents/DataSets/bmrb_clean/clean_rings/testdf_bin_coildiff_data.pkl')
testdf_bin_coildiff_normdata.to_pickle('/home/bennett/Documents/DataSets/bmrb_clean/clean_rings/testdf_bin_coildiff_normdata.pkl')

# Need to prepare some DataFrames for RNNs that drop the _i-1 and _i+1 columns
reslevel_traindf = traindf_bin_coildiff.drop(im1_cols_bin + ip1_cols_bin, axis=1)
reslevel_testdf = testdf_bin_coildiff.drop(im1_cols_bin + ip1_cols_bin, axis=1)
restraindf_coildiff_norm = traindf_coildiff_norm.drop(im1_cols + ip1_cols, axis=1)
restestdf_coildiff_norm = testdf_coildiff_norm.drop(im1_cols + ip1_cols, axis=1)

# Can read in pickled data
bin_rcoil_diff_traindf_normdata = pd.read_pickle('/Users/kcbennett/Documents/data/ShiftX2/bmrb_clean/bin_rcoil_diff_traindf_normdata.pkl')
bin_rcoil_diff_testdf_normdata = pd.read_pickle('/Users/kcbennett/Documents/data/ShiftX2/bmrb_clean/bin_rcoil_diff_testdf_normdata.pkl')

lossf = 'mean_squared_error'


# We begin with some data processing functions
# Let's write a whitener function to whiten whatever columns are specified
def whitener(columns, data, means=None, stds=None, full=False, test_data=False):
    '''Function to standardize (whiten) the specified columns of the given DataFrame.
    Supports using externally-defined means and deviations as is necessary in the 
    case of transforming test_data
    
    columns = Name(s) of column(s) to be whitened (List)
    data = DataFrame containing the column(s) (Pandas DataFrame)
    means = Means to use to standardize the data (Pandas Series)
    stds = Standard deviations to use to standardize the data (Pandas Series)
    full = Whiten the full DataFrame (Bool)
    test_data = Use external means/stds (Bool)
    '''
    df = data.copy()
    if not test_data:
        if full:
            mean = df.mean()
            std = df.std()
            df = (df - mean) / std
        else:
            mean = df[columns].mean()
            std = df[columns].std()
            df[columns] = (df[columns] - mean) / std
    else:
        df[columns] = (df[columns] - means) / stds
        mean = 0
        std = 1
    return mean, std, df


# Now let's write a couple functions to do variable importance testing.
# First, we'll write an ablater function that runs k-fold cross validation
# on the data with given feature(s) removed.
def varimportance_ablater(feats, k, data, atom, feval, model, mod_args, mod_kwargs):
    '''Function to remove (ablate) specified feature(s) from the data and then perform
    k-fold cross validation with a given set of hyper-parameters.  Measures reliance of
    the entire algorithm (including learning process) on the ablated features by comparing
    to performance on full data
    
    feats = Names of columns (features) to be ablated (List)
    k = Number of folds for the cross validation (Int)
    data = Features and targets (Pandas DataFrame)
    atom = Name of atom for which to predict shifts (Str)
    feval = Function to evaluate rmsd for the model (Function)
    model = Function that takes data and returns a trained model (Function)
    mod_args = List of arguments for model-generating function (List)
    mod_kwargs = Dictionary of keyword arguments for model-generating function (Dict)
    '''
    df = data.copy()
    df = df.drop(feats, axis=1)
    train_rmsd, test_rmsd, train_rmsd_spread, test_rmsd_spread = kfold_crossval(k, data, atom, feval, model, mod_args, mod_kwargs)
    return train_rmsd, test_rmsd, train_rmsd_spread, test_rmsd_spread


# Next we'll write a function that scrambles the values for specified feature(s) and then 
# evaluates the performance of a given model 
def varimportance_shuffler(feats, data, feval, mean, std, mod, atom):
    '''Function to test the reliance of a given model on specified feature(s).  It
    does this by shuffling the values in the data columns associated with the given 
    feature(s) and then evaluating the performance of the model on this shuffled data
    
    feats = Feature(s) to be shuffled (List)
    data = Data consisting of features and targets (Pandas DataFrame)
    feval = Function to evaluate the rmsd (Function)
    mean = Mean of the shifts to be predicted (Float)
    std = Standard deviation of the shifts to be predicted (Float)
    mod = A trained model (Keras Model)
    atom = Name of atom for which to predict shifts (Str)
    '''
    for feat in feats:
        data[feat] = np.random.permutation(data[feat])
    err = feval(mean, std, data, mod, atom)
    return err


# Need a function to convert sequential information from Blosum to binary representation    
def blosum_to_binary(data):
    '''Function to remove columns containing blosum numbers in the provided data frame
    and then insert new columns for the binary sequence representation
    
    data = Data including context information like File ID etc (Pandas DataFrame)
    '''
    bin_seq_map = dict(zip([list(IUPACData.protein_letters_3to1.keys())[i].upper() for i in range(20)], np.identity(20).tolist()))
    df = data.copy()
    df = df.drop(col_blosum, axis=1)

    # First add the new columns, initialized to be NaN everywhere
    for col in bin_seq_cols:
        df[col] = np.nan
    
    # Now go through the df row by row and fill in the correct values
    for i in range(len(data)):
        # Get file ID, residue name, and chain from the given residue as well as neighbors
        fid_i = data.loc[i]['FILE_ID']
        res_i = data.loc[i]['RESNAME']
        chain_i = data.loc[i]['CHAIN']
        if i == 0:
            fid_im1 = None
            res_im1 = None
            chain_im1 = None
            bin_seq_im1 = [0] * 20
        else:
            fid_im1 = data.loc[i-1]['FILE_ID']
            res_im1 = data.loc[i-1]['RESNAME']
            chain_im1 = data.loc[i-1]['CHAIN']
        if i == len(data) - 1:
            fid_ip1 = None
            res_ip1 = None
            chain_ip1 = None
            bin_seq_ip1 = [0] * 20
        else:
            fid_ip1 = data.loc[i+1]['FILE_ID']
            res_ip1 = data.loc[i+1]['RESNAME']
            chain_ip1 = data.loc[i+1]['CHAIN']
        
        # Make binary list for residue i-1
        if fid_im1 == fid_i and chain_im1 == chain_i:
            bin_seq_im1 = bin_seq_map[res_im1]
        else:
            bin_seq_im1 = [0] * 20
        
        # Make binary list for residue i+1
        if fid_ip1 == fid_i and chain_ip1 == chain_i:
            bin_seq_ip1 = bin_seq_map[res_ip1]
        else:
            bin_seq_ip1 = [0] * 20
        
        # Make binary list for residue i
        bin_seq_i = bin_seq_map[res_i]
        
        # Combine binary sequences
        bin_seq = bin_seq_im1 + bin_seq_i + bin_seq_ip1
        df.loc[i, bin_seq_cols] = bin_seq
    
    return df
    

# Need to write a function to add the random coil shifts
def add_rand_coils(data):
    '''Function to add random coil chemical shifts to each residue
    in the DataFrame.
    
    data - Feature and target data (Pandas DataFrame)
    '''
    AAlist = [list(IUPACData.protein_letters_3to1.keys())[i].upper()
          for i in range(20)]
    rc_ala = {}
    rc_ala['N'] = [123.8, 118.7, 120.4, 120.2, 120.3, 108.8, 118.2, 119.9,
                   120.4, 121.8, 119.6, 118.7, 0, 119.8, 120.5, 115.7,
                   113.6, 119.2, 121.3, 120.3]
    rc_ala['H'] = [8.24, (8.32 + 8.43) / 2, 8.34, 8.42, 8.30, 8.33, 8.42, 8.00,
                   8.29, 8.16, 8.28, 8.40, 0, 8.32, 8.23, 8.31, 8.15, 8.03,
                   8.25, 8.12]
    rc_ala['HA'] = [4.32, 4.55, 4.71, 4.64, 4.35, 4.62, 3.96, 4.73, 4.17, 4.32,
                    4.34, 4.48, 4.74, 4.42, 4.34, 4.3, 4.47, 4.35, 4.12, 4.66,
                    4.55]
    rc_ala['C'] = [177.8, 174.6, 176.3, 176.6, 175.8, 174.9, 174.1, 176.4, 176.6,
                   177.6, 176.3, 175.2, 177.3, 176.0, 176.3, 174.6, 174.7, 176.3,
                   176.1, 175.9]
    rc_ala['CA'] = [52.5, (58.2 + 55.4) / 2, 54.2, 56.6, 57.7, 45.1, 55.0, 61.1,
                    56.2, 55.1, 55.4, 53.1, 63.3, 55.7, 56.0, 58.3, 61.8, 62.2,
                    57.5, 57.9]
    rc_ala['CB'] = [19.1, (28 + 41.1) / 2, 41.1, 29.9, 39.6, 0, 29, 38.8, 33.1,
                    42.4, 32.9, 38.9, 32.1, 29.4, 30.9, 63.8, 69.8, 32.9, 29.6,
                    38.8]
    randcoil_ala = {i: dict(zip(AAlist, rc_ala[i])) for i in atom_names}
    # When the residue in question is followed by a Proline, we instead use:
    rc_pro = {}
    rc_pro['N'] = [125, 119.9, 121.4, 121.7, 120.9, 109.1, 118.2, 121.7, 121.6,
                   122.6, 120.7, 119.0, 0, 120.6, 121.3, 116.6, 116.0, 120.5,
                   122.2, 120.8]
    rc_pro['H'] = [8.19, 8.30, 8.31, 8.34, 8.13, 8.21, 8.37, 8.06, 8.18,
                   8.14, 8.25, 8.37, 0, 8.29, 8.2, 8.26, 8.15, 8.02, 8.09,
                   8.1]
    rc_pro['HA'] = [4.62, 4.81, 4.90, 4.64, 4.9, 4.13, 5.0, 4.47, 4.60, 4.63, 4.82,
                    5.0, 4.73, 4.65, 4.65, 4.78, 4.61, 4.44, 4.99, 4.84]
    rc_pro['C'] = [175.9, 173, 175, 174.9, 174.4, 174.5, 172.6, 175.0, 174.8,
                   175.7, 174.6, 173.6, 171.4, 174.4, 174.5, 173.1, 173.2, 174.9,
                   174.8, 174.8]
    rc_pro['CA'] = [50.5, 56.4, 52.2, 54.2, 55.6, 44.5, 53.3, 58.7, 54.2, 53.1,
                    53.3, 51.3, 61.5, 53.7, 54.0, 56.4, 59.8, 59.8, 55.7, 55.8]
    rc_pro['CB'] = [18.1, 27.1, 40.9, 29.2, 39.1, 0, 29.0, 38.7, 32.6, 41.7,
                    32.4, 38.7, 30.9, 28.8, 30.2, 63.3, 69.8, 32.6, 28.9, 38.3]
    randcoil_pro = {i: dict(zip(AAlist, rc_pro[i])) for i in atom_names}
    
    df = data.copy()

    # First add the new columns, initialized to be NaN everywhere
    for col in rcoil_cols:
        df[col] = np.nan
    
    # Now go through the df row by row and fill in the correct values
    for i in range(len(data)):
        # Get file ID, residue name, and chain from the given residue as well as neighbors
        fid_i = data.loc[i]['FILE_ID']
        res_i = data.loc[i]['RESNAME']
        chain_i = data.loc[i]['CHAIN']
        if i == len(data) - 1:
            fid_ip1 = None
            res_ip1 = None
            chain_ip1 = None
        else:
            fid_ip1 = data.loc[i+1]['FILE_ID']
            res_ip1 = data.loc[i+1]['RESNAME']
            chain_ip1 = data.loc[i+1]['CHAIN']
        
        # Set next residue to be PRO iff next residue in the chain is PRO, else ALA
        if fid_ip1 == fid_i and chain_ip1 == chain_i and res_ip1 == 'PRO':
            resnext = 'PRO'
        else:
            resnext = 'ALA'
        
        # Make list of rcoil shifts for each atom for this residue
        if resnext == 'PRO':
            rcoil_shifts = [randcoil_pro[i][res_i] for i in atom_names]
        else:
            rcoil_shifts = [randcoil_ala[i][res_i] for i in atom_names]
        
        # Set row of random coil shifts in the DataFrame
        df.loc[i, rcoil_cols] = rcoil_shifts
    
    return df


# Need to write a function to get targets as differences between raw shifts and the
# random coil and ring current values
def diff_targets(data, rings=True, coils=True):
    '''Function that replaces the shifts column with the difference between the raw
    shifts and the values in the columns given
    
    data = Feature and target data (Pandas DataFrame)
    rings = Subtract ring current columns from shift columns (Bool)
    coils = Subtract random coil columns from shift columns (Bool)
    '''
    df = data.copy()
    if rings:
        df[atom_names] = df[atom_names].values - df[ring_cols].fillna(0).values
        df.drop(ring_cols, axis=1, inplace=True)
    
    if coils:
        df[atom_names] = df[atom_names].values - df[rcoil_cols].values
        df.drop(rcoil_cols, axis=1, inplace=True)
    
    return df


# Need some auxiliary functions to prepare data and generate batches for RNNs

def sep_by_chains(data, atom=None, split=None):
    '''Function to split data by chains and return lists of indices for 
    training and validation data respectively or just lists of indices
    for all chains in the data.
    
    data = Residue-level features and targets (Pandas DataFrame)
    atom = Name of atom - used to ensure that the split contains
           the appropriate number of shifts rather than naively
           splitting along chains (Str)
    split = Fraction of data to set aside for validation (Float or None)
    '''
    grouped = data.groupby(['FILE_ID', 'CHAIN'])
    groups = list(grouped.groups.values())
    if split is None:
        return groups # If we don't need to split into training/validation, just return list of chains
    else:
        dat = data[data[atom].notnull()]
        num_shifts = len(dat) # Find the total number of shifts for this atom type
        shuff_groups = random.sample(groups, len(groups)) # Shuffle groups to randomize order
        i = 0 # Initialize index
        p = 0 # Initialize percent of data set aside
        val_shifts = 0 # Initialize number of validation shifts to use
        val_list = [] # Initialize list of lists of indices
        train_list = shuff_groups.copy()
        while p < split:
            val_list.append(shuff_groups[i]) # Append the i'th chain to list of val indices
            chain = data.iloc[shuff_groups[i]] # DataFrame for i'th chain
            new_shifts = len(chain[chain[atom].notnull()])
            val_shifts += new_shifts
            p = val_shifts / num_shifts
            i += 1
        del train_list[:i]
        return train_list, val_list
            

def chain_batch_generator(data, idxs, atom, window, norm_shifts=(0, 1), rings=True, rcoil=False, sample_weights=True, randomize=False):
    '''Takes full DataFrame of all train and validation chains
    along with a list of index lists for the various chains to be batched
    and the window (length of sub-samples).
    
    data = Feature and target data (Pandas DataFrame)
    idxs = List of index objects each specifying all the residues in a given chain (List)
    atom = Name of atom --needed to normalize shifts (Str)
    window = Length of subsequences into which chains are to be chopped for batch training (Int)
    norm_shifts = Normalize target shifts (Bool)
    rings = Ring current columns are included as features (Bool)
    rcoil = Random coil columns are included as features (Bool)
    sample_weights = Return sample weights of same shape as target array with entries of 1 for all
                     timesteps (sequence entries) except 0's for timesteps with no shift data (NumpyArray)
    randomize = Return batches with order of subsequences randomized - to test if state information 
                between subsequences is being (usefully) implemented (Bool)
    '''
    dat = data.copy()
    dat.drop(['Unnamed: 0', 'FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM', 'CHAIN'], axis=1, inplace=True)
    
    if norm_shifts:
        #all_shifts = dat[atom].fillna(0)
        #shifts_mean = all_shifts.mean()
        #shifts_std = all_shifts.std()
        shifts_mean = norm_shifts[0]
        shifts_std = norm_shifts[1]

    while True:
        # Need to go through list of chains in order
        for i, chain_idxs in enumerate(idxs):
            chain = dat.iloc[chain_idxs]
            weights = chain[atom].notnull()
            weights *= 1
            weights = weights.values
            l = len(chain)
            shifts = chain[atom]
            feats = chain.drop(atom_names, axis=1)
            
            if norm_shifts:
                shift_norm = (shifts - shifts_mean) / shifts_std
                shift_norm = shift_norm.values
            else:
                shift_norm = shifts.values
            # ATTN: Normalizing the shifts like this makes the formerly NaN values
            # now a non-zero number in general.  Should be ok if we use sample_weights
            # shift_norm = shift_norm * weights # reset formerly NaN shifts to zero
            shift_norm = np.nan_to_num(shift_norm)
        
            
            # Need to drop the random coil and ring current columns for the other atoms if 
            # such columns are in the data.
            if rings:
                ring_col = atom + '_RC'
                rem1 = ring_cols.copy()
                rem1.remove(ring_col)
                feats = feats.drop(rem1, axis=1)
                feats[ring_col] = feats[ring_col].fillna(value=0)
            if rcoil:
                rcoil_col = 'RCOIL_' + atom
                rem2 = rcoil_cols.copy()
                rem2.remove(rcoil_col)
                feats = feats.drop(rem2, axis=1)
            feats = feats.values
            num_feats = feats.shape[1]
            
            # Find the number of sub-sequences in this batch
            remainder = l % window
            if remainder == 0:
                n = l // window
                even = True # Just in case needed for some future purpose
            else:
                n = l // window + 1
                even = False
                # Fill out arrays to full size
                # full_size = n * window  # NOT NEEDED
                padding = window - remainder
                shift_norm = np.pad(shift_norm, (0, padding), mode='constant')
                feats = np.pad(feats, ((0, padding), (0, 0)), mode='constant')
                weights = np.pad(weights, (0, padding), mode='constant')
            
            # Reshape into subsequences for batch
            shift_norm = shift_norm.reshape((n, window, 1))
            feats = feats.reshape((n, window, num_feats))
            weights = weights.reshape((n, window))
            if np.array_equal(weights, np.zeros_like(weights)):
                pass
            else:
                if sample_weights:
                    if randomize:
                        np.random.seed(1)
                        np.random.shuffle(shift_norm)
                        np.random.seed(1)
                        np.random.shuffle(feats)
                        np.random.seed(1)
                        np.random.shuffle(weights)
                        yield feats, shift_norm, weights
                    else:
                        yield feats, shift_norm, weights
                else:
                    yield feats, shift_norm


def sparta_model(data, atom, epochs, per=5, tol=1.0, pretrain=None, rings=False, rcoil=False):
    '''Constructs a model from the given features and shifts for
    the requested atom.  The model is trained for the given number
    of epochs with the loss being checked every per epochs.  Training
    stops when this loss increases by more than tol. The architecture
    matches SPARTA+ (i.e., a single layer of 30 neurons).
    
    data = Feature and target data (Pandas DataFrame)
    atom = Name of atom to predict (Str)
    epochs = Maximum number of epochs to train
    per = Number of epochs in a test strip for pretraining (Int)
    tol = Pretraining parameter (Float)
    pretrain = Whether or not and how to do pretraining.  Accepts
    None, GL, PQ, or UP (Str)
    
    -For pretraining information, see L. Prechelt, "Early Stopping -- but when?",
    Neural Networks: Tricks of the trade. Springer, Berlin, Heidelberg, 1998. 55-69.
    '''
    dat = data[data[atom].notnull()]
    feats = dat.drop(atom_names, axis=1)
    shifts = dat[atom]
    shifts_mean = shifts.mean()
    shifts_std = shifts.std()
    shift_norm = (shifts - shifts_mean) / shifts_std
    shift_norm = shift_norm.values
    
    if rings:
        ring_col = atom + '_RC'
        rem1 = ring_cols.copy()
        rem1.remove(ring_col)
        feats = feats.drop(rem1, axis=1)
        feats[ring_col] = feats[ring_col].fillna(value=0)
    if rcoil:
        rcoil_col = 'RCOIL_' + atom
        rem2 = rcoil_cols.copy()
        rem2.remove(rcoil_col)
        feats = feats.drop(rem2, axis=1)

    feats = feats.values
    feat_train, feat_val, shift_train, shift_val = skl.model_selection.train_test_split(feats, shift_norm, test_size=0.2)
    dim_in = feats.shape[1]

    # Build model
    mod = keras.models.Sequential()
    mod.add(keras.layers.Dense(units=30, activation='tanh', input_dim=dim_in))
    mod.add(keras.layers.Dense(units=1, activation='linear'))
    mod.compile(loss='mean_squared_error', optimizer='sgd')

    # Initialize some outputs
    hist_list = []
    val_list = []
    param_list = []

    # Do pretraining to determine the best number of epochs for training
    val_min = 10 ** 10
    up_count = 0
    if pretrain is not None:
        for i in range(int(epochs/per)):
            pt1 = mod.evaluate(feat_val, shift_val, verbose=0)
            val_list.append(pt1)
    
            if pt1 < val_min:
                val_min = pt1
    
            hist = mod.fit(feat_train, shift_train, batch_size=64, epochs=per)
            hist_list += hist.history['loss']
            pt2 = mod.evaluate(feat_val, shift_val, verbose=0)
            
            if pretrain == 'GL' or pretrain == 'PQ':
                gl = 100 * (pt2/val_min - 1)
                
                if pretrain == 'GL':
                    param_list.append(gl)
                    if gl > tol:
                        print('Broke loop at round ' + str(i))
                        break
                
                if pretrain == 'PQ':
                    strip_avg = np.array(hist.history['loss']).mean()
                    strip_min = min(np.array(hist.history['loss']))
                    p = 1000 * (strip_avg / strip_min - 1)
                    pq = gl / p
                    param_list.append(pq)
                    if pq > tol:
                        print('Broke loop at round ' + str(i))
                        break
            
            if pretrain == 'UP':
                if pt2 > pt1:
                    up_count += 1
                    param_list.append(up_count)
                    if up_count >= tol:
                        print('Broke loop at round ' + str(i))
                        break
                else:
                    up_count = 0
                    param_list.append(up_count)
            
            print('The validation loss at round ' + str(i) + ' is ' + str(pt2))

        min_val_idx = min((val, idx) for (idx, val) in enumerate(val_list))[1]
        val_epochs = min_val_idx * per
    else:
        val_epochs = epochs


    # Retrain model on full dataset for same number of epochs as was
    # needed to obtain min validation loss
    mod = keras.models.Sequential()
    mod.add(keras.layers.Dense(units=30, activation='tanh', input_dim=dim_in))
    mod.add(keras.layers.Dense(units=1, activation='linear'))
    mod.compile(loss='mean_squared_error', optimizer='sgd')
    mod.fit(feats, shift_norm, batch_size=64, epochs=val_epochs)

    return shifts_mean, shifts_std, val_list, hist_list, param_list, mod


def deep_model(data, atom, activ, arch, lrate, mom, dec, max_epochs, min_epochs=1, per=5, tol=1.0, do=0, wreg=0, norm_shifts=True, pretrain=None, bnorm=False, lsuv=False, nest=False, rcoil=False, rings=False):
    '''Constructs a model from the given features and shifts for
    the requested atom.  The model is trained for the given number
    of epochs with the loss being checked every per epochs.  Training
    stops when this loss increases by more than tol. The arch argument
    is a list specifying the number of hidden units at each layer.
    
    data = Feature and target data (Pandas DataFrame)
    atom = Name of atom to predict (Str)
    activ = Activation function (Str)
    arch = List of the number of neurons per layer (List)
    lrate = Learning rate for SGD (Float)
    mom = Momentum for SGD (Float)
    dec = Decay for SGD learning rate (Float)
    min_epochs = Minimum number of epochs to train (Int)
    max_epochs = Maximum number of epochs to train (Int)
    per = Number of epochs in a test strip for pretraining (Int)
    tol = Pretraining parameter (Float)
    do = Dropout percentage between 0 and 1 (Float)
    wreg = Parameter for L1 regularization of dense layers (Float)
    activ = activation for dense layers before prelu layers (should probably always be linear if prelu layers are present)
    pretrain = Whether or not and how to do pretraining.  Accepts None, GL, PQ, or UP (Str)
    bnorm = Use batch normalization (Bool)
    lsuv = Use layer-wise sequential unit variance initialization (Bool)
    nest = Use Nesterov momentum (Bool)
    rings = DataFrame contains ring-current columns (Bool)
    rcoil = DataFrame contains random-coil columns (Bool)
    norm_shifts = Normalize the target shifts rather than using raw values (Bool)    
    '''
    dat = data[data[atom].notnull()]
    feats = dat.drop(atom_names, axis=1)
    shifts = dat[atom]
    
    # Normalize shifts if this is desired
    if norm_shifts:
        shifts_mean = shifts.mean()
        shifts_std = shifts.std()
        shift_norm = (shifts - shifts_mean) / shifts_std
        shift_norm = shift_norm.values
    else:
        shifts_mean = 0
        shifts_std = 1
        shift_norm = shifts.values

    
    # Need to drop the random coil and ring current columns for the other atoms if 
    # such columns are in the data.
    if rings:
        ring_col = atom + '_RC'
        rem1 = ring_cols.copy()
        rem1.remove(ring_col)
        feats = feats.drop(rem1, axis=1)
        feats[ring_col] = feats[ring_col].fillna(value=0)
    if rcoil:
        rcoil_col = 'RCOIL_' + atom
        rem2 = rcoil_cols.copy()
        rem2.remove(rcoil_col)
        feats = feats.drop(rem2, axis=1)
    
# The following is an alternative way to include the rcoil shifts rather than simply normalizing
# those columns.  This is probably equivalent since the rcoil and raw shifts are similarly 
# distributed.  It is thus not currently implemented.
    # If using random coil values as features, standardize them in same way as target shifts
#    if rcoil and rccs_feat:
#        feats[rccs_feats] = (feats[rccs_feats] - shifts_mean) / shifts_std

    # Split up the data into train and validation
    feats = feats.values
    feat_train, feat_val, shift_train, shift_val = skl.model_selection.train_test_split(feats, shift_norm, test_size=0.2)
    dim_in = feats.shape[1]

    
    opt = keras.optimizers.SGD(lr=lrate, momentum=mom, decay=dec, nesterov=nest)
    
    if do == 0:
        dropout = False
    else:
        dropout=True
        
    mod = keras.models.Sequential()
    mod.add(keras.layers.Dense(units=arch[0], activation=activ, input_dim=dim_in, kernel_regularizer=keras.regularizers.l1(wreg)))
    if bnorm:
        mod.add(keras.layers.BatchNormalization())
    if dropout:
        mod.add(keras.layers.Dropout(do))
    for i in arch[1:]:
        mod.add(keras.layers.Dense(units=i, activation=activ, kernel_regularizer=keras.regularizers.l1(wreg)))
        if bnorm:
            mod.add(keras.layers.BatchNormalization())
        if dropout:
            mod.add(keras.layers.Dropout(do))
    mod.add(keras.layers.Dense(units=1, activation='linear'))
    mod.compile(loss='mean_squared_error', optimizer=opt)
    
    if lsuv:
        mod = LSUVinit(mod, feat_train[:64])
        
    # Initialize some outputs
    hist_list = []
    val_list = []
    param_list = []

    # Do pretraining to determine the best number of epochs for training
    val_min = 10 ** 10
    up_count = 0
    if pretrain is not None:
        for i in range(int(epochs/per)):
            pt1 = mod.evaluate(feat_val, shift_val, verbose=0)
            val_list.append(pt1)
    
            if pt1 < val_min:
                val_min = pt1
    
            hist = mod.fit(feat_train, shift_train, batch_size=64, epochs=per)
            hist_list += hist.history['loss']
            pt2 = mod.evaluate(feat_val, shift_val, verbose=0)
            
            if pretrain == 'GL' or pretrain == 'PQ'
                gl = 100 * (pt2/val_min - 1)
                
                if pretrain == 'GL':
                    param_list.append(gl)
                    if gl > tol:
                        print('Broke loop at round ' + str(i))
                        break
                
                if pretrain == 'PQ':
                    strip_avg = np.array(hist.history['loss']).mean()
                    strip_min = min(np.array(hist.history['loss']))
                    p = 1000 * (strip_avg / strip_min - 1)
                    pq = gl / p
                    param_list.append(pq)
                    if pq > tol:
                        print('Broke loop at round ' + str(i))
                        break
            
            if pretrain == 'UP':
                if pt2 > pt1:
                    up_count += 1
                    param_list.append(up_count)
                    if up_count >= tol:
                        print('Broke loop at round ' + str(i))
                        break
                else:
                    up_count = 0
                    param_list.append(up_count)
            
            print('The validation loss at round ' + str(i) + ' is ' + str(pt2))

        min_val_idx = min((val, idx) for (idx, val) in enumerate(val_list))[1]
        val_epochs = max(min_val_idx * per, min_epochs)
    else:
        val_epochs = epochs

    # Retrain model on full dataset for same number of epochs as was
    # needed to obtain min validation loss
    mod = keras.models.Sequential()
    mod.add(keras.layers.Dense(units=arch[0], activation=activ, input_dim=dim_in, kernel_regularizer=keras.regularizers.l1(wreg)))
    if bnorm:
        mod.add(keras.layers.BatchNormalization())
    if dropout:
        mod.add(keras.layers.Dropout(do))
    for i in arch[1:]:
        mod.add(keras.layers.Dense(units=i, activation=activ, kernel_regularizer=keras.regularizers.l1(wreg)))
        if bnorm:
            mod.add(keras.layers.BatchNormalization())
        if dropout:
            mod.add(keras.layers.Dropout(do))
    mod.add(keras.layers.Dense(units=1, activation='linear'))
    mod.compile(loss='mean_squared_error', optimizer=opt)
    mod.fit(feats, shift_norm, batch_size=64, epochs=val_epochs)

    return shifts_mean, shifts_std, val_list, hist_list, param_list, mod


def deep_model_prelu(data, atom, arch, lrate, mom, dec, max_epochs, min_epochs=1, per=5, tol=1.0, do=0, wreg=0, activ='linear', pretrain=None, bnorm=False, lsuv=False, nest=False, rings=False, rcoil=False, norm_shifts=True):
    '''Constructs a model from the given features and shifts for
    the requested atom.  The model is trained for the given number
    of epochs with the loss being checked every per epochs.  Training
    stops when this loss increases by more than tol. The arch argument
    is a list specifying the number of hidden units at each layer.
    
    data = Feature and target data (Pandas DataFrame)
    atom = Name of atom to predict (Str)
    arch = List of the number of neurons per layer (List)
    lrate = Learning rate for SGD (Float)
    mom = Momentum for SGD (Float)
    dec = Decay for SGD learning rate (Float)
    max_epochs = Maximum number of epochs to train (Int)
    min_epochs = Minimum number of epochs to train (Int)
    per = Number of epochs in a test strip for pretraining (Int)
    tol = Pretraining parameter (Float)
    do = Dropout percentage between 0 and 1 (Float)
    wreg = Parameter for L1 regularization of dense layers (Float)
    activ = activation for dense layers before prelu layers (should probably always be linear if prelu layers are present)
    pretrain = Whether or not and how to do pretraining.  Accepts None, GL, PQ, or UP (Str)
    bnorm = Use batch normalization (Bool)
    lsuv = Use layer-wise sequential unit variance initialization (Bool)
    nest = Use Nesterov momentum (Bool)
    rings = DataFrame contains ring-current columns (Bool)
    rcoil = DataFrame contains random-coil columns (Bool)
    norm_shifts = Normalize the target shifts rather than using raw values (Bool)
    '''

    dat = data[data[atom].notnull()]
    feats = dat.drop(atom_names, axis=1)
    shifts = dat[atom]
    
    # Normalize shifts if this is desired
    if norm_shifts:
        shifts_mean = shifts.mean()
        shifts_std = shifts.std()
        shift_norm = (shifts - shifts_mean) / shifts_std
        shift_norm = shift_norm.values
    else:
        shifts_mean = 0
        shifts_std = 1
        shift_norm = shifts.values

    
    # Need to drop the random coil and ring current columns for the other atoms if 
    # such columns are in the data.
    if rings:
        ring_col = atom + '_RC'
        rem1 = ring_cols.copy()
        rem1.remove(ring_col)
        feats = feats.drop(rem1, axis=1)
        feats[ring_col] = feats[ring_col].fillna(value=0)
    if rcoil:
        rcoil_col = 'RCOIL_' + atom
        rem2 = rcoil_cols.copy()
        rem2.remove(rcoil_col)
        feats = feats.drop(rem2, axis=1)
    
# The following is an alternative way to include the rcoil shifts rather than simply normalizing
# those columns.  This is probably equivalent since the rcoil and raw shifts are similarly 
# distributed.  It is thus not currently implemented.
    # If using random coil values as features, standardize them in same way as target shifts
#    if rcoil and rccs_feat:
#        feats[rccs_feats] = (feats[rccs_feats] - shifts_mean) / shifts_std

    # Split up the data into train and validation
    feats = feats.values
    feat_train, feat_val, shift_train, shift_val = skl.model_selection.train_test_split(feats, shift_norm, test_size=0.2)
    dim_in = feats.shape[1]

    
    opt = keras.optimizers.SGD(lr=lrate, momentum=mom, decay=dec, nesterov=nest)
    
    if do == 0:
        dropout = False
    else:
        dropout=True
    
    # Build model
    mod = keras.models.Sequential()
    mod.add(keras.layers.Dense(units=arch[0], activation=activ, input_dim=dim_in, kernel_regularizer=keras.regularizers.l1(wreg)))
    if bnorm:
        mod.add(keras.layers.BatchNormalization())
    mod.add(keras.layers.advanced_activations.PReLU())
    if dropout:
        mod.add(keras.layers.Dropout(do))
    for i in arch[1:]:
        mod.add(keras.layers.Dense(units=i, activation=activ, kernel_regularizer=keras.regularizers.l1(wreg)))
        if bnorm:
            mod.add(keras.layers.BatchNormalization())
        mod.add(keras.layers.advanced_activations.PReLU())
        if dropout:
            mod.add(keras.layers.Dropout(do))
    mod.add(keras.layers.Dense(units=1, activation='linear'))
    mod.compile(loss='mean_squared_error', optimizer=opt)
    
    if lsuv:
        mod = LSUVinit(mod, feat_train[:64])

    # Initialize some outputs
    hist_list = []
    val_list = []
    param_list = []

    # Do pretraining to determine the best number of epochs for training
    val_min = 10 ** 10
    up_count = 0
    if pretrain is not None:
        for i in range(int(max_epochs/per)):
            pt1 = mod.evaluate(feat_val, shift_val, verbose=0)
            val_list.append(pt1)
    
            if pt1 < val_min:
                val_min = pt1
    
            hist = mod.fit(feat_train, shift_train, batch_size=64, epochs=per)
            hist_list += hist.history['loss']
            pt2 = mod.evaluate(feat_val, shift_val, verbose=0)
            
            if pretrain == 'GL' or pretrain == 'PQ':
                gl = 100 * (pt2/val_min - 1)
                
                if pretrain == 'GL':
                    param_list.append(gl)
                    if gl > tol:
                        print('Broke loop at round ' + str(i))
                        break
                
                if pretrain == 'PQ':
                    strip_avg = np.array(hist.history['loss']).mean()
                    strip_min = min(np.array(hist.history['loss']))
                    p = 1000 * (strip_avg / strip_min - 1)
                    pq = gl / p
                    param_list.append(pq)
                    if pq > tol:
                        print('Broke loop at round ' + str(i))
                        break
            
            if pretrain == 'UP':
                if pt2 > pt1:
                    up_count += 1
                    param_list.append(up_count)
                    if up_count >= tol:
                        print('Broke loop at round ' + str(i))
                        break
                else:
                    up_count = 0
                    param_list.append(up_count)
            
            print('The validation loss at round ' + str(i) + ' is ' + str(pt2))

        min_val_idx = min((val, idx) for (idx, val) in enumerate(val_list))[1]
        val_epochs = max(min_val_idx * per, min_epochs)
    else:
        val_epochs = max_epochs

    # Retrain model on full dataset for same number of epochs as was
    # needed to obtain min validation loss
    mod = keras.models.Sequential()
    mod.add(keras.layers.Dense(units=arch[0], activation=activ, input_dim=dim_in, kernel_regularizer=keras.regularizers.l1(wreg)))
    if bnorm:
        mod.add(keras.layers.BatchNormalization())
    mod.add(keras.layers.advanced_activations.PReLU())
    if dropout:
        mod.add(keras.layers.Dropout(do))
    for i in arch[1:]:
        mod.add(keras.layers.Dense(units=i, activation=activ, kernel_regularizer=keras.regularizers.l1(wreg)))
        if bnorm:
            mod.add(keras.layers.BatchNormalization())
        mod.add(keras.layers.advanced_activations.PReLU())
        if dropout:
            mod.add(keras.layers.Dropout(do))
    mod.add(keras.layers.Dense(units=1, activation='linear'))
    mod.compile(loss='mean_squared_error', optimizer=opt)
    mod.fit(feats, shift_norm, batch_size=64, epochs=val_epochs)

    return shifts_mean, shifts_std, val_list, hist_list, param_list, mod


def branch_model(data, atom, activ, arch, lrate, mom, dec, min_epochs, max_epochs, per, tol, do, reg, pretrain=True, bnorm=False, dropout=False, nest=False, rcoil=False, rccs_feat=False, merge_softmax=False):
    '''Constructs a model from the given features and shifts for
    the requested atom.  The model is trained for the given number
    of epochs with the loss being checked every per epochs.  Training
    stops when this loss increases by more than tol. The arch argument
    is a list specifying the number of hidden units at each layer.
    The architecture for this network is split into two branches that
    handle structural and sequential information respectively rather
    than being fully connected.  The arch parameter is thus given as a
    list, the first two elements of which are themselves lists giving 
    the neurons for each layer of each branch and the remaining elements
    giving the neurons for the remainder of the network, after the two 
    branches meet. Parameters rcoil and rccs_feat determine whether 
    random coil chemical shifts are used and, if so, whether they are
    subtracted off or used as features.'''
    
    # First define the column names for easy access
    col_phipsi = ['PHI_'+i for i in ['COS_i-1', 'SIN_i-1']]
    col_phipsi += [i+j for i in ['PHI_', 'PSI_'] for j in ['COS_i', 'SIN_i']]
    col_phipsi += ['PSI_'+i for i in ['COS_i+1', 'SIN_i+1']]
    col_chi = [i+j+k for k in ['_i-1', '_i', '_i+1'] for i in ['CHI1_', 'CHI2_'] for j in ['COS', 'SIN', 'EXISTS']]
    col_hbprev = ['O_'+i+'_i-1' for i in ['d_HA', '_COS_H', '_COS_A', '_EXISTS']]
    col_hbond = [i+j+'_i' for i in ['Ha_', 'HN_', 'O_'] for j in ['d_HA', '_COS_H', '_COS_A', '_EXISTS']]
    col_hbnext = ['HN_'+i+'_i+1' for i in ['d_HA', '_COS_H', '_COS_A', '_EXISTS']]
    col_s2 = ['S2'+i for i in ['_i-1', '_i', '_i+1']]
    struc_cols = col_phipsi + col_chi + col_hbprev + col_hbond + col_hbnext + col_s2
    blosum_names = ['BLOSUM62_NUM_'+list(IUPACData.protein_letters_3to1.keys())[i].upper() for i in range(20)]
    col_blosum = [blosum_names[i]+j for j in ['_i-1', '_i', '_i+1'] for i in range(20)]
    seq_cols = col_blosum

    # Now seperate out the features and shifts
    dat = data[data[atom].notnull()]
    feats = dat.drop(atom_names, axis=1)
    struc_feats = dat[struc_cols].values
    seq_feats = dat[seq_cols].values
    if rcoil:
        rccs_feats = dat['RC_' + atom]
    if rcoil and not rccs_feat:
        rccs_names = ['RC_' + i for i in atom_names]
        raw_shifts = dat[atom]
        shifts = raw_shifts - rccs_feats
    else:
        shifts = dat[atom]
    
    # Standardize the shifts
    shifts_mean = shifts.mean()
    shifts_std = shifts.std()
    shift_norm = (shifts - shifts_mean) / shifts_std
    
    # For early stopping pretraining, do a train_test_split
    feat_train, feat_val, shift_train, shift_val = skl.model_selection.train_test_split(feats, shift_norm, test_size=0.2)
    struc_feat_train = feat_train[struc_cols].values
    struc_feat_val = feat_val[struc_cols].values
    seq_feat_train = feat_train[seq_cols].values
    seq_feat_val = feat_val[seq_cols].values
    shift_train = shift_train.values
    shift_val = shift_val.values
    
    # If using random coil chemical shifts as input features, standardize them
    if rcoil and rccs_feat:
        rc_feat_train = feat_train['RC_' + atom].values
        rc_feat_val = feat_val['RC_' + atom].values
        rc_feat_train = (rc_feat_train - shifts_mean) / shifts_std
        rc_feat_val = (rc_feat_val - shifts_mean) / shifts_std
    
    # Define input dimensions and optimization procedure
    struc_dim_in = struc_feat_train.shape[1]
    seq_dim_in = seq_feat_train.shape[1]
    opt = keras.optimizers.SGD(lr=lrate, momentum=mom, decay=dec, nesterov=nest)
    
    # Build model
    # Process structure and sequence information seperately
    struc_in = keras.layers.Input(shape=(struc_dim_in,), name='struc_input')
    struc = keras.layers.Lambda(lambda x: x)(struc_in)
    seq_in = keras.layers.Input(shape=(seq_dim_in,), name='seq_input')
    seq = keras.layers.Lambda(lambda x: x)(seq_in)
    for i in arch[0]:
        struc = keras.layers.Dense(i, activation=activ, kernel_regularizer=keras.regularizers.l1(reg[0]))(struc)
        if bnorm:
            struc = keras.layers.BatchNormalization()(struc)
        if dropout:
            struc = keras.layers.Dropout(do)(struc)
    for i in arch[1]:
        seq = keras.layers.Dense(i, activation=activ, kernel_regularizer=keras.regularizers.l1(reg[1]))(seq)
        if bnorm:
            struc = keras.layers.BatchNormalization()(seq)
        if dropout:
            struc = keras.layers.Dropout(do)(seq)

    # Concatenate structure and sequence (and random coil) information
    if rcoil and rccs_feat:
        rccs = keras.layers.Input(shape=(1,), name='rccs_input')
        merge = keras.layers.concatenate([struc, seq, rccs])
    else:
        merge = keras.layers.concatenate([struc, seq])
    
    # Process merged info, either a single layer with softmax or 
    # multiple layers with same network-wide activation function
    if merge_softmax:
        merge = keras.layers.Dense(arch[2], activation='softmax', kernel_regularizer=keras.regularizers.l1(reg[2]))(merge)
    else:
        for i in arch[2:]:
            merge = keras.layers.Dense(i, activation=activ, kernel_regularizer=keras.regularizers.l1(reg[2]))(merge)
    
    predicted_shift = keras.layers.Dense(units=1, activation='linear')(merge)
    if rcoil and rccs_feat:
        model = keras.Model(inputs=[struc_in, seq_in, rccs], outputs=predicted_shift)
    else:
        model = keras.Model(inputs=[struc_in, seq_in], outputs=predicted_shift)

    model.compile(loss='mean_squared_error', optimizer=opt)

    # Initialize some outputs
    hist_list = []
    val_list = []

    # Train until the validation loss gets too far above the observed min
    val_min = 10 ** 10
    if pretrain:
        for i in range(int(max_epochs/per)):
            if rcoil and rccs_feat:
                pt1 = model.evaluate([struc_feat_val, seq_feat_val, rc_feat_val], shift_val, verbose=0)
                hist = model.fit([struc_feat_train, seq_feat_train, rc_feat_train], shift_train, batch_size=64, epochs=per)
                pt2 = model.evaluate([struc_feat_val, seq_feat_val, rc_feat_val], shift_val, verbose=0)
            else:
                pt1 = model.evaluate([struc_feat_val, seq_feat_val], shift_val, verbose=0)
                hist = model.fit([struc_feat_train, seq_feat_train], shift_train, batch_size=64, epochs=per)
                pt2 = model.evaluate([struc_feat_val, seq_feat_val], shift_val, verbose=0)
            val_list.append(pt1)
            if pt1 < val_min:
                val_min = pt1
            hist_list += hist.history['loss']
            delt1 = pt1 - val_min
            delt2 = pt2 - val_min
            print('The validation loss at round ' + str(i) + ' is ' + str(pt2))
            if delt1 > tol and delt2 > tol:
                print('Broke loop at round ' + str(i))
                break
            if pt2 is np.nan:
                print('Broke loop because of NaN')
                break
        min_val_idx = min((val, idx) for (idx, val) in enumerate(val_list))[1]
        val_epochs = max(min_val_idx * per, min_epochs)
    else:
        val_epochs = max_epochs
    # Retrain model on full dataset for same number of epochs as was
    # needed to obtain min validation loss
    struc_in = keras.layers.Input(shape=(struc_dim_in,), name='struc_input')
    struc = keras.layers.Lambda(lambda x: x)(struc_in)
    seq_in = keras.layers.Input(shape=(seq_dim_in,), name='seq_input')
    seq = keras.layers.Lambda(lambda x: x)(seq_in)

    for i in arch[0]:
        struc = keras.layers.Dense(i, activation=activ, kernel_regularizer=keras.regularizers.l1(reg[0]))(struc)
        if bnorm:
            struc = keras.layers.BatchNormalization()(struc)
        if dropout:
            struc = keras.layers.Dropout(do)(struc)
    for i in arch[1]:
        seq = keras.layers.Dense(i, activation=activ, kernel_regularizer=keras.regularizers.l1(reg[1]))(seq)
        if bnorm:
            struc = keras.layers.BatchNormalization()(seq)
        if dropout:
            struc = keras.layers.Dropout(do)(seq)

    # Concatenate structure and sequence (and random coil) information
    if rcoil and rccs_feat:
        rccs = keras.layers.Input(shape=(1,), name='rccs_input')
        merge = keras.layers.concatenate([struc, seq, rccs])
    else:
        merge = keras.layers.concatenate([struc, seq])
    
    # Process merged info, either a single layer with softmax or 
    # multiple layers with same network-wide activation function
    if merge_softmax:
        merge = keras.layers.Dense(arch[2], activation='softmax', kernel_regularizer=keras.regularizers.l1(reg[2]))(merge)
    else:
        for i in arch[2:]:
            merge = keras.layers.Dense(i, activation=activ, kernel_regularizer=keras.regularizers.l1(reg[2]))(merge)
    
    predicted_shift = keras.layers.Dense(units=1, activation='linear')(merge)
    if rcoil and rccs_feat:
        model = keras.Model(inputs=[struc_in, seq_in, rccs], outputs=predicted_shift)
    else:
        model = keras.Model(inputs=[struc_in, seq_in], outputs=predicted_shift)

    model.compile(loss='mean_squared_error', optimizer=opt)
    
    if rcoil and rccs_feat:
        model.fit([struc_feats, seq_feats, rccs_feats], shift_norm, batch_size=64, epochs=val_epochs)
    else:
        model.fit([struc_feats, seq_feats], shift_norm, batch_size=64, epochs=val_epochs)

    return shifts_mean, shifts_std, val_list, hist_list, model


def bidir_lstm_model(data, atom, arch=[[30]], opt='rmsprop', epochs=100, min_epochs=1, per=5, tol=2, do=0.0, lstm_do=0.0, rec_do=0.0, val_split=0.2, window=10, rings=True, rcoil=False, pretrain=None, norm_shifts=True, activ='linear', prelu=False, randomize=False):

    dat = data.copy()
    feats = dat.drop(atom_names, axis=1)
    feats = feats.drop(['Unnamed: 0', 'FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM', 'CHAIN'], axis=1)
    # Need to drop the random coil and ring current columns for the other atoms if 
    # such columns are in the data.
    if rings:
        ring_col = atom + '_RC'
        rem1 = ring_cols.copy()
        rem1.remove(ring_col)
        feats = feats.drop(rem1, axis=1)
        feats[ring_col] = feats[ring_col].fillna(value=0)
    if rcoil:
        rcoil_col = 'RCOIL_' + atom
        rem2 = rcoil_cols.copy()
        rem2.remove(rcoil_col)
        feats = feats.drop(rem2, axis=1)
    num_feats = feats.shape[1]
    
    if do > 0:
        dropout=True
    else:
        dropout=False
    
    # Get shift statistics
    if norm_shifts:
        all_shifts = data[data[atom].notnull()]
        all_shifts = all_shifts[atom]
        shifts_mean = all_shifts.mean()
        shifts_std = all_shifts.std()

    # Split the data by chain and train/validation sets
    train_set, val_set = sep_by_chains(data, atom=atom, split=val_split)
    full_set = train_set + val_set
    
    # Create generators for training and validation data as well as full data
    if randomize:
        train_gen = chain_batch_generator(data, train_set, atom, window, norm_shifts=(shifts_mean, shifts_std), rings=rings, rcoil=rcoil, randomize=True)
        val_gen = chain_batch_generator(data, val_set, atom, window, norm_shifts=(shifts_mean, shifts_std), rings=rings, rcoil=rcoil, randomize=True)
        full_gen = chain_batch_generator(data, full_set, atom, window, norm_shifts=(shifts_mean, shifts_std), rings=rings, rcoil=rcoil, randomize=True)
    else:
        train_gen = chain_batch_generator(data, train_set, atom, window, norm_shifts=(shifts_mean, shifts_std), rings=rings, rcoil=rcoil)
        val_gen = chain_batch_generator(data, val_set, atom, window, norm_shifts=(shifts_mean, shifts_std), rings=rings, rcoil=rcoil)
        full_gen = chain_batch_generator(data, full_set, atom, window, norm_shifts=(shifts_mean, shifts_std), rings=rings, rcoil=rcoil)
    
    # Count number of empty examples in train, val, and full sets
    train_count = 0
    val_count = 0
    for i, chain_idx in enumerate(train_set):
        chain = data.iloc[chain_idx]
        weights = chain[atom].notnull()
        weights *= 1
        if np.array_equal(weights, np.zeros_like(weights)):
            train_count += 1
    for i, chain_idx in enumerate(val_set):
        chain = data.iloc[chain_idx]
        weights = chain[atom].notnull()
        weights *= 1
        if np.array_equal(weights, np.zeros_like(weights)):
            val_count += 1
    full_count = train_count + val_count
    train_steps = len(train_set) - train_count
    val_steps = len(val_set) - val_count
    full_steps = len(full_set) - full_count
    
    # Build model
    mod = keras.models.Sequential()
    for num_nodes in arch[0]:
        mod.add(keras.layers.Bidirectional(keras.layers.LSTM(num_nodes, dropout=lstm_do, recurrent_dropout=rec_do, return_sequences=True), batch_input_shape=(None, window, num_feats)))
        if dropout:
            mod.add(keras.layers.TimeDistributed(keras.layers.Dropout(do)))
    for num_nodes in arch[1:]:
        mod.add(keras.layers.TimeDistributed(keras.layers.Dense(units=num_nodes, activation=activ)))
        if prelu:
            mod.add(keras.layers.TimeDistributed(keras.layers.advanced_activations.PReLU()))
        if dropout:
            mod.add(keras.layers.TimeDistributed(keras.layers.Dropout(do)))
    mod.add(keras.layers.TimeDistributed(keras.layers.Dense(1, activation='linear')))
    mod.compile(loss='mean_squared_error', optimizer=opt, sample_weight_mode='temporal')
    
    # Initialize some outputs
    hist_list = []
    val_list = []
    param_list = []
    
    # Do pretraining to determine the best number of epochs for training
    val_min = 10 ** 10
    up_count = 0
    if pretrain is not None:
        for i in range(int(epochs/per)):
            pt1 = mod.evaluate_generator(val_gen, steps=val_steps)
            val_list.append(pt1)
    
            if pt1 < val_min:
                val_min = pt1
    
            hist = mod.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=per)
            hist_list += hist.history['loss']
            pt2 = mod.evaluate_generator(val_gen, steps=val_steps)
            
            if pretrain == 'GL' or pretrain == 'PQ':
                gl = 100 * (pt2/val_min - 1)
                
                if pretrain == 'GL':
                    param_list.append(gl)
                    if gl > tol:
                        print('Broke loop at round ' + str(i))
                        break
                
                if pretrain == 'PQ':
                    strip_avg = np.array(hist.history['loss']).mean()
                    strip_min = min(np.array(hist.history['loss']))
                    p = 1000 * (strip_avg / strip_min - 1)
                    pq = gl / p
                    param_list.append(pq)
                    if pq > tol:
                        print('Broke loop at round ' + str(i))
                        break
            
            if pretrain == 'UP':
                if pt2 > pt1:
                    up_count += 1
                    param_list.append(up_count)
                    if up_count >= tol:
                        print('Broke loop at round ' + str(i))
                        break
                else:
                    up_count = 0
                    param_list.append(up_count)
            
            print('The validation loss at round ' + str(i) + ' is ' + str(pt2))

        min_val_idx = min((val, idx) for (idx, val) in enumerate(val_list))[1]
        val_epochs = max(min_val_idx * per, min_epochs)
    else:
        val_epochs = epochs
    
    # Rebuild model
    mod = keras.models.Sequential()
    for num_nodes in arch[0]:
        mod.add(keras.layers.Bidirectional(keras.layers.LSTM(num_nodes, dropout=lstm_do, recurrent_dropout=rec_do, return_sequences=True), batch_input_shape=(None, window, num_feats)))
        if do:
            mod.add(keras.layers.TimeDistributed(keras.layers.Dropout(do)))
    for num_nodes in arch[1:]:
        mod.add(keras.layers.TimeDistributed(keras.layers.Dense(units=num_nodes, activation=activ)))
        if prelu:
            mod.add(keras.layers.TimeDistributed(keras.layers.advanced_activations.PReLU()))
        if do:
            mod.add(keras.layers.TimeDistributed(keras.layers.Dropout(do)))
    mod.add(keras.layers.TimeDistributed(keras.layers.Dense(1, activation='linear')))
    mod.compile(loss='mean_squared_error', optimizer=opt, sample_weight_mode='temporal')
    mod.fit_generator(full_gen, steps_per_epoch=full_steps, epochs=val_epochs)
    return shifts_mean, shifts_std, val_list, hist_list, param_list, mod
    
    
# Here, we build some evaluators to combine operations needed to get the rmsd of different types of models
def sparta_eval(mean, std, data, model, atom, rings=False, rcoils=False):
    dat = data[data[atom].notnull()]
    feats = dat.drop(atom_names, axis=1)
    
    # If ring currents or random coil shifts are in the data, drop 
    # those from the other atoms and fill any NaNs to 0
    if rings:
        ring_col = atom + '_RC'
        rem1 = ring_cols.copy()
        rem1.remove(ring_col)
        feats = feats.drop(rem1, axis=1)
        feats[ring_col] = feats[ring_col].fillna(value=0)
    if rcoils:
        rcoil_col = 'RCOIL_' + atom
        rem2 = rcoil_cols.copy()
        rem2.remove(rcoil_col)
        feats = feats.drop(rem2, axis=1)
    
    feats = feats.values    
    shifts = dat[atom].values
    shifts_norm = (shifts - mean) / std
    mod_eval = model.evaluate(feats, shifts_norm, verbose=0)
    return np.sqrt(mod_eval) * std

def branch_eval(mean, std, data, model, atom):
    dat = data[data[atom].notnull()]
    feats = dat.drop(atom_names, axis=1).values
    struc_feats = dat[struc_cols].values
    seq_feats = dat[seq_cols].values
    shifts = dat[atom].values
    shifts_norm = (shifts - mean) / std
    mod_eval = model.evaluate([struc_feats, seq_feats], shifts_norm, verbose=0)
    return np.sqrt(mod_eval) * std
    
def lstm_eval(mean, std, data, model, atom, window=10, randomize=False):
    idxs = sep_by_chains(data, atom)
    if randomize:
        gen = chain_batch_generator(data, atom=atom, idxs=idxs, window=window, norm_shifts=(mean, std), randomize=True)
    else:
        gen = chain_batch_generator(data, atom=atom, idxs=idxs, window=window, norm_shifts=(mean, std))
    count = 0
    for i, chain_idx in enumerate(idxs):
        chain = data.iloc[chain_idx]
        weights = chain[atom].notnull()
        weights *= 1
        if np.array_equal(weights, np.zeros_like(weights)):
            count += 1
    steps = len(idxs) - count
    err = np.sqrt(model.evaluate_generator(gen, steps=steps)) * std
    return err


# Need to write a function that does k-fold cross-validation 
def kfold_crossval(k, data, atom, feval, model, mod_args, mod_kwargs, out='Summary', lstm=False, window=10):
    '''Function to perform k-fold cross validation on a given model with fixed
    hyper-parameters.
    
    k = Number of folds for the cross validation (Int)
    data = Features and targets (Pandas DataFrame)
    atom = Name of atom for which to predict shifts (Str)
    feval = Function to evaluate rmsd for the model (Function)
    model = Function that takes data and returns a trained model (Function)
    mod_args = List of arguments for model-generating function (List)
    mod_kwargs = Dictionary of keyword arguments for model-generating function (Dict)
    lstm = Assume model takes data like an lstm (by chain) rather than by residue (Bool)
    
    '''
    test_rmsd_list = np.array([])
    train_rmsd_list = np.array([])
    if lstm:
        idxs = sep_by_chains(data, atom)
        kf = skl.model_selection.KFold(n_splits=k)
        test_rmsd_list = np.array([])
        train_rmsd_list = np.array([])
        for train_selector, test_selector in kf.split(idxs):
            train_idxs, test_idxs = np.take(idxs, train_selector), np.take(idxs, test_selector)
            full_train_idx, full_test_idx = train_idxs[0], test_idxs[0]
            for new_idx in train_idxs[1:]:
                full_train_idx = full_train_idx.append(new_idx)
            for new_idx in test_idxs[1:]:
                full_test_idx = full_test_idx.append(new_idx)
            traindf, testdf = data.iloc[full_train_idx], data.iloc[full_test_idx]
            mean, std, val_list, history, param_list, mod = model(traindf, atom, **mod_kwargs)
            test_rmsd = feval(mean, std, testdf, mod, atom, window=window)
            train_rmsd = feval(mean, std, traindf, mod, atom, window=window)
            test_rmsd_list = np.append(test_rmsd_list, test_rmsd)
            train_rmsd_list = np.append(train_rmsd_list, train_rmsd)
    
    else:        
        kf = skl.model_selection.KFold(n_splits=k)
        for train_index, test_index in kf.split(data):
            train_df, test_df = data.iloc[train_index], data.iloc[test_index]
            mean, std, val_list, history, mod = model(train_df, atom, *mod_args, **mod_kwargs)
            test_rmsd = feval(mean, std, test_df, mod, atom)
            test_rmsd_list = np.append(test_rmsd_list, test_rmsd)
            train_rmsd = feval(mean, std, train_df, mod, atom)
            train_rmsd_list = np.append(train_rmsd_list, train_rmsd)

    train_rmsd = train_rmsd_list.mean()
    train_rmsd_spread = train_rmsd_list.std()
    test_rmsd = test_rmsd_list.mean()
    test_rmsd_spread = test_rmsd_list.std()
    if out=='summary':
        return train_rmsd, test_rmsd, train_rmsd_spread, test_rmsd_spread
    if out=='full':
        return train_rmsd_list, test_rmsd_list
        
        
# We can check that this works
mod_args = [[100, 60, 30], 0.01, 0.70, 5*10**-6, 100, 2000, 25, 10**-2]
mod_kwargs = {'do' : 0.2, 'wreg' : 0, 'activ' : 'linear', 'pretrain' : True, 'bnorm' : True, 'nest' : True, 'lsuv' : True, 'rcoil' : False, 'rccs_feat' : False}
ret = kfold_crossval(3, combined_clean_bmrb, 'C', sparta_eval, deep_model_prelu, mod_args, mod_kwargs)


# Note, the alternative (manual) evaluation of the rmsd below doesn't work
# unless you reshape the arrays to be of the same size as shifts is (N,) 
# while preds is (N,1).  I've discovered this at least twice now so don't forget!

#def sparta_eval2(mean, std, data, model, atom):
#    dat = data[data[atom].notnull()]
#    feats = dat.drop(atom_names, axis=1).values
#    shifts = dat[atom].values
#    preds = model.predict(feats)
#    preds = preds * std + mean
#    sq = (preds - shifts) ** 2
#    return sq.mean()
#
#
#def sparta_eval3(mean, std, data, model, atom):
#    dat = data[data[atom].notnull()]
#    feats = dat.drop(atom_names, axis=1).values
#    shifts = dat[atom].values
#    shifts_norm = (shifts - mean) / std
#    preds = model.predict(feats)
#    sq = (preds - shifts_norm) ** 2
#    return sq.mean()


# Below are some random tests to check errors for different nets by hand
cmean, cstd, cval_list, chist_list, sparta_cmod = sparta_model(train_data, 'C', 25000, 100, 5*10**-3)
hmean, hstd, hval_list, hhist_list, sparta_hmod = sparta_model(train_data, 'H', 25000, 100, 5*10**-3)
nmean, nstd, nval_list, nhist_list, sparta_nmod = sparta_model(train_data, 'N', 25000, 100, 5*10**-3)
camean, castd, caval_list, cahist_list, sparta_camod = sparta_model(train_data, 'CA', 25000, 100, 5*10**-3)
cbmean, cbstd, cbval_list, cbhist_list, sparta_cbmod = sparta_model(train_data, 'CB', 25000, 100, 10**-2)
hamean, hastd, haval_list, hahist_list, sparta_hamod = sparta_model(train_data, 'HA', 25000, 100, 5*10**-3)

cmean, cstd, redo_cval_list3, redo_deep_chist_list3, deep_mod_relu= deep_model(bin_rcoil_diff_traindf_normdata, 'CB', 'relu', [100, 60, 30], 0.01, 0.70, 5*10**-6, 50, 2000, 25, 10**-2, do=0.2, wreg=0, pretrain=True, bnorm=True, nest=True, lsuv=True)
cmean, cstd, cval_list0, chist_list0, cparam_list0, deep_mod1= deep_model_prelu_test(traindf_fcoils_normdata, 'CB', [100, 60, 30], 0.01, 0.70, 5*10**-6, 1000, per=5, tol=10, do=0.2, wreg=0, pretrain='PQ', bnorm=True, nest=True, rcoil=True, rings=True)
cmean, cstd, redo_cval_list2, redo_deep_chist_list2, deep_mod2= deep_model_prelu(train_data, 'C', [100, 60, 30], 0.01, 0.70, 5*10**-6, 50, 2000, 25, 10**-2, do=0.2, wreg=0, pretrain=True, bnorm=True, dropout=True, nest=True, lsuv=True)


branch_arch = [[60, 50, 30], [60, 50, 30], 30]
cmean, cstd, redo_cval_list3, redo_deep_chist_list3, branch_mod1 = branch_model('relu', branch_arch, 0.01, 0.70, 5*10**-6, train_data, 'C', 50, 2000, 25, 10**-2, do=0.2, reg=0, pretrain=True, bnorm=True, dropout=True, nest=True)

c_error = sparta_eval(cmean, cstd, testdf_fcoils_normdata, deep_mod1, 'CB', rings=True, rcoils=True)
h_error = sparta_eval(hmean, hstd, test_data, sparta_hmod, 'H')
n_error = sparta_eval(nmean, nstd, test_data, sparta_nmod, 'N')
ca_error = sparta_eval(camean, castd, test_data, sparta_camod, 'CA')
cb_error = sparta_eval(cbmean, cbstd, test_data, sparta_cbmod, 'CB')
ha_error = sparta_eval(hamean, hastd, test_data, sparta_hamod, 'HA')

training_error = branch_eval(cmean, cstd, train_data, branch_mod1, 'C')
testdat_error = branch_eval(cmean, cstd, test_data, branch_mod1, 'C')

sparta_eval(nmean, nstd, train_data, sparta_nmod, 'N')
sparta_eval(cmean, cstd, test_data, sparta_cmod, 'C')
sparta_eval(hmean, hstd, train_data, sparta_hmod, 'H')
sparta_eval(camean, castd, train_data, sparta_camod, 'CA')
sparta_eval(cbmean, cbstd, train_data, sparta_cbmod, 'CB')
sparta_eval(hamean, hastd, train_data, sparta_hamod, 'HA')



# A loop to try different hyper-parameters on dense nets
archs = [[100, 60, 30]]
drops = [0.4]
lrates = [0.005, 0.01, 0.05, 0.1]
decays = [10**-6, 10**-5, 10**-4]
momenta = [.7, .9, .95]
activs = ['tanh', 'relu']
nester = [True]

results = []
for ip in drops:
    for i in lrates:
        for j in decays:
            for k in activs:
                for l in momenta:
                    for m in nester:
                        cmean, cstd, cval_list3, deep_chist_list3, deep100603015_cmod_bndo = deep_model(k, [100, 60, 30], i, l, j, train_data, 'C', 
                                                                                            min_epochs=50, max_epochs=2000, per=25, tol=10**-2, do=ip, reg=0.01, bnorm=True, dropout=True, nest=m)
                        min_val_idx = min((val, idx) for (idx, val) in enumerate(cval_list3))[1]
                        val_epochs = max(min_val_idx * 25, 50)
                        training_error = np.sqrt(sparta_eval(cmean, cstd, train_data, deep100603015_cmod_bndo, 'C')) * cstd
                        testdat_error = np.sqrt(sparta_eval(cmean, cstd, test_data, deep100603015_cmod_bndo, 'C')) * cstd
                        res = [[100, 60, 30], i, j, k, l, m, ip, val_epochs, training_error, testdat_error]
                        print('Finished round ' + str(res))
                        results.append(res)
        resdf = pd.DataFrame(results, columns=['Architecture', 'Learning_Rate', 'Decay', 'Activation', 'Momentum', 'Nesterov', 'Dropout', 'Epochs', 'Training_Error', 'Test_Error'])
        resdf.to_pickle('/Users/kcbennett/Documents/Git_Collaborations_THG/shiftpred/hyper_search2_reg.pkl')
            
            
results_df = pd.DataFrame(results, columns=['Architecture', 'Learning_Rate', 'Decay', 'Activation', 'Momentum', 'Nesterov', 'Dropout', 'Epochs', 'Training_Error', 'Test_Error'])
results_df.to_pickle('/Users/kcbennett/Documents/Git_Collaborations_THG/shiftpred/hyper_search.pkl')         


# A loop to record results of prelu networks on different representations of the data (normalized, differenced, binary, etc.)     
prelu_results = []
for atom in ['H', 'CA', 'CB', 'C', 'N']:
#    mean, std, val_list, hist_list, mod = sparta_model(clean_bmrb_traindf_data, atom, 800, 25, 5*10**-3)
#    train_err = sparta_eval(mean, std, clean_bmrb_traindf_data, mod, atom)
#    test_err = sparta_eval(mean, std, clean_bmrb_testdf_data, mod, atom)
#    res0 = [atom, 'Our_SPARTA+',  'Blosum62', 'Raw', train_err, test_err]
    mean0, std0, val_list0, hist_list0, param_list0, mod0 = deep_model_prelu_test(traindf_fcoils_normdata, atom, [100, 60, 30], 0.01, 0.70, 5*10**-6, 50, 1000, per=5, tol=2, do=0.2, wreg=0, pretrain='PQ', bnorm=True, nest=True, rings=True, rcoil=True)
    train_err0 = sparta_eval(mean0, std0, traindf_fcoils_normdata, mod0, atom, rings=True, rcoils=True)
    test_err0 = sparta_eval(mean0, std0, testdf_fcoils_normdata, mod0, atom, rings=True, rcoils=True)
    res0 = [atom, 'Deep_Prelu', 'Blosum62', 'Normalized', 'Feat', 'Feat', train_err0, test_err0]

    mean1, std1, val_list1, hist_list1, param_list1, mod1 = deep_model_prelu_test(traindf_bin_fcoils_data, atom, [100, 60, 30], 0.01, 0.70, 5*10**-6, 50, 1000, per=5, tol=2, do=0.2, wreg=0, pretrain='PQ', bnorm=True, nest=True, rings=True, rcoil=True)
    train_err1 = sparta_eval(mean1, std1, traindf_bin_fcoils_normdata, mod1, atom, rings=True, rcoils=True)
    test_err1 = sparta_eval(mean1, std1, testdf_bin_fcoils_normdata, mod1, atom, rings=True, rcoils=True)
    res1 = [atom, 'Deep_Prelu', 'Binary', 'Raw', 'Feat', 'Feat', train_err1, test_err1]

    mean2, std2, val_list2, hist_list2, param_list2, mod2 = deep_model_prelu_test(traindf_bin_fcoils_normdata, atom, [100, 60, 30], 0.01, 0.70, 5*10**-6, 50, 1000, per=5, tol=2, do=0.2, wreg=0, pretrain='PQ', bnorm=True, nest=True, rings=True, rcoil=True)
    train_err2 = sparta_eval(mean2, std2, traindf_bin_fcoils_normdata, mod2, atom, rings=True, rcoils=True)
    test_err2 = sparta_eval(mean2, std2, testdf_bin_fcoils_normdata, mod2, atom, rings=True, rcoils=True)
    res2 = [atom, 'Deep_Prelu', 'Binary', 'Normalized', 'Feat', 'Feat', train_err2, test_err2]

    mean3, std3, val_list3, hist_list3, param_list3, mod3 = deep_model_prelu_test(traindf_coildiff_normdata, atom, [100, 60, 30], 0.01, 0.70, 5*10**-6, 50, 1000, per=5, tol=2, do=0.2, wreg=0, pretrain='PQ', bnorm=True, nest=True, rings=True)
    train_err3 = sparta_eval(mean3, std3, traindf_coildiff_normdata, mod3, atom, rings=True)
    test_err3 = sparta_eval(mean3, std3, testdf_coildiff_normdata, mod3, atom, rings=True)
    res3 = [atom, 'Deep_Prelu', 'Blosum62', 'Normalized', 'Feat', 'Diff', train_err3, test_err3]
    
    mean4, std4, val_list4, hist_list4, param_list4, mod4 = deep_model_prelu_test(traindf_bin_coildiff_data, atom, [100, 60, 30], 0.01, 0.70, 5*10**-6, 50, 1000, per=5, tol=2, do=0.2, wreg=0, pretrain='PQ', bnorm=True, nest=True, rings=True)
    train_err4 = sparta_eval(mean4, std4, traindf_bin_coildiff_data, mod4, atom, rings=True)
    test_err4 = sparta_eval(mean4, std4, testdf_bin_coildiff_data, mod4, atom, rings=True)
    res4 = [atom, 'Deep_Prelu', 'Binary', 'Raw', 'Feat', 'Diff', train_err4, test_err4]
    
    mean5, std5, val_list5, hist_list5, param_list5, mod5 = deep_model_prelu_test(traindf_bin_coildiff_normdata, atom, [100, 60, 30], 0.01, 0.70, 5*10**-6, 50, 1000, per=5, tol=2, do=0.2, wreg=0, pretrain='PQ', bnorm=True, nest=True, rings=True)
    train_err5 = sparta_eval(mean5, std5, traindf_bin_coildiff_normdata, mod5, atom, rings=True)
    test_err5 = sparta_eval(mean5, std5, testdf_bin_coildiff_normdata, mod5, atom, rings=True)
    res5 = [atom, 'Deep_Prelu', 'Binary', 'Normalized', 'Feat', 'Diff', train_err5, test_err5]
#    mean, std, val_list, hist_list, mod = deep_model_prelu(bin_clean_bmrb_traindf_data, atom, [100, 60, 30], 0.01, 0.70, 5*10**-6, 100, 2000, 25, 10**-2, do=0.2, wreg=0, pretrain=True, bnorm=True, nest=True)
#    train_err2 = sparta_eval(mean, std, bin_clean_bmrb_traindf_data, mod, atom)
#    test_err2 = sparta_eval(mean, std, bin_clean_bmrb_testdf_data, mod, atom)
#    min_val_idx = min((val, idx) for (idx, val) in enumerate(val_list))[1]
#    val_epochs2 = max(min_val_idx * 25, 100)
#    res2 = [atom, 'Deep_Prelu', 'Binary', 'Raw', train_err2, test_err2, val_epochs2]
#    mean, std, val_list, hist_list, mod = deep_model_prelu(clean_bmrb_traindf_white, atom, [100, 60, 30], 0.01, 0.70, 5*10**-6, 100, 2000, 25, 10**-2, do=0.2, wreg=0, pretrain=True, bnorm=True, nest=True)
#    min_val_idx = min((val, idx) for (idx, val) in enumerate(val_list))[1]
#    val_epochs3 = max(min_val_idx * 25, 100)
#    train_err3 = sparta_eval(mean, std, clean_bmrb_traindf_white, mod, atom)
#    test_err3 = sparta_eval(mean, std, clean_bmrb_testdf_white, mod, atom)
#    res3 = [atom, 'Deep_Prelu', 'Blosum62', 'Normalized', train_err3, test_err3, val_epochs3]
    print('Finished atom ' + atom)
    prelu_results.append(res0)
    prelu_results.append(res1)
    prelu_results.append(res2)
    prelu_results.append(res3)
    prelu_results.append(res4)
    prelu_results.append(res5)
    prelu_resdf = pd.DataFrame(prelu_results, columns=['Atom', 'Model', 'Seq_encoding', 'Feat_data', 'Rings', 'RCoils', 'Train_err', 'Test_err'])
    prelu_resdf.to_pickle('/home/bennett/Documents/DataSets/bmrb_clean/clean_rings/DeepPrelu_ringcoil_Results.pkl')
#    prelu_resdf.to_pickle('/Users/kcbennett/Documents/Git_Collaborations_THG/chemshift_prediction/DeepPrelu_ringcoil_Results.pkl')
    
prelu_resdf = pd.read_pickle('/Users/kcbennett/Documents/data/ShiftX2/bmrb_clean/DeepPrelu_Results.pkl')
prelu_resdf2 = pd.read_pickle('/Users/kcbennett/Documents/data/ShiftX2/bmrb_clean/DeepPrelu_Results2.pkl')
for i in range(len(atom_names)):
    res = [atom_names[i], 'SPARTA+', 'Blosum62', 'Raw', np.nan, sparta_results[i]]
    prelu_results.append(res)
resdf = pd.DataFrame(prelu_results, columns=['Atom', 'Model', 'Seq_encoding', 'Feat_data', 'Train_err', 'Test_err'])
resdf1 = pd.concat([prelu_resdf0, resdf], ignore_index=True)    
prelu_resdf2.loc[0, ['Atom', 'Model', 'Seq_encoding', 'Feat_data']].values
prelu_resdf[(prelu_resdf['Seq_encoding'] == prelu_resdf2.loc[0, 'Seq_encoding']) & (prelu_resdf['Feat_data'] == prelu_resdf2.loc[0, 'Feat_data'])] = prelu_resdf2.values
prelu_resdf3 = prelu_resdf.copy()
resdf.to_pickle('/Users/kcbennett/Documents/data/ShiftX2/bmrb_clean/DeepPrelu_results2.pkl')
resdf1.sort_values(by=['Atom', 'Model', 'Seq_encoding', 'Feat_data']).to_csv('/Users/kcbennett/Documents/data/ShiftX2/bmrb_clean/DeepPrelu_results_csv.csv')
resdf.to_csv('/Users/kcbennett/Documents/data/ShiftX2/bmrb_clean/DeepPrelu_results_csv.csv')

resdf.insert(loc=4, column='RC', value='False')
resdf.insert(loc=5, column='Target', value='Raw')
# Compare to SPARTA+ architecture
for atom in ['HA', 'H', 'CA', 'CB', 'C']:
    mean0, std0, val_list0, hist_list0, param_list0, mod0 = sparta_model(bin_rcoil_diff_traindf_normdata, atom, 400, per=5, tol=2, pretrain='PQ', rings=True)
    train_err0 = sparta_eval(mean0, std0, bin_rcoil_diff_traindf_normdata, mod0, atom, rings=True)
    test_err0 = sparta_eval(mean0, std0, bin_rcoil_diff_testdf_normdata, mod0, atom, rings=True)
    res0 = [atom, 'Deep_Prelu', 'Blosum62', 'Normalized', 'Feat', 'Feat', train_err0, test_err0]
    new_sparta_results.append(res0)



# Below are some auxilliary considerations relevant for RNNs

# Some quick stuff to check the distribution of chain lengths    
test_chain_lengths = []
train_chain_lengths = []
for i in range(61):
    chain_len = len(list(clean_bmrb_testdf.groupby(['FILE_ID', 'CHAIN']).groups.values())[i])
    test_chain_lengths.append(chain_len)
    
test_chain_lengths
for i in range(233):
    chain_len = len(list(clean_bmrb_traindf.groupby(['FILE_ID', 'CHAIN']).groups.values())[i])
    train_chain_lengths.append(chain_len)

# Need to count the number of skipped chains for each atom    
test_count_dict = {}
for atom in atom_names:
    count = 0
    for i, chain_idx in enumerate(test_idxs):
        chain = reslevel_testdf.iloc[chain_idx]
        weights = chain[atom].notnull()
        weights *= 1
        if np.array_equal(weights, np.zeros_like(weights)):
            count += 1
    test_count_dict[atom] = len(test_idxs) - count

train_count_dict = {}
for atom in atom_names:
    count = 0
    for i, chain_idx in enumerate(train_idxs):
        chain = reslevel_traindf.iloc[chain_idx]
        weights = chain[atom].notnull()
        weights *= 1
        if np.array_equal(weights, np.zeros_like(weights)):
            count += 1
    train_count_dict[atom] = len(train_idxs) - count

# A loop to record errors on LSTMs
lstm_results = []
param_plots = []
val_plots = []
hist_plots = []
for atom in atom_names:
    at_mean, at_std, at_vals, at_hist, at_params, at_mod = bidir_lstm_model(reslevel_traindf, atom, do=0.2, lstm_do=0.2, prelu=True, epochs=300, opt='nadam', pretrain='PQ', tol=2, arch=[[50], 30])
    test_idxs = sep_by_chains(reslevel_testdf, atom)
    test_gen = chain_batch_generator(reslevel_testdf, atom=atom, idxs=test_idxs, window=10, norm_shifts=(at_mean, at_std))
    err = np.sqrt(at_mod.evaluate_generator(test_gen, steps=test_count_dict[atom])) * at_std
    lstm_results.append([atom, 'nadam', 'PQ=2', '[[50], 30]', True, 0.2, 0.2, len(at_hist), err])
    param_plots.append(at_params)
    val_plots.append(at_vals)
    hist_plots.append(at_hist)
    
# Some by-hand LSTM testing  
mod_test = keras.models.Sequential()
mod_test.add(keras.layers.Bidirectional(keras.layers.LSTM(30, return_sequences=True), batch_input_shape=(None, 10, 110)))
mod_test.add(keras.layers.TimeDistributed(keras.layers.Dense(1, activation='linear')))
mod_test.compile(loss='mean_squared_error', optimizer='rmsprop', sample_weight_mode='temporal')
mod_test.fit_generator(example_train_generator, steps_per_epoch=len(train_idxs), epochs=100)

test_idxs = sep_by_chains(reslevel_testdf, 'C')
test_gen = chain_batch_generator(reslevel_testdf, atom='C', idxs=test_idxs, window=20, norm_shifts=(at_mean, at_std))
np.sqrt(at_mod.evaluate_generator(test_gen, steps=test_count_dict['C'])) * at_std

full_train_idxs = sep_by_chains(reslevel_traindf, 'C')
full_train_gen = chain_batch_generator(reslevel_traindf, atom='C', idxs=full_train_idxs, window=20, norm_shifts=(at_mean, at_std))
np.sqrt(at_mod.evaluate_generator(full_train_gen, steps=test_count_dict['C'])) * at_std
# NOTE -- Try both a masking layer as well as using sample_weights for the fit_generator

at_mean, at_std, at_vals, at_hist, at_params, at_mod = bidir_lstm_model(restraindf_coildiff_norm, 'C', do=0.2, lstm_do=0.2, prelu=True, epochs=300, opt='nadam', pretrain='PQ', tol=2, arch=[[50], 30])
lstm_eval(at_mean2, at_std2, restestdf_coildiff_norm, at_mod2, 'C')



# Below is some code for plotting errors.  May be useful again

rates = [0.005, 0.01, 0.05, 0.1]
momenta = [0.7, 0.9, 0.95]
for act in activs:
    for dec in decays:
        train_errs = resdf1.loc[(resdf1['Decay'] == dec) & (resdf1['Activation'] == act)]['Training_Error']
        test_errs = resdf1.loc[(resdf1['Decay'] == dec) & (resdf1['Activation'] == act)]['Test_Error']
        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, train_errs, width,
                        color='SkyBlue', label='Training')
        rects2 = ax.bar(ind, test_errs, width,
                        color='IndianRed', label='Testing', alpha=0.5)
        
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Loss (MSE)')
        ax.set_xlabel('Learning Rate (Momentum = [0.7, 0.9, 0.95])')
        ax.set_title(act + ', Drop=20, Dec=' + str(dec))
        ax.set_xticks([2, 6, 10, 14])
        ax.set_xticklabels([str(i) for i in rates])
        ax.axhline(1.37)
        ax.axhline(.974)
        path = '/home/bennett/Documents/Git_Collaborations_THG/shiftpred/performance_pics/'
        file = act + 'Drop20' + 'Dec' + str(dec)
        plt.savefig(path + file)







resdf1 = pd.read_pickle('/home/bennett/Documents/Git_Collaborations_THG/shiftpred/hyper_search.pkl')
train_errs = resdf1.loc[(resdf1['Decay'] == 10**-5) & (resdf1['Activation'] == 'relu')]['Training_Error']
test_errs = resdf1.loc[(resdf1['Decay'] == 10**-5) & (resdf1['Activation'] == 'relu')]['Test_Error']

fig, ax = plt.subplots()
ind = np.array([int(i/3)+i+1 for i in range(len(train_errs))])  # the x locations for the groups
width = 0.45  # the width of the bars

rects1 = ax.bar(ind, train_errs, width,
                color='SkyBlue', label='Training')
rects2 = ax.bar(ind, test_errs, width,
                color='IndianRed', label='Test', alpha=0.5)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Loss (MSE)')
ax.set_xlabel('Learning Rate (Momentum = [0.7, 0.9, 0.95])')
ax.set_title('Relu, Drop=20%, Dec=10^-6')
ax.set_xticks([2, 6, 10, 14])
ax.set_xticklabels([str(i) for i in rates])
ax.axhline(1.37)
ax.axhline(.974)
ax.
#path = '/home/bennett/Documents/Git_Collaborations_THG/shiftpred/performance_pics/'
#file = act + ', Drop=' + str(drop) + ', Dec=' + str(dec)
#plt.savefig(path + file)

plt.show()




















