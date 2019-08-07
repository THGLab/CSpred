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
#mpl.use("Agg")
import matplotlib.pyplot as plt
from Bio.SeqUtils import IUPACData
from keras.utils import plot_model
from keras import backend as K
from numpy import newaxis
USE_MULTIPROCESSING=True
VERBOSITY=2
#from chemshift_prediction.lsuv_init import LSUVinit

# The functions in this file are intended for use with specific data.
# Wherever possible, model-building functions have been written to 
# allow their use on other data as well and so column names are not 
# an issue.  However, certain functions, particularly the noise injector,
# do rely on column names for certain options (e.g., using angular
# uncertianty to inject cosine/sine noise)

# The following are the columns needed to define the functions below.
# The complete list of columns for the data that these functions are
# intended to process can be found in eda_and_prediction.py.
# These are needed for the model-building functions directly
atom_names = ['HA', 'H', 'CA', 'CB', 'C', 'N']
rcoil_cols = ['RCOIL_' + atom for atom in atom_names]
ring_cols = [atom + '_RC' for atom in atom_names]
#rcoil_cols = ['RC_' + atom for atom in atom_names]
#ring_cols = [atom + '_RING' for atom in atom_names]
xtra_ring_cols = ['HA2_RING', 'HA3_RING']
cols_to_drop = ['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM',"RES", 'CHAIN', 'RESNAME_ip1', 'RESNAME_im1', 'BMRB_RES_NUM', 'CG', 'RCI_S2', 'MATCHED_BMRB']

# These are needed for the structure-sequence branch model
col_phipsi = [i+j for i in ['PHI_', 'PSI_'] for j in ['COS_i-1', 'SIN_i-1']]
col_phipsi += [i+j for i in ['PHI_', 'PSI_'] for j in ['COS_i', 'SIN_i']]
col_phipsi += [i+j for i in ['PHI_', 'PSI_'] for j in ['COS_i+1', 'SIN_i+1']]
col_chi = [i+j+k for k in ['_i-1', '_i', '_i+1'] for i in ['CHI1_', 'CHI2_'] for j in ['COS', 'SIN', 'EXISTS']]
col_hbprev = ['O_'+i+'_i-1' for i in ['d_HA', '_COS_H', '_COS_A', '_EXISTS', '_ENERGY']]
col_hbond = [i+j+'_i' for i in ['Ha_', 'HN_', 'O_'] for j in ['d_HA', '_COS_H', '_COS_A', '_EXISTS', '_ENERGY']]
col_hbnext = ['HN_'+i+'_i+1' for i in ['d_HA', '_COS_H', '_COS_A', '_EXISTS', '_ENERGY']]
col_s2 = ['S2'+i for i in ['_i-1', '_i', '_i+1']]
struc_cols = col_phipsi + col_chi + col_hbprev + col_hbond + col_hbnext + col_s2
blosum_names = ['BLOSUM62_NUM_'+sorted(IUPACData.protein_letters_3to1.keys())[i].upper() for i in range(20)]
col_blosum = [blosum_names[i]+j for j in ['_i-1', '_i', '_i+1'] for i in range(20)]
seq_cols = col_blosum
bin_seq_cols = ['BINSEQREP_'+ list(IUPACData.protein_letters_3to1.keys())[i].upper() + j for j in ['_i-1', '_i', '_i+1'] for i in range(20)]

# These are the names of new columns that were not included in the SPARTA+ feature set
hsea_names = ['HSE_CA' + i  for i in ['_U', '_D', '_Angle']]
hseb_names = ['HSE_CB' + i  for i in ['_U', '_D']]
hse_cols = [name + i for i in ['_i-1', '_i', '_i+1'] for name in hsea_names + hseb_names]
dssp_ss_names = ['A_HELIX_SS', 'B_BRIDGE_SS', 'STRAND_SS', '3_10_HELIX_SS', 'PI_HELIX_SS', 'TURN_SS', 'BEND_SS', 'NONE_SS']
dssp_asa_names = ['REL_ASA', 'ABS_ASA']
dssp_pp_names = ['DSSP_PHI', 'DSSP_PSI']
dssp_hb_names = ['NH-O1_ENERGY', 'NH-O2_ENERGY', 'O-NH1_ENERGY', 'O-NH2_ENERGY']
dssp_cols = [name + i for i in ['_i-1', '_i', '_i+1'] for name in dssp_ss_names + dssp_asa_names + dssp_pp_names + dssp_hb_names]
dssp_expp_cols = [name + i for i in ['_i-1', '_i', '_i+1'] for name in dssp_ss_names + dssp_asa_names + dssp_hb_names]
dssp_ssi_cols = [name + '_i' for name in dssp_ss_names]
dssp_ss_cols = [name + i for i in ['_i-1', '_i', '_i+1'] for name in dssp_ss_names]
dssp_norm_cols = [name + i for i in ['_i-1', '_i', '_i+1'] for name in dssp_asa_names + dssp_hb_names]

# Make separate lists of cosine, sine, distance columns
phipsicos_cols = [i + 'COS_i-1' for i in ['PHI_', 'PSI_']]
phipsicos_cols += [i + 'COS_i' for i in ['PHI_', 'PSI_']]
phipsicos_cols += [i + 'COS_i+1' for i in ['PHI_', 'PSI_']]
chicos_cols = [i + 'COS' + k for k in ['_i-1', '_i', '_i+1'] for i in ['CHI1_', 'CHI2_']]
hbondcos_cols = ['O_'+i+'_i-1' for i in ['_COS_H', '_COS_A']]
hbondcos_cols += [i+j+'_i' for i in ['Ha_', 'HN_', 'O_'] for j in ['_COS_H', '_COS_A']]
hbondcos_cols += ['HN_'+i+'_i+1' for i in ['_COS_H', '_COS_A']]
cos_cols = phipsicos_cols + chicos_cols + hbondcos_cols
phipsisin_cols = [i + 'SIN_i-1' for i in ['PHI_', 'PSI_']]
phipsisin_cols += [i + 'SIN_i' for i in ['PHI_', 'PSI_']]
phipsisin_cols += [i + 'SIN_i-1' for i in ['PHI_', 'PSI_']]
chisin_cols = [i + 'SIN' + k for k in ['_i-1', '_i', '_i+1'] for i in ['CHI1_', 'CHI2_']]
sin_cols = phipsisin_cols + chisin_cols
# And a list for columns containing distance information
hbondd_cols = ['O_d_HA_i-1']
hbondd_cols += [i+'d_HA_i' for i in ['Ha_', 'HN_', 'O_']]
hbondd_cols += ['HN_d_HA_i+1']

# Need a list of columns that will be subject to noise if we wish to use noise injection
angle_cols = cos_cols + sin_cols
noisy_cols = angle_cols + hbondd_cols + col_s2

# These are the names of the columns with sequence information beyond the tri-peptide level
ext_seq_cols = ['RESNAME_i' + i + str(j) for i in ['+', '-'] for j in range(1,21)]

# Define the names for the original Sparta+ Data Columns (same set as above but different order so we can assign column labels to data obtained directly from Yang)
orig_cols = col_blosum[:20]
orig_cols += [i + j for i in ['PHI_', 'PSI_'] for j in ['SIN_i-1', 'COS_i-1']]
orig_cols += [i + j + '_i-1' for i in ['CHI1_', 'CHI2_'] for j in ['SIN', 'COS', 'EXISTS']]
orig_cols += col_blosum[20:40]
orig_cols += [i + j for i in ['PHI_', 'PSI_'] for j in ['SIN_i', 'COS_i']]
orig_cols += [i + j + '_i' for i in ['CHI1_', 'CHI2_'] for j in ['SIN', 'COS', 'EXISTS']]
orig_cols += col_blosum[40:]
orig_cols += [i + j for i in ['PHI_', 'PSI_'] for j in ['SIN_i+1', 'COS_i+1']]
orig_cols += [i + j + '_i+1' for i in ['CHI1_', 'CHI2_'] for j in ['SIN', 'COS', 'EXISTS']]
orig_cols += ['O_'+i+'_i-1' for i in ['_EXISTS', 'd_HA', '_COS_A', '_COS_H']]
orig_cols += [i+j+'_i' for i in ['HN_', 'Ha_', 'O_'] for j in ['_EXISTS', 'd_HA', '_COS_A', '_COS_H']]
orig_cols += ['HN_'+i+'_i+1' for i in ['_EXISTS', 'd_HA', '_COS_A', '_COS_H']]
orig_cols += ['S2'+i for i in ['_i-1', '_i', '_i+1']]

# These are the names of the boolean columns in the Sparta+ features that indicate existance of chi1/chi2 angles as well as possible hydrogen bonds
exist_cols = [i + 'EXISTS' + k for k in ['_i-1', '_i', '_i+1'] for i in ['CHI1_', 'CHI2_']]
exist_cols += ['O_'+ '_EXISTS' +'_i-1', 'HN_' + '_EXISTS' + '_i+1']
exist_cols += [i + '_EXISTS' + '_i' for i in ['Ha_', 'HN_', 'O_']]
# For completeness, here are the remaining columns
non_exist_cols = orig_cols.copy()
for x in exist_cols:
    non_exist_cols.remove(x)

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

# These are the names of the columns that are not in Sparta+ but are in the un-augmented features from our extraction
cols_notinsp = dssp_cols + hse_cols + ext_seq_cols

# Form lists for numerical features and non-numerical features
non_numerical_cols=dssp_ss_cols + exist_cols

# We begin with some data processing functions

# Here is a function to augment the sparta+ feature set by squaring certain of the columns
def featsq(data, columns):
    '''Function to square the given columns from the data and store the results in new columns
    
    data = Contains the data to be squared (Pandas DataFrame)
    columns = The names of the columns to be squared (List of Str)
    '''
    dat = data.copy()
    for col in columns:
        sq_col_name = col + '_sq'
        dat[sq_col_name] = dat[col] ** 2
    return dat


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


def add_chainbreak_column(data, n_skips=True, both_ways=False):
    '''Function to add a column to provided data that indicates presence/absence of
    any discontinuities in the chain
    
    data = Data including context information like FILE_ID etc. (Pandas DataFrame)
    n_skps = If False then just indicate presence (1) or absence (0) of chain break.
            If True, then  count number of skipped residues and map to [0, 1] (Bool)
    both_ways = If False then add a column only for right discontinuities.  If True then
                also add a column for left discontinuities (Bool).
    '''
    df = data.copy()
    df['CHAIN_RDISCONT'] = np.nan
    if both_ways:
        df['CHAIN_LDISCONT'] = np.nan
    
    for i in range(len(data)):
        fid_i = data.loc[i]['FILE_ID']
        res_i = data.loc[i]['RESNAME']
        chain_i = data.loc[i]['CHAIN']
        resn_i = data.loc[i]['RES_NUM']
        if i == 0:
            fid_im1 = None
            res_im1 = None
            chain_im1 = None
            resn_im1 = None
        else:
            fid_im1 = data.loc[i-1]['FILE_ID']
            res_im1 = data.loc[i-1]['RESNAME']
            chain_im1 = data.loc[i-1]['CHAIN']
            resn_im1 = data.loc[i-1]['RES_NUM']
        if i == len(data) - 1:
            fid_ip1 = None
            res_ip1 = None
            chain_ip1 = None
            resn_ip1 = None
        else:
            fid_ip1 = data.loc[i+1]['FILE_ID']
            res_ip1 = data.loc[i+1]['RESNAME']
            chain_ip1 = data.loc[i+1]['CHAIN']
            resn_ip1 = data.loc[i+1]['RES_NUM']
            
        # Fill in Right Discontinuity column
        if (fid_i == fid_ip1) and (chain_i == chain_ip1):
            rescheck_ip1 = data.loc[i]['RESNAME_ip1']
            rescheck_im1 = data.loc[i+1]['RESNAME_im1']
            nskips = abs(resn_ip1 - resn_i - 1)
            nskips = min(20, nskips)
            if (rescheck_ip1 == res_ip1) and (rescheck_im1 == res_i) and (nskips == 0):
                df.loc[i, 'CHAIN_RDISCONT'] = 0.0
            else:
                if n_skips:
                    df.loc[i, 'CHAIN_RDISCONT'] = 1.0 - 0.5 ** nskips
                else:
                    df.loc[i, 'CHAIN_RDISCONT'] = 1.0
        else:
            df.loc[i, 'CHAIN_RDISCONT'] = 1.0
            
            
        if both_ways:
            if (fid_i == fid_im1) and (chain_i == chain_im1):
                rescheck_ip1 = data.loc[i-1]['RESNAME_ip1']
                rescheck_im1 = data.loc[i]['RESNAME_im1']
                nskips = abs(resn_i - resn_im1 - 1)
                nskips = min(20, nskips)
                if (rescheck_ip1 == res_i) and (rescheck_im1 == res_im1) and (nskips == 0):
                    df.loc[i, 'CHAIN_LDISCONT'] = 0.0
                else:
                    if n_skips:
                        df.loc[i, 'CHAIN_LDISCONT'] = 1.0 - 0.5 ** nskips
                    else:
                        df.loc[i, 'CHAIN_LDISCONT'] = 1.0
            else:
                df.loc[i, 'CHAIN_LDISCONT'] = 1.0
                        
    return df        
        
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

def filter_data(data,filter_columns):
    '''Function that returns subset of data with no null elements in filter columns.
    data = Feature and target data (Pandas DataFrame)
    filter_columns = List of names of columns needed for filtering (List)
    '''
    filtered=data.copy()
    for column in filter_columns:
        filtered=filtered[filtered[column].notnull()]
    return filtered

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
    complete_group=[]
    for group in groups:
        length=len(group)
        chain=data.loc[group]
        res_nums=list(chain['RES_NUM'])
        chain_idx=[[group[0]]]
        for n in range(1,length):
            if res_nums[n]-1==res_nums[n-1]:
                chain_idx[-1].append(group[n])
            else:
                chain_idx.append([group[n]])
        complete_group.extend(chain_idx)
    groups=complete_group
    if split is None:
        return groups # If we don't need to split into training/validation, just return list of chains
    else:
        #dat = data[data[atom].notnull()]#?
        dat=data.copy()
        num_shifts = len(dat) # Find the total number of shifts for this atom type
        shuff_groups = random.sample(groups, len(groups)) # Shuffle groups to randomize order
        i = 0 # Initialize index
        p = 0 # Initialize percent of data set aside
        val_shifts = 0 # Initialize number of validation shifts to use
        val_list = [] # Initialize list of lists of indices
        train_list = shuff_groups.copy()
        while p < split:
            val_list.append(shuff_groups[i]) # Append the i'th chain to list of val indices
            new_shifts = len(shuff_groups[i])
            val_shifts += new_shifts
            p = val_shifts / num_shifts
            i += 1
        del train_list[:i]
        return train_list, val_list
            

def decide_duplicate_weights(window,total_length,len_to_start):
    num_duplicate=np.zeros((len(total_length),window))
    for i in range(len(total_length)):
        for j in range(window):
            num_duplicate[i][j]+=min(window-j-1,len_to_start[i])+ \
                min(j,total_length[i]-len_to_start[i]-window)+1
    normalization=np.tile(np.max(num_duplicate,axis=1),(window,1)).T
    return normalization/num_duplicate

def chain_batch_generator(data, idxs, atom, window, norm_shifts=(0, 1), sample_weights=True, randomize=True, rolling=True, center_only=False, batch_size=32):
    '''
    Takes full DataFrame of all train and validation chains
    along with a list of index lists for the various chains to be batched
    and the window (length of sub-samples).
    
    data = Feature and target data   (Pandas DataFrame)
    idxs = List of index objects each specifying all the residues in a given chain (List)
    atom = List of atoms --needed to normalize shifts (Str)
    window = Length of subsequences into which chains are to be chopped for batch training (Int)
    norm_shifts = Tuple with the first element being the mean for different shifts, and the second element
    being the standard deviation of the shifts (List of arrays)
    sample_weights = Return sample weights of same shape as target array with entries of 1 for all timesteps (sequence entries) except 0's for timesteps with no shift data (NumpyArray)
    randomize = Return batches with order of subsequences randomized - to test if state information between subsequences is being (usefully) implemented (Bool)
    rolling = Use rolling windows as opposed to non-overlapping windows for subsequences (Bool)
    center_only = predict only the central residue for rolling windows (Bool)
    batch_size = size of pieces generated in each batch (Int)
    '''
    num_shifts=len(atom)
    dat = data.copy()
    shifts = dat[atom]
    feats = dat.drop(atom_names, axis=1).fillna(0)
    
    if norm_shifts:
        #all_shifts = dat[atom].fillna(0)
        #shifts_mean = all_shifts.mean()
        #shifts_std = all_shifts.std()
        shifts_mean = norm_shifts[0]
        shifts_std = norm_shifts[1]
        shift_norm = (shifts - shifts_mean) / shifts_std
    else:
        shift_norm = shifts
    # prepare shifts and features data

    # ATTN: Normalizing the shifts like this makes the formerly NaN values
    # now a non-zero number in general.  Should be ok if we use sample_weights
    # shift_norm = shift_norm * weights # reset formerly NaN shifts to zero
    # shift_norm = np.nan_to_num(shift_norm)
    num_feats = feats.shape[1]
    all_pieces=[]
    for chain in idxs:
        chain_length=len(chain)
        if chain_length<window:
            pass
        else:
            for i in range(chain_length-window+1):
                # record the distance from the head of the piece to the start of the chain
                dist_to_start=i
                # also include chain length so that sample weights can be decided by the position of piece in chain
                all_pieces.append((chain[i:i+window],(dist_to_start,chain_length)))
    if randomize:
        random.shuffle(all_pieces)
    num_all_pieces=len(all_pieces)
    while True:
        for i in range(int(num_all_pieces/batch_size)):
            # parts that can fill a batch
            batch_pieces=[all_pieces[n] for n in range(i*batch_size,(i+1)*batch_size)]
            batch_piece_indices=[piece[0] for piece in batch_pieces]
            batch_feats=np.zeros((batch_size,window,num_feats))
            # When center only is toggled, only output the shifts for the central residue
            if center_only:
                batch_shifts=np.zeros((batch_size,num_shifts))
            else:
                batch_shifts=np.zeros((batch_size,window,num_shifts))
            for j in range(batch_size):
                batch_feats[j,:,:]=feats.loc[batch_piece_indices[j]].values
                if center_only:
                    batch_shifts[j,:]=shift_norm.loc[batch_piece_indices[j][int((window-1)/2)]].values
                else:
                    batch_shifts[j,:,:]=shift_norm.loc[batch_piece_indices[j]].values
            batch_shifts=np.split(batch_shifts,num_shifts,axis=-1)
            if not sample_weights:
                yield batch_feats, batch_shifts
            else:
                information_array=np.array([piece_information[1] for piece_information in batch_pieces])
                batch_weights=decide_duplicate_weights(window,information_array[:,1],information_array[:,0])
                returned_sample_weights=[]
                if not center_only:
                    for atom_shifts in batch_shifts:
                        is_valid=np.logical_not(np.isnan(atom_shifts).any(axis=2)) # Only valid for NOT center_only!!
                        returned_sample_weights.append(is_valid*batch_weights)
                else:
                    for atom_shifts in batch_shifts:
                        is_valid=np.logical_not(np.isnan(atom_shifts))
                        returned_sample_weights.append(is_valid.flatten())
                yield batch_feats, [np.nan_to_num(batch_shift,0) for batch_shift in batch_shifts], returned_sample_weights
    # Does it make sense to generate zeros for these remaining pieces?

        remaining_pieces=[all_pieces[n] for n in range(int(num_all_pieces/batch_size)*batch_size,num_all_pieces)]
        batch_feats=np.zeros((batch_size,window,num_feats))
        if center_only:
            batch_shifts=np.zeros((batch_size,num_shifts))
        else:
            batch_shifts=np.zeros((batch_size,window,num_shifts))
        for i in range(len(remaining_pieces)):
            piece_indices=[piece[0] for piece in remaining_pieces]
            batch_feats[i,:,:]=feats.loc[piece_indices[i]].values
            if center_only:
                batch_shifts[i,:]=shift_norm.loc[piece_indices[i][int((window-1)/2)]].values
            else:
                batch_shifts[i,:,:]=shift_norm.loc[piece_indices[i]].values
        batch_shifts=np.split(batch_shifts,num_shifts,axis=-1)
        if not sample_weights:
            yield batch_feats, batch_shifts
        else:
            information_array=np.array([piece_information[1] for piece_information in remaining_pieces])
            batch_weights=decide_duplicate_weights(window,information_array[:,1],information_array[:,0])
            batch_weights=np.vstack((batch_weights,np.zeros((batch_size-len(remaining_pieces),window))))
            returned_sample_weights=[]
            for atom_shifts in batch_shifts:
                if center_only:
                    is_valid=np.logical_not(np.isnan(atom_shifts))
                    returned_sample_weights.append(is_valid.flatten())
                else:
                    is_valid=np.logical_not(np.isnan(atom_shifts).any(axis=2))
                    returned_sample_weights.append(is_valid*batch_weights)
            yield batch_feats, [np.nan_to_num(batch_shift,0) for batch_shift in batch_shifts], returned_sample_weights
        if randomize:
            random.shuffle(all_pieces)


# Let's also write a generator for residue-level data that can inject noise
def res_level_generator(data, atom, noise={}, noise_type='percent', noise_dist='uniform', batch_size=64, norm_shifts=(0, 1), seed=None):
    '''Constructs a generator that yields batches of example residues and shifts with
    noise optionally injected.
    
    data - Feature and target data (Pandas DataFrame)
    atom - Name of atom to predict (Str)  -- SHOULD ALSO SAY IT IS THE NAME OF THE TARGET COLUMN!!
    noise - Magnitude of noise to add to each column in data or 
            or for different types (angles, lengths, shifts, etc.) (Dict)
            Can also apply same noise level to each column (Float)
    noise_type - Use "percent" or "absolute" levels of noise.  Also accepts
                "angle" wherein noise is injected for the angles rather than their
                trig functions as in the existing columns. If "angle" then noise dict
                is expected to have entries for "angle", "length", and "shift" (Str)
    noise_dist - Distribution from which to generate noise; can be "uniform" or "normal" (Str)
    batch_size - Number of examples to yield per batch (Int)
    norm_shifts - Normalize target shifts with provided mean and std (Tuple)
    seed - Seed to use for randomizing order between epochs
    '''
    dat = data[data[atom].notnull()]
    noise_names = noisy_cols.copy()
    for col in cols_to_drop:
        try:
            dat = dat.drop(col, axis=1)
        except ValueError:
            pass
    feats = dat.drop(atom_names, axis=1)
    shifts = dat[atom]
    
    if len(dat) < batch_size:
        raise Exception('Fewer examples were found in data than specified batch_size')

    if norm_shifts:
        shifts_mean = norm_shifts[0]
        shifts_std = norm_shifts[1]
        shifts = (shifts - shifts_mean) / shifts_std
    
    try:
        ring_col = atom + '_RC'
        rem1 = ring_cols.copy()
        rem1.remove(ring_col)
        feats = feats.drop(rem1, axis=1)
        feats[ring_col] = feats[ring_col].fillna(value=0)
        noise_names += [ring_col]
        rings = True
    except KeyError:
        rings = False
    except ValueError:
        rings = False
    try:
        rcoil_col = 'RCOIL_' + atom
        rem2 = rcoil_cols.copy()
        rem2.remove(rcoil_col)
        feats = feats.drop(rem2, axis=1)
        noise_names += [rcoil_col]
    except ValueError:
        pass
    
    n = len(feats) // batch_size # Number of full batches per epoch
    rem = len(feats) % batch_size # Size of extra batch
    
    while True:
        these_feats = feats.copy()
        these_shifts = shifts.copy()
    
        if (type(noise) is float) or (type(noise) is int):
            for name in noise_names:
                if noise_dist is 'uniform':
                    mult_noise = noise * (2 * np.random.rand(len(dat)) -1 )
                if noise_dist is 'normal':
                    mult_noise = noise * np.random.randn(len(dat))
                these_feats[name] = these_feats[name] * (mult_noise + 1)
            if noise_dist is 'uniform':
                shift_mult_noise = noise * (2 * np.random.rand(len(dat)) -1 )
            if noise_dist is 'normal':
                shift_mult_noise = noise * np.random.randn(len(dat))
            these_shifts = these_shifts * (1 + shift_mult_noise)

        
        if noise_type is 'percent':
            for name in noise:
                if noise_dist is 'uniform':
                    mult_noise = noise[name] * (2 * np.random.rand(len(dat)) -1 )
                if noise_dist is 'normal':
                    mult_noise = noise * np.random.randn(len(dat))
                try:
                    these_feats[name] = these_feats[name] * (mult_noise + 1)
                except KeyError:
                    pass
            if noise_dist is 'uniform':
                shift_mult_noise = noise[atom] * (2 * np.random.rand(len(dat)) -1 )
            if noise_dist is 'normal':
                shift_mult_noise = noise[atom] * np.random.randn(len(dat))
            these_shifts = these_shifts * (1 + shift_mult_noise)

            
        if noise_type is 'angle':
            for cos_col in cos_cols:
                sigma = np.abs(np.sin(np.arccos(dat[cos_col]))) * noise['angle']
                if noise_dist is 'uniform':
                    add_noise = sigma * (2 * np.random.rand(len(dat)) -1)
                if noise_dist is 'normal':
                    add_noise = sigma * np.random.randn(len(dat))
                these_feats[cos_col] = these_feats[cos_col] + add_noise
            for sin_col in sin_cols:
                sigma = np.abs(np.cos(np.arcsin(dat[sin_col]))) * noise['angle']
                if noise_dist is 'uniform':
                    add_noise = sigma * (2 * np.random.rand(len(dat)) -1)
                if noise_dist is 'normal':
                    add_noise = sigma * np.random.randn(len(dat))
                these_feats[sin_col] = these_feats[sin_col] + add_noise
            for length_col in hbondd_cols:
                sigma = noise['length']
                if noise_dist is 'uniform':
                    add_noise = sigma * (2 * np.random.rand(len(dat)) -1)
                if noise_dist is 'normal':
                    add_noise = sigma * np.random.randn(len(dat))
                these_feats[length_col] = these_feats[length_col] + add_noise
            if noise_dist is 'uniform':
                shift_add_noise = noise[atom] * (2 * np.random.rand(len(dat)) -1)
            if noise_dist is 'normal':
                shift_add_noise = noise[atom] * np.random.randn(len(dat))
            these_shifts = these_shifts + shift_add_noise
            if rings: #Not sure how to choose absolute error for rings so use percent
                if noise_dist is 'uniform':
                    mult_noise = sigma * (2 * np.random.rand(len(dat)) - 1)
                if noise_dist is 'normal':
                    mult_noise = sigma * np.random.rand(len(dat))
                these_feats[ring_col] = these_feats[ring_col] * (1 + mult_noise)
        
        # Now randomly permute the order of the features and targets and yield in batches
        if seed:
            np.random.seed(seed)
            shuff_idx = np.random.permutation(np.arange(len(these_feats)))
        else:
            shuff_idx = np.random.permutation(np.arange(len(these_feats)))
        for batch_num in range(n):
            sel = shuff_idx[batch_num * batch_size : batch_size * (batch_num + 1)]
            yield these_feats.iloc[sel], these_shifts.iloc[sel]
        if rem is not 0:
            sel = shuff_idx[n * batch_size :]
            yield these_feats.iloc[sel], these_shifts.iloc[sel]


def data_prep(data, atom, reject_outliers=None, norm_shifts=True, norm_stats=None, split_numeric=False, split_class=False):
    '''Function to prepare features and shifts from the given DataFrame to facilitate
    processing by the model-generating functions below.
    
    args:
        data - DataFrame containing feature and target data (Pandas DataFrame)
        atom - Name of atom for which the shifts are desired (Str)
        reject_outliers - Eliminate entries for which the targets differ by more than this number of standard deviations (None or Float)
        norm_shifts - Whether or not to normalize the target shifts (Bool)
        norm_stats - Statistics [mean, std] to use for normalizing shifts.  If norm_shifts==True and norm_stats==None, use stats from data (None or List) 
        split_numeric - If true, return two feature dataframes, one only containing numerical features and the other only containing non-numerical features
        split_class - For numerical splitting model, split out the classes so that it can be put into the middle level of the network (only effective when split_numeric=True)
        
    returns:
        feats, shifts - Feature and shift data (Pandas DataFrames)
    '''
    dat = data[data[atom].notnull()]
    for col in cols_to_drop:
        try:
            dat = dat.drop(col, axis=1)
        except KeyError:
            pass
        except ValueError:
            pass

    if reject_outliers is not None:
        n_sig = max(reject_outliers, 0.1) # Don't accept a cutoff less than 0.1 * std
        rej_mean = dat[atom].mean()
        rej_std = dat[atom].std()
        up_rej = rej_mean + n_sig * rej_std
        low_rej = rej_mean - n_sig * rej_std
        dat = dat[(dat[atom] > low_rej) & (dat[atom] < up_rej)]

    # If other atoms are in the data then drop them
    feats = dat.drop(atom_names, axis=1)
    shifts = dat[atom]
    
    # Normalize shifts if this is desired
    if norm_shifts:
        if norm_stats is None:
            shifts_mean = shifts.mean()
            shifts_std = shifts.std()
            shift_norm = (shifts - shifts_mean) / shifts_std
            shift_norm = shift_norm.values
        else:
            shifts_mean = norm_stats[0]
            shifts_std = norm_stats[1]
            shift_norm = (shifts - shifts_mean) / shifts_std
            shift_norm = shift_norm.values
    else:
        shifts_mean = 0
        shifts_std = 1
        shift_norm = shifts.values

    
    # Need to drop the random coil and ring current columns for the other atoms if 
    # such columns are in the data.
    try:
        ring_col = atom + '_RC'
        rem1 = ring_cols.copy()
        rem1.remove(ring_col)
        feats = feats.drop(rem1, axis=1)
        feats[ring_col] = feats[ring_col].fillna(value=0)
    except KeyError:
        pass
    except ValueError:
        pass
    try:
        rcoil_col = 'RCOIL_' + atom
        rem2 = rcoil_cols.copy()
        rem2.remove(rcoil_col)
        feats = feats.drop(rem2, axis=1)
    except KeyError:
        pass
    except ValueError:
        pass
    
    if split_numeric:
        feats_non_numeric = feats[non_numerical_cols].values
        feats_numeric=feats[[col for col in feats.columns if col not in non_numerical_cols]]
        if split_class:
            class_cols=[col for col in feats_numeric.columns if "CLASS_" in col]
            feats_class=feats[class_cols].values
            feats_numeric=feats_numeric[[col for col in feats_numeric.columns if col not in class_cols]].values
            feats=[feats_numeric,feats_non_numeric,feats_class]
        else:
            feats=[feats_numeric.values,feats_non_numeric]
    else:
        feats = feats.values
    
    if reject_outliers is not None:
        return feats, shift_norm, rej_mean, rej_std
    else:
        return feats, shift_norm, shifts_mean, shifts_std

def rnn_data_prep(data, atoms, window, batch_size=64,  norm_shifts=True, norm_stats=None,val_split=None):
    '''Function to prepare features and shifts for the RNN model (timeseries of features)
    
    args:
        data - DataFrame containing feature and target data (Pandas DataFrame)
        atoms - List of all the atoms needed to be predicted (List)
        window - Size of window for generating data (Odd int)
        batch_size - Number of examples in each batch (int)
        norm_shifts - Whether or not to normalize the target shifts (Bool)
        norm_stats - Statistics [mean, std] to use for normalizing shifts.  If norm_shifts==True and norm_stats==None, use stats from data (None or List)
        val_split - None, or a fraction of validation data 
        
    returns:
        train_generators, shifts - Feature and shift data (generators)
    '''
    dat=data.copy()
    # Need to drop the random coil and ring current columns for the other atoms if 
    # such columns are in the data.
    try:
        ring_col = [single_atom + '_RC' for single_atom in atoms]
        rem1 = ring_cols.copy()
        for column in ring_col:
            rem1.remove(column)
        dat.drop(rem1, axis=1, inplace=True)
        dat[ring_col] = dat[ring_col].fillna(value=0)
    except KeyError:
        pass
    except ValueError:
        pass
    try:
        rcoil_col = ['RCOIL_' + single_atom for single_atom in atoms]
        rem2 = rcoil_cols.copy()
        for column in rcoil_col:
            rem2.remove(column)
        dat.drop(rem2, axis=1, inplace=True)
    except KeyError:
        pass
    except ValueError:
        pass
    # Get shift statistics
    if norm_shifts:
        all_shifts = filter_data(dat,atoms)
        all_shifts = all_shifts[atoms]
        shifts_mean = all_shifts.mean(axis=0)
        shifts_std = all_shifts.std(axis=0)
    else:
        if norm_stats is not None:
            shifts_mean=norm_stats[0]
            shifts_std=norm_stats[1]
        else:
            shifts_mean = 0
            shifts_std = 1
    # Split the data by chain according to whether we need to do train/test split
    if val_split is None:
        chains_set=sep_by_chains(dat, atom=atoms)
    else:
        train_set, val_set = sep_by_chains(dat, atom=atoms, split=val_split)
        full_set = train_set + val_set
    # Metadata has already been used (in sep_by_chains), so now can safely remove them from featues
    for dcol in cols_to_drop:
        try:
            dat.drop(dcol, axis=1,inplace=True)
        except KeyError:
            pass
        except ValueError:
            pass
    # Get total number of features
    feats = dat.drop(atom_names, axis=1)
    num_feats = feats.shape[1]
    # Create generators 
    if val_split is None:
        data_gen = chain_batch_generator(dat,chains_set,atoms,window, norm_shifts=(shifts_mean, shifts_std),batch_size=batch_size,center_only=True)
        steps=math.ceil(sum([len(chain)-window+1 for chain in chains_set if len(chain)>=window])/batch_size)
    else:
        train_gen = chain_batch_generator(dat, train_set, atoms, window,
        norm_shifts=(shifts_mean, shifts_std), batch_size=batch_size,center_only=True)
        val_gen = chain_batch_generator(dat, val_set, atoms, window, norm_shifts=(shifts_mean, shifts_std), 
        batch_size=batch_size,center_only=True)
        full_gen = chain_batch_generator(dat, full_set, atoms, window, norm_shifts=(shifts_mean, shifts_std),
        batch_size=batch_size,center_only=True)
        train_steps = math.ceil(sum([len(chain)-window+1 for chain in train_set if len(chain)>=window])/batch_size)
        val_steps = math.ceil(sum([len(chain)-window+1 for chain in val_set if len(chain)>=window])/batch_size)
        full_steps = train_steps+val_steps
        data_gen=(train_gen,val_gen,full_gen)
        steps=(train_steps,val_steps,full_steps)
    return data_gen,steps,num_feats,shifts_mean,shifts_std



def mtl_data_prep(data, atoms, reject_outliers=None, norm_shifts=True, mode='un-masked'):
    '''Function to prepare features and shifts from the given DataFrame to facilitate
    processing by the model-generating functions below that use multi-task learning.
    There is significant conceptual overlap with the previous data_prep function but
    we define this second function for simplicity since the use is sufficiently 
    different.
    
    args:
        data - DataFrame containing feature and target data (Pandas DataFrame)
        atom - Name of atom for which the shifts are desired (Str)
        reject_outliers - Eliminate entries for which the targets differ by more than this number of standard deviations (None or Float)
        norm_shifts - Whether or not to normalize the target shifts (Bool)
        mode - Accepts 'testing', 'masked', or 'un-masked'.  If for testing, then returned shifts should be for all residues with non-null shifts for each atom rather than only those residues with non-null for all atoms, which is the 'un-masked' behavior.  If 'masked', then all residues with at least 1 non-NaN shifts will have the NaN shifts set to a pre-specified value (hard coded as 10**3) so that they can be easily identified for a custom loss function (see masked_mse) which will then drop the loss contribution coming from shifts with this pre-specified value.  This functionality allows using all residues with at least 1 non-NaN shift for training of MTL models (Str)

        
    returns:
        feats, shifts_norms, shifts_means, shifts_stds - Feature, shifts, shift means, and shift stds for all atoms requested data (Pandas DataFrames)
    '''
    dat = data.copy()
    # Drop unnecessary columns
    for col in cols_to_drop:
        try:
            dat = dat.drop(col, axis=1)
        except ValueError:
            pass
    # Fix any NaNs in ring-current columns
    for atom in atom_names:
        try:
            ring_col = atom + '_RC'
            ring_med = dat[dat[ring_col].notnull()][ring_col].median()
            dat[ring_col] = dat[ring_col].fillna(value=ring_med)
        except KeyError:
            pass

    if mode == 'testing':
        dat_dict = {}
        feats = []
        for atom in atoms:
            dat_dict[atom] = dat[dat[atom].notnull()]#.fillna(value=0)
            these_feats = dat_dict[atom].drop(atom, axis=1)
            for at in atom_names:
                try:
                    these_feats = these_feats.drop(at, axis=1)
                except ValueError:
                    pass
            to_remove = list(set(atom_names) - set(atoms))
            for at in to_remove:
                try:
                    ring_col = at + '_RC'
                    these_feats = these_feats.drop(ring_col, axis=1)
                except KeyError:
                    pass
                except ValueError:
                    pass
                try:
                    rcoil_col = 'RCOIL_' + at
                    these_feats = these_feats.drop(rcoil_col, axis=1)
                except ValueError:
                    pass
            feats.append(these_feats.values)
        shifts_list = [dat_dict[atom][atoms] for atom in atoms]
        shifts_means = len(atoms) * [0]
        shifts_stds = len(atoms) * [1]
        shifts_norms = [shifts.fillna(value=0).values for shifts in shifts_list]
    
    
    elif mode == 'masked':
        mask_value = 10**3
        dat = dat[dat[atoms].isnull().sum(axis=1) < len(atoms)]
        dat.index = pd.RangeIndex(start=0, stop=len(dat), step=1)
        feats = dat.drop(atoms, axis=1)
        if norm_shifts:
            shifts_means = dat[atoms].mean()
            shifts_stds = dat[atoms].std()
            dat[atoms] = (dat[atoms] - shifts_means) / shifts_stds
        else:
            shifts_means = len(atoms) * [0]
            shifts_stds = len(atoms) * [1]
        for atom in atoms:
            dat.loc[dat[atom].isnull(), atom] = mask_value
        shifts_norms = [dat[atom].values for atom in atoms]
        feats = dat.drop(atoms, axis=1)
        for atom in atom_names:
            try:
                feats = feats.drop(atom, axis=1)
            except ValueError:
                pass
        to_remove = list(set(atom_names) - set(atoms))
        for atom in to_remove:
            try:
                ring_col = atom + '_RC'
                feats = feats.drop(ring_col, axis=1)
            except KeyError:
                pass
            except ValueError:
                pass
            try:
                rcoil_col = 'RCOIL_' + atom
                feats = feats.drop(rcoil_col, axis=1)
            except ValueError:
                pass
        feats = feats.values
        
    
    elif mode == 'un-masked':
            
        for atom in atoms:
            dat = dat[dat[atom].notnull()]
            shifts_list = [dat[atom] for atom in atoms]
            
    # Rejecting outliers is not currently implemented for MTL
        if reject_outliers is not None:
            raise ValueError('reject_outliers is not yet implemented for MTL so the value of this kwarg must be None')
    #        n_sig = max(reject_outliers, 0.1) # Don't accept a cutoff less than 0.1 * std
    #        rej_mean = dat[atom].mean()
    #        rej_std = dat[atom].std()
    #        up_rej = rej_mean + n_sig * rej_std
    #        low_rej = rej_mean - n_sig * rej_std
    #        dat = dat[(dat[atom] > low_rej) & (dat[atom] < up_rej)]
    
    
        # Normalize shifts if this is desired
        if norm_shifts:
            shifts_means = [dat[atom].mean() for atom in atoms]
            shifts_stds = [dat[atom].std() for atom in atoms]
            shifts_norms = []
            for i in range(len(atoms)):
                shifts_norms.append((shifts_list[i].values - shifts_means[i]) / shifts_stds[i])
        else:
            shifts_means = len(atoms) * [0]
            shifts_stds = len(atoms) * [1]
            shifts_norms = [shifts.values for shifts in shifts_list]

        # If other atoms are in the data then drop them from features
        feats = dat.drop(atoms[0], axis=1)
        for atom in atom_names:
            try:
                feats = feats.drop(atom, axis=1)
            except ValueError:
                pass
    
        # Need to drop the random coil and ring current columns for the other atoms if 
        # such columns are in the data.
    
        to_remove = list(set(atom_names) - set(atoms))
        for atom in to_remove:
            try:
                ring_col = atom + '_RC'
                feats = feats.drop(ring_col, axis=1)
            except KeyError:
                pass
            except ValueError:
                pass
            try:
                rcoil_col = 'RCOIL_' + atom
                feats = feats.drop(rcoil_col, axis=1)
            except ValueError:
                pass
        
        
    #    for atom in atoms:
    #        try:
    #            ring_col = atom + '_RC'
    #            ring_mean = feats[ring_col].fillna(value=0).mean()
    #            feats[ring_col] = feats[ring_col].fillna(value=ring_mean)
    #        except KeyError:
    #            pass
    #        except ValueError:
    #            pass
    
        feats = feats.values
    
    if reject_outliers is not None:
        pass
#        return feats, shift_norm, rej_mean, rej_std
    else:
        return feats, shifts_norms, shifts_means, shifts_stds


#Here is a a loss function for MTL that allows use of all of the examples with at least 1 non-NaN shift when training a model

def masked_mse(y_true, y_pred):
    mask_value = 10**3
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    return keras.losses.mean_squared_error(y_true * mask, y_pred * mask)


def make_optimizer(opt_type, lrate, mom, dec, nest, clip_norm=False, clip_val=False, opt_override=False):
    '''Function to make an optimizer for use in the model-generating functions
    below.
    
    args:
        lrate - Learning rate for optimizer (Float)
        mom - Momentum for optimizer (Float)
        dec - Decay for optimizer learning rate (Float)
        opt_type - Optimization procedure to use (Str - sgd, rmsprop, adam, etc.)
        nest - Use Nesterov momentum (Bool)
        opt_override - Override default parameters for optimization (Bool)
        clip_val - Clip values parameter for optimizer (Float)
        clip_norm - Clip norm parameter for optimizer (Float)
        
    returns:
        opt - Constructed optimizer (Keras Optimizer)
    '''
        # Define optimization procedure
    if opt_type == 'sgd':
        if opt_override:
            opt = keras.optimizers.SGD(lr=lrate, momentum=mom, decay=dec, nesterov=nest, clipnorm=clip_norm, clipvalue=clip_val)
        else:
            opt = keras.optimizers.SGD()
    elif opt_type == 'rmsprop':
        if opt_override:
            opt = keras.optimizers.RMSprop(lr=lrate, rho=mom, decay=dec, clipnorm=clip_norm, clipvalue=clip_val)
        else:
            opt = keras.optimizers.RMSprop(clipnorm=clip_norm, clipvalue=clip_val)
    elif opt_type == 'adagrad':
        if opt_override:
            opt = keras.optimizers.Adagrad(lr=lrate, decay=dec, clipnorm=clip_norm, clipvalue=clip_val)
        else:
            opt = keras.optimizers.Adagrad(clipnorm=clip_norm, clipvalue=clip_val)
    elif opt_type == 'adadelta':
        if opt_override:
            opt = keras.optimizers.Adadelta(lr=lrate, rho=mom, decay=dec, clipnorm=clip_norm, clipvalue=clip_val)
        else:
            opt = keras.optimizers.Adadelta(clipnorm=clip_norm, clipvalue=clip_val)
    elif opt_type == 'adam':
        if opt_override:
            try:
                beta1 = mom[0]
                beta2 = mom[1]
            except TypeError:
                beta1 = mom
                beta2 = mom
                print('Only one momentum given for adam-type optimizer.  Using th== value for both beta1 and beta2')
            if nest:
                opt = keras.optimizers.Nadam(lr=lrate, beta_1=beta1, beta_2=beta2, schedule_decay=dec, clipnorm=clip_norm, clipvalue=clip_val)
            else:
                opt = keras.optimizers.Adam(lr=lrate, beta_1=beta1, beta_2=beta2, decay=dec, clipnorm=clip_norm, clipvalue=clip_val)
        else:
            if nest:
                opt = keras.optimizers.Nadam(clipnorm=clip_norm, clipvalue=clip_val)
            else:
                opt = keras.optimizers.Adam(clipnorm=clip_norm, clipvalue=clip_val)
    elif opt_type == 'adamax':
        if opt_override:
            try:
                beta1 = mom[0]
                beta2 = mom[1]
            except TypeError:
                beta1 = mom
                beta2 = mom
                print('Only one momentum given for adam-type optimizer.  Using th== value for both beta1 and beta2')
            opt = keras.optimizers.Adamax(lr=lrate, beta_1=mom, beta_2=mom, decay=dec, clipnorm=clip_norm, clipvalue=clip_val)
        else:
            opt = keras.optimizers.Adamax(clipnorm=clip_norm, clipvalue=clip_val)
    
    return opt


def make_reg(reg, reg_type):
    '''Function to construct a weight-matrix regularizer for use in the model-generating functions below.
    
    args:
        reg - Magnitude of the regularization
        reg_type - Type of regualrization to use (Str - "L1", "L2", or "L1_L2" currently supported)
        
    returns:
        regularizer - Regularizer for the weight matrices (Keras Regularizer)
    '''
    if reg_type is None:
        regularizer = keras.regularizers.l2(0)
    if reg_type == 'L1':
        regularizer = keras.regularizers.l1(reg)
    elif reg_type == 'L2':
        regularizer = keras.regularizers.l2(reg)
    elif reg_type == 'L1_L2':
        try:
            l1reg = reg[0]
            l2reg = reg[1]
        except TypeError:
            l1reg = reg
            l2reg = reg
            print('reg_type is L1_L2 but reg was not passed a list.  Using reg for both L1 and L2')
        regularizer = keras.regularizers.l1_l2(l1=l1reg, l2=l2reg)
    
    return regularizer


def early_stopping(mod, method, tol, per, epochs, min_epochs, batch_size, feat_train=None, shift_train=None, feat_val=None, shift_val=None, train_gen=None, train_steps=None, noise=False, mtl=False):
    '''Function to execute an early-stopping routine to determine the "optimal" number of epochs to train.
    
    args:
        mod - Model on which to run the early-stopping routine (Keras Model)
        method - Name of the method used for early stopping (Str - 'GL', 'PQ', or 'UP')
        tol - Tolerance for the method (Float)
        epochs - Maximum number of epochs to train (Int)
        min_epochs - Minimum number of epochs to train (Int)
        batch_size - Size of batch for fitting (Int)
        feat_train - Training set features (Numpy Array or None if Noise)
        shift_train - Training set shifts (Numpy Array or None if Noise)
        feat_val - Validation set features (Numpy Array or None if Noise)
        shift_val - Validation set shifts (Numpy Array or None if Noise)
        train_gen - Generator for training set (Generator or None if not Noise)
        train_steps - Steps per epoch for training generator (Int)
        noise - Whether or not to use noise (Bool)
        mtl - Whether or not the model uses multi-task learning and therefore has multiple outputs and shifts are therefore given as lists (Bool)
        
    returns:
        val_epochs - The identified number of epochs obtained by minimizing the validation error (Int)
        hist_list - List of the training history from the early-stopping routine (List)
        val_list - List of the validation errors from the early-stopping routine (List)
        param_list - List of the successive parameter values of the early-stopping metric (List)
    '''
    # Initialize some outputs
    hist_list = []
    val_list = []
    param_list = []
    
    val_min = 10 ** 10
    up_count = 0
    for i in range(int(epochs/per)):
        if mtl:
            pt1 = mod.evaluate(feat_val, [shift_val[i] for i in range(len(shift_val))], verbose=0)
        else:
            pt1 = mod.evaluate(feat_val, shift_val, verbose=0)
        val_list.append(pt1)

        if pt1 < val_min:
            val_min = pt1

        if noise:
            hist = mod.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=per)
        else:
            if mtl:
                hist = mod.fit(feat_train, [shift_train[i] for i in range(len(shift_train))], batch_size=batch_size, epochs=per)
            else:
                hist = mod.fit(feat_train, shift_train, batch_size=batch_size, epochs=per)
        hist_list += hist.history['loss']
        if mtl:
            pt2 = mod.evaluate(feat_val, [shift_val[i] for i in range(len(shift_val))], verbose=0)
        else:
            pt2 = mod.evaluate(feat_val, shift_val, verbose=0)
        
        if method == 'GL' or method == 'PQ':
            gl = 100 * (pt2/val_min - 1)
            
            if method == 'GL':
                param_list.append(gl)
                if gl > tol and (i+1)*per>=min_epochs:
                    print('Broke loop at round ' + str(i+1))
                    break
            
            if method == 'PQ':
                strip_avg = np.array(hist.history['loss']).mean()
                strip_min = min(np.array(hist.history['loss']))
                p = 1000 * (strip_avg / strip_min - 1)
                pq = gl / p
                param_list.append(pq)
                if pq > tol and (i+1)*per>=min_epochs:
                    print('Broke loop at round ' + str(i+1))
                    break
        
        if method == 'UP':
            if pt2 > pt1:
                up_count += 1
                param_list.append(up_count)
                if up_count >= tol and (i+1)*per>=min_epochs:
                    print('Broke loop at round ' + str(i+1))
                    break
            else:
                up_count = 0
                param_list.append(up_count)
        
        print('The validation loss at round ' + str(i+1) + ' is ' + str(pt2))

    min_epochs_pos=max(int(min_epochs/per-1),0)  
    val_list_after_min_epoch=val_list[min_epochs_pos:]    
    min_val_idx=np.argmin(val_list_after_min_epoch)+min_epochs_pos
    val_epochs=min_val_idx * per
    
    return val_epochs, hist_list, val_list, param_list

def generator_early_stopping(mod, atoms, shifts_std, method, tol, per, epochs, min_epochs, batch_size, train_gen, train_steps, val_gen, val_steps):
    '''Function to execute an early-stopping routine to determine the "optimal" number of epochs to train (for generator).

     args:
        mod - Model on which to run the early-stopping routine (Keras Model)
        atoms - List of atoms the model is trained for (List)
        shifts_std - Standard deviations of the original atom shifts (List or Pandas Dataframe)
        method - Name of the method used for early stopping (Str - 'GL', 'PQ', or 'UP')
        tol - Tolerance for the method (Float)
        per - Number of epochs before each evaluation (Int)
        epochs - Maximum number of epochs to train (Int)
        min_epochs - Minimum number of epochs to train (Int)
        batch_size - Size of batch for fitting (Int)
        train_gen - Generator for training set (Generator)
        train_steps - Steps per epoch for training generator (Int)
        val_gen - Generator for validation set (Generator)
        val_steps - Steps per epoch for validation generator (Int)
        
    returns:
        val_epochs - The identified number of epochs obtained by minimizing the validation error (Int)
        hist_list - List of the training history from the early-stopping routine (List)
        val_list - List of the validation errors from the early-stopping routine (List)
        param_list - List of the successive parameter values of the early-stopping metric (List)
    '''
    # Initialize some outputs
    hist_list = []
    val_list = []
    param_list = []
    
    val_min = 10 ** 10
    up_count = 0
    pt1=val_min
    for i in range(int(epochs/per)):
        if pt1==val_min:
            evaluate_result = mod.evaluate_generator(val_gen,steps=val_steps,use_multiprocessing=USE_MULTIPROCESSING,verbose=VERBOSITY)
            if type(evaluate_result) is np.float64:
                evaluate_result=[evaluate_result,evaluate_result] 
                pt1 = sum(evaluate_result)/2
            val_list.append(pt1)
        else:
            pt1=pt2
        if pt1 < val_min:
            val_min = pt1
        hist = mod.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=per,use_multiprocessing=USE_MULTIPROCESSING,verbose=VERBOSITY)
        hist_list += hist.history['loss']
        evaluate_result=mod.evaluate_generator(val_gen, steps=val_steps,use_multiprocessing=USE_MULTIPROCESSING,verbose=VERBOSITY)
        if type(evaluate_result) is np.float64:
            evaluate_result=[evaluate_result,evaluate_result] 
        pt2 = sum(evaluate_result)/2
        val_list.append(pt2)
        print('The validation loss at round ' + str(i+1) + ' is ' + str(pt2))
        print([atom_type+":"+str(np.sqrt(atom_error)*atom_std) for atom_type,atom_error,atom_std in zip(atoms,evaluate_result[1:],shifts_std)])

        if method == 'GL' or method == 'PQ':
            gl = 100 * (pt2/val_min - 1)
            
            if method == 'GL':
                param_list.append(gl)
                if gl > tol and (i+1)*per>=min_epochs:
                    print('Broke loop at round ' + str(i+1))
                    break
            
            if method == 'PQ':
                strip_avg = np.array(hist.history['loss']).mean()
                strip_min = min(np.array(hist.history['loss']))
                p = 1000 * (strip_avg / strip_min - 1)
                pq = gl / p
                param_list.append(pq)
                if pq > tol and (i+1)*per>=min_epochs:
                    print('Broke loop at round ' + str(i+1))
                    break
        
        if method == 'UP':
            if pt2 > pt1:
                up_count += 1
                param_list.append(up_count)
                if up_count >= tol and (i+1)*per>=min_epochs:
                    print('Broke loop at round ' + str(i+1))
                    break
            else:
                up_count = 0
                param_list.append(up_count)
      
    min_epochs_pos=max(int(min_epochs/per-1),0)  
    val_list_after_min_epoch=val_list[min_epochs_pos:]    
    min_val_idx=np.argmin(val_list_after_min_epoch)+min_epochs_pos
    val_epochs=min_val_idx * per
    return val_epochs, hist_list, val_list, param_list

def fc_model(data, atom, arch, activ='prelu', lrate=0.001, mom=0, dec=10**-6, epochs=100, min_epochs=5, per=5, tol=1.0, opt_type='sgd', do=0.0, drop_last_only=False, reg=0.0, reg_type=None, early_stop=None, es_data=None, bnorm=False, lsuv=False, nest=False, norm_shifts=True, opt_override=False, clip_val=0.0, clip_norm=0.0, reject_outliers=None, noise=None, noise_type='angle', noise_dist='uniform', batch_size=64, lsuv_batch=64):
    '''Constructs a model from the given features and shifts for the requested atom.  The model is trained for the given number of epochs with the loss being checked every per epochs.  Training stops when this loss increases by more than tol. The arch argument is a list specifying the number of hidden units at each layer.
    
    args:
        data - Feature and target data (Pandas DataFrame)
        atom - Name of atom to predict (Str)
        arch - List of the number of neurons per layer (List)
        activ - Activation function to use (Str - prelu, relu, tanh, etc.)
        lrate - Learning rate for SGD (Float)
        mom - Momentum for SGD (Float)
        dec - Decay for SGD learning rate (Float)
        epochs - Maximum number of epochs to train (Int)
        min_epochs - Minimum number of epochs to train (Int)
        per - Number of epochs in a test strip for early-stopping (Int)
        tol - Early-stopping parameter (Float)
        opt_type - Optimization procedure to use (Str - sgd, rmsprop, adam, etc.)
        do - Dropout percentage between 0 and 1 (Float)
        drop_last_only - Apply dropout only at last layer (Bool)
        reg - Parameter for weight regularization of dense layers (Float)
        reg_type - Type of weight regularization to use (Str - L1 or L2)
        early-stop - Whether or not and how to do early-stopping (None or Str - 'GL', 'PQ', or 'UP')
        es_data - Data with which to do the early-stopping (None or Pandas DataFrame)
        bnorm - Use batch normalization (Bool)
        lsuv - Use layer-wise sequential unit variance initialization (Bool)
        nest - Use Nesterov momentum (Bool)
        norm_shifts - Normalize the target shifts rather than using raw values (Bool)
        opt_override - Override default parameters for optimization (Bool)
        clip_val - Clip values parameter for optimizer (Float)
        clip_norm - Clip norm parameter for optimizer (Float)
        reject_outliers - Eliminate entries for which the targets differ by more than this number of standard deviations (Float or None)
        noise - Dictionary of noise magnitudes for each column if using fit_generator and
                want to inject noise (Dict or Float/Int)
        noise_type - Argument to be fed to noise generator (Str) -- See res_level_generator()
        noise_dist - Distribution from which to draw noise values -- "uniform" or "normal" (Str)
        batch_size - Number of examples per batch (Int)
        lsuv_batch - Number of examples used for LSUV initialization
        
    returns:
        
    '''    
    feats, shifts, shifts_mean, shifts_std = data_prep(data, atom, reject_outliers=reject_outliers, norm_shifts=norm_shifts)

    dim_in = feats.shape[1]
    
    # If using noise, make data generators
    if noise:
        if early_stop and (es_data is not None):
            raise ValueError('Early-stopping routine is not setup to handle both an explicit early-stopping data set and noise generation.')
        train_dat, val_dat = skl.model_selection.train_test_split(data, test_size=0.2, random_state=seed)
        train_gen = res_level_generator(train_dat, atom, noise=noise, noise_type=noise_type, noise_dist=noise_dist, batch_size=batch_size, norm_shifts=(shifts_mean, shifts_std))
        full_gen = res_level_generator(train_dat, atom, noise=noise, noise_type=noise_type, noise_dist=noise_dist, batch_size=batch_size, norm_shifts=(shifts_mean, shifts_std))
        n_train = len(train_dat) // batch_size # Number of full batches per epoch
        rem_train = len(train_dat) % batch_size # Size of extra batch
        n_full = len(data) // batch_size
        rem_full = len(data) % batch_size
        train_steps = min(rem_train, 1) + n_train
        full_steps = min(rem_full, 1) + n_full

    # Set a boolean variable for presence of dropout layers
    if (float(do) > 0) and (not drop_last_only):
        dropout = True
    else:
        dropout = False

    opt = make_optimizer(opt_type, lrate, mom, dec, nest, clip_norm=clip_norm, clip_val=clip_val, opt_override=opt_override)
    
    regularizer = make_reg(reg, reg_type)
    
    # Build model
    mod = keras.models.Sequential()
    if activ is 'prelu':
        mod.add(keras.layers.Dense(units=arch[0], activation='linear', input_dim=dim_in, kernel_regularizer=regularizer))
        mod.add(keras.layers.advanced_activations.PReLU())
    else:
        mod.add(keras.layers.Dense(units=arch[0], activation=activ, input_dim=dim_in, kernel_regularizer=regularizer))
    if bnorm:
        mod.add(keras.layers.BatchNormalization())
    if dropout:
        mod.add(keras.layers.Dropout(do))
    for i in arch[1:]:
        if activ is 'prelu':
            mod.add(keras.layers.Dense(units=i, activation='linear', kernel_regularizer=regularizer))
            mod.add(keras.layers.advanced_activations.PReLU())
        else:
            mod.add(keras.layers.Dense(units=i, activation=activ, kernel_regularizer=regularizer))
        if bnorm:
            mod.add(keras.layers.BatchNormalization())
        if dropout:
            mod.add(keras.layers.Dropout(do))
    if drop_last_only:
        mod.add(keras.layers.Dropout(do))
    mod.add(keras.layers.Dense(units=1, activation='linear'))
    mod.compile(loss='mean_squared_error', optimizer=opt)
    
    if lsuv:
        mod = LSUVinit(mod, feat_train[:lsuv_batch])
        
    # Get initial weights to reset model after pretraining
    weights = mod.get_weights()

    # Do early-stopping routine if desired 
    if early_stop is not None:
        if noise:
            val_epochs, hist_list, val_list, param_list = early_stopping(mod, early_stop, tol, per, epochs, min_epochs, batch_size, feat_train=None, shift_train=None, feat_val=feat_val, shift_val=shift_val, train_gen=train_gen, train_steps=train_steps, noise=noise)
        else:
            if es_data is None:
                # Split up the data into train and validation
                seed = np.random.randint(1, 100)
                feat_train, feat_val, shift_train, shift_val = skl.model_selection.train_test_split(feats, shifts, test_size=0.2, random_state=seed)
                val_epochs, hist_list, val_list, param_list = early_stopping(mod, early_stop, tol, per, epochs, min_epochs, batch_size, feat_train=feat_train, shift_train=shift_train, feat_val=feat_val, shift_val=shift_val, train_gen=None, train_steps=None, noise=noise)
            else:
                val_feats, val_shifts, _, _ = data_prep(es_data, atom, reject_outliers=None, norm_shifts=norm_shifts, norm_stats=[shifts_mean, shifts_std]) #Prepare es_data using statistics from training set 
                val_epochs, hist_list, val_list, param_list = early_stopping(mod, early_stop, tol, per, epochs, min_epochs, batch_size, feat_train=feats, shift_train=shifts, feat_val=val_feats, shift_val=val_shifts, train_gen=None, train_steps=None, noise=False)
    else:
        val_epochs = epochs
        val_list = []
        hist_list = []
        param_list = []

    # Retrain model on full dataset for same number of epochs as was
    # needed to obtain min validation loss
    mod.set_weights(weights)
    if es_data is None:    
        if noise:
            mod.fit_generator(full_gen, steps_per_epoch=full_steps, epochs=val_epochs)
        else:
            mod.fit(feats, shifts, batch_size=batch_size, epochs=val_epochs)
    else:
        data_all=pd.concat([data,es_data],ignore_index=True)
        feats, shifts, shifts_mean, shifts_std = data_prep(data_all, atom, reject_outliers=reject_outliers, norm_shifts=norm_shifts)
        mod.fit(feats,shifts, batch_size=batch_size, epochs=val_epochs)
    return shifts_mean, shifts_std, val_list, hist_list, param_list, mod

def outer_product(inputs):
    """
    inputs: list of two tensors (of equal dimensions, 
        for which you need to compute the outer product
    """
    x, y = inputs
    batchSize = K.shape(x)[0]
    outerProduct = x[:,:, newaxis] * y[:,newaxis,:]
    outerProduct = K.reshape(outerProduct, (batchSize, -1))
    # returns a flattened batch-wise set of tensors
    return outerProduct

def numerical_splitting_model(data, atom, arch, skip_connection=True, has_class= False,class_bins=20, activ='prelu', lrate=0.001, mom=0, dec=10**-6, epochs=100, min_epochs=5, per=5, tol=1.0, opt_type='sgd', do=0.0, drop_last_only=False, reg=0.0, reg_type=None, early_stop=None, es_data=None, bnorm=False, lsuv=False, nest=False, norm_shifts=True, opt_override=False, clip_val=0.0, clip_norm=0.0, reject_outliers=None, noise=None, noise_type='angle', noise_dist='uniform', batch_size=64, lsuv_batch=64):
    '''Constructs a model from the given features and shifts for the requested atom.  The model is trained for the given number of epochs with the loss being checked every per epochs.  Training stops when this loss increases by more than tol. The architecture is formed with the numerical network and non-numerical network doing an outer product at some level. More descriptions below.
    
    args:
        data - Feature and target data (Pandas DataFrame)
        atom - Name of atom to predict (Str)
        arch - List with structure [[num_neurons],[non-num_neurons],[neurons_after_outer_product]] (List)
        skip_connection - Whether set skip connections for numerical and non-numerical inputs and concatenate them with the outer product layer (Bool)
        has_class - Whether class labels are contained in the features
        class_bins - Number of bins for classifying the shifts into (in input) (int)
        activ - Activation function to use (Str - prelu, relu, tanh, etc.)
        lrate - Learning rate for SGD (Float)
        mom - Momentum for SGD (Float)
        dec - Decay for SGD learning rate (Float)
        epochs - Maximum number of epochs to train (Int)
        min_epochs - Minimum number of epochs to train (Int)
        per - Number of epochs in a test strip for early-stopping (Int)
        tol - Early-stopping parameter (Float)
        opt_type - Optimization procedure to use (Str - sgd, rmsprop, adam, etc.)
        do - Dropout percentage between 0 and 1 (Float)
        drop_last_only - Apply dropout only at last layer (Bool)
        reg - Parameter for weight regularization of dense layers (Float)
        reg_type - Type of weight regularization to use (Str - L1 or L2)
        early-stop - Whether or not and how to do early-stopping (None or Str - 'GL', 'PQ', or 'UP')
        bnorm - Use batch normalization (Bool)
        lsuv - Use layer-wise sequential unit variance initialization (Bool)
        nest - Use Nesterov momentum (Bool)
        norm_shifts - Normalize the target shifts rather than using raw values (Bool)
        opt_override - Override default parameters for optimization (Bool)
        clip_val - Clip values parameter for optimizer (Float)
        clip_norm - Clip norm parameter for optimizer (Float)
        reject_outliers - Eliminate entries for which the targets differ by more than this number of standard deviations (Float or None)
        noise - Dictionary of noise magnitudes for each column if using fit_generator and
                want to inject noise (Dict or Float/Int)
        noise_type - Argument to be fed to noise generator (Str) -- See res_level_generator()
        noise_dist - Distribution from which to draw noise values -- "uniform" or "normal" (Str)
        batch_size - Number of examples per batch (Int)
        lsuv_batch - Number of examples used for LSUV initialization
        
    returns:
        
    '''

    feats, shifts, shifts_mean, shifts_std = data_prep(data, atom, reject_outliers=reject_outliers, norm_shifts=norm_shifts,split_numeric=True,split_class=True)
    if has_class:
        feats_numeric,feats_non_numeric,feats_class=feats
    else:
        feats_numeric,feats_non_numeric=feats
    # Split up the data into train and validation
    
    
    dim_num_in = feats_numeric.shape[1]
    dim_non_num_in = feats_non_numeric.shape[1]
    
    # If using noise, make data generators
    if noise:
        if early_stop and (es_data is not None):
            raise ValueError('Early-stopping routine is not setup to handle both an explicit early-stopping data set and noise generation.')
        train_dat, val_dat = skl.model_selection.train_test_split(data, test_size=0.2, random_state=seed)
        train_gen = res_level_generator(train_dat, atom, noise=noise, noise_type=noise_type, noise_dist=noise_dist, batch_size=batch_size, norm_shifts=(shifts_mean, shifts_std))
        full_gen = res_level_generator(train_dat, atom, noise=noise, noise_type=noise_type, noise_dist=noise_dist, batch_size=batch_size, norm_shifts=(shifts_mean, shifts_std))
        n_train = len(train_dat) // batch_size # Number of full batches per epoch
        rem_train = len(train_dat) % batch_size # Size of extra batch
        n_full = len(dat) // batch_size
        rem_full = len(dat) % batch_size
        train_steps = min(rem_train, 1) + n_train
        full_steps = min(rem_full, 1) + n_full

    # Set a boolean variable for presence of dropout layers
    if (float(do) > 0) and (not drop_last_only):
        dropout = True
    else:
        dropout = False

    opt = make_optimizer(opt_type, lrate, mom, dec, nest, clip_norm=clip_norm, clip_val=clip_val, opt_override=opt_override)
    
    regularizer = make_reg(reg, reg_type)
    
    # Build model
    inp_num = keras.layers.Input((dim_num_in,))
    inp_non_num = keras.layers.Input((dim_non_num_in,))
    layer_num=inp_num
    layer_non_num=inp_non_num
    for node in arch[0]:
        if activ is 'prelu':
            layer_num=keras.layers.Dense(units=node, activation='linear',kernel_regularizer=regularizer)(layer_num)
            layer_num=keras.layers.advanced_activations.PReLU()(layer_num)
        else:
            layer_num=keras.layers.Dense(units=node, activation=activ,kernel_regularizer=regularizer)(layer_num)
        if bnorm:
            layer_num=keras.layers.BatchNormalization()(layer_num)
        if dropout:
            layer_num=keras.layers.Dropout(do)(layer_num)

    for node in arch[1]:
        if activ is 'prelu':
            layer_non_num=keras.layers.Dense(units=node, activation='linear',kernel_regularizer=regularizer)(layer_non_num)
            layer_non_num=keras.layers.advanced_activations.PReLU()(layer_non_num)
        else:
            layer_non_num=keras.layers.Dense(units=node, activation=activ,kernel_regularizer=regularizer)(layer_non_num)
        if bnorm:
            layer_non_num=keras.layers.BatchNormalization()(layer_non_num)
        if dropout:
            layer_non_num=keras.layers.Dropout(do)(layer_non_num)
    
    layer=keras.layers.Lambda(outer_product, output_shape=(arch[0][-1]*arch[1][-1], ))([layer_num, layer_non_num])
    concat_layer=[layer]
    if skip_connection:
        concat_layer.extend([inp_num,inp_non_num])
    if has_class:
        inp_class=keras.layers.Input((class_bins,))
        concat_layer.append(inp_class)
    layer=keras.layers.Concatenate()(concat_layer)
    for node in arch[2]:
        if activ is 'prelu':
            layer=keras.layers.Dense(units=node, activation='linear',kernel_regularizer=regularizer)(layer)
            layer=keras.layers.advanced_activations.PReLU()(layer)
        else:
            layer=keras.layers.Dense(units=node, activation=activ,kernel_regularizer=regularizer)(layer)
        if bnorm:
            layer=keras.layers.BatchNormalization()(layer)
        if dropout:
            layer=keras.layers.Dropout(do)(layer)

    output=keras.layers.Dense(units=1, activation='linear')(layer)
    inputs=[inp_num,inp_non_num]
    if has_class:
        inputs.append(inp_class)
    mod=keras.models.Model(input=inputs,output=output)
    mod.compile(loss='mean_squared_error', optimizer=opt)
    try:    
        plot_model(mod,"model.png",show_shapes=True)
    except:
        pass 
    if lsuv:
        mod = LSUVinit(mod, feat_train[:lsuv_batch])
        
    # Get initial weights to reset model after pretraining
    weights = mod.get_weights()

    # Do early-stopping routine if desired 
    if early_stop is not None:
        if noise:
            val_epochs, hist_list, val_list, param_list = early_stopping(mod, early_stop, tol, per, epochs, min_epochs, batch_size, feat_train=None, shift_train=None, feat_val=feat_val, shift_val=shift_val, train_gen=train_gen, train_steps=train_steps, noise=noise)
        else:
            if es_data is None:
                seed = np.random.randint(1, 100)
                feat_num_train, feat_num_val, feat_non_num_train, feat_non_num_val, shift_train, shift_val = skl.model_selection.train_test_split(feats_numeric,feats_non_numeric, shifts, test_size=0.2, random_state=seed)
                val_epochs, hist_list, val_list, param_list = early_stopping(mod, early_stop, tol, per, epochs, min_epochs, batch_size, feat_train=[feat_num_train,feat_non_num_train], shift_train=shift_train, feat_val=[feat_num_val,feat_non_num_val], shift_val=shift_val, train_gen=None, train_steps=None, noise=noise)
            else:
                val_feats, val_shifts, _, _ = data_prep(es_data, atom, reject_outliers=None, norm_shifts=norm_shifts, norm_stats=[shifts_mean, shifts_std],split_numeric=True,split_class=has_class) #Prepare es_data using statistics from training set 
                val_epochs, hist_list, val_list, param_list = early_stopping(mod, early_stop, tol, per, epochs, min_epochs, batch_size, feat_train=feats, shift_train=shifts, feat_val=val_feats, shift_val=val_shifts, train_gen=None, train_steps=None, noise=False)
    else:
        val_epochs = epochs
        val_list = []
        hist_list = []
        param_list = []

    # Retrain model on full dataset for same number of epochs as was
    # needed to obtain min validation loss
    mod.set_weights(weights)
    if es_data is None:
        if noise:
            mod.fit_generator(full_gen, steps_per_epoch=full_steps, epochs=val_epochs)
        else:
            #mod.fit(feats, shift_norm, batch_size=batch_size, epochs=val_epochs)
            mod.fit(list(feats), shifts, batch_size=batch_size, epochs=val_epochs)
    else:
        data_all=pd.concat([data,es_data],ignore_index=True)
        feats, shifts, shifts_mean, shifts_std = data_prep(data_all, atom, reject_outliers=reject_outliers, norm_shifts=norm_shifts,split_numeric=True,split_class=has_class)
        mod.fit(feats,shifts, batch_size=batch_size, epochs=val_epochs)
    return shifts_mean, shifts_std, val_list, hist_list, param_list, mod

def conv_1d_block(num_nodes,pooling_filter,pooling_stride,inp,activ="elu",pooling="average"):
    layer=inp
    for idx,node in enumerate(num_nodes):
        conv=keras.layers.Conv1D(node,kernel_size=1 if idx<len(num_nodes)-1 else 3,padding="same")(layer)
        bn=keras.layers.BatchNormalization()(conv)
        if activ=="relu":
            layer=keras.layers.ReLU()(bn)
        elif activ=="prelu":
            layer=keras.layers.PReLU()(bn)
        elif activ=="elu":
            layer=keras.layers.ELU()(bn)
        else:
            layer=bn
    if pooling=="average":
        output=keras.layers.AveragePooling1D(pooling_filter,strides=pooling_stride)(layer)
    elif pooling=="max":
        output=keras.layers.MaxPooling1D(pooling_filter,strides=pooling_stride)(layer)
    return output

def conv_block(input_layer,num_units,filter_size):
    layer=input_layer
    for unit,filter in zip(num_units,filter_size):
        layer=Conv1D(unit,filter,padding="same")(layer)
        layer=BatchNormalization()(layer)
        layer=ELU()(layer)
    return layer

def dense_block(input_layer,n):
    conv1=conv_block(input_layer,[n,n,n],[3,3,5])
    conv2=conv_block(conv1,[n,n,n],[3,3,5])
    comb1=Concatenate()([conv1,conv2])
    conv3=conv_block(comb1,[n,n,n],[3,3,5])
    comb2=Concatenate()([conv1,conv2,conv3])
    conv4=conv_block(comb2,[n,n,n],[3,3,5])
    ap=AveragePooling1D(3,strides=2)(conv4)
    return ap

def cnn_model(data, atom,arch, activ='elu', lrate=0.001, mom=0, dec=10**-6, epochs=100, min_epochs=5, per=5, tol=1.0, opt_type='sgd', do=0.0, drop_last_only=False, reg=0.0, reg_type=None, early_stop=None, es_data=None,val_split=0.2, bnorm=False, lsuv=False, nest=False, norm_shifts=True, opt_override=False, clip_val=0.0, clip_norm=0.0, reject_outliers=None, noise=None, noise_type='angle', noise_dist='uniform', batch_size=64, lsuv_batch=64,window=7):
    '''Constructs a model from the given features and shifts for the requested atom.  The model is trained for the given number of epochs with the loss being checked every per epochs.  Training stops when this loss increases by more than tol. The arch argument is a list specifying the number of hidden units at each layer.
    
    args:
        data - Feature and target data (Pandas DataFrame)
        atom - Name of atom to predict (Str)
        arch - List of the number of neurons per layer (List)
        activ - Activation function to use (Str - prelu, relu, tanh, etc.)
        lrate - Learning rate for SGD (Float)
        mom - Momentum for SGD (Float)
        dec - Decay for SGD learning rate (Float)
        epochs - Maximum number of epochs to train (Int)
        min_epochs - Minimum number of epochs to train (Int)
        per - Number of epochs in a test strip for early-stopping (Int)
        tol - Early-stopping parameter (Float)
        opt_type - Optimization procedure to use (Str - sgd, rmsprop, adam, etc.)
        do - Dropout percentage between 0 and 1 (Float)
        drop_last_only - Apply dropout only at last layer (Bool)
        reg - Parameter for weight regularization of dense layers (Float)
        reg_type - Type of weight regularization to use (Str - L1 or L2)
        early-stop - Whether or not and how to do early-stopping (None or Str - 'GL', 'PQ', or 'UP')
        es_data = Dataframe for doing early-stopping, used for fixed train/val/test split
        bnorm - Use batch normalization (Bool)
        lsuv - Use layer-wise sequential unit variance initialization (Bool)
        nest - Use Nesterov momentum (Bool)
        norm_shifts - Normalize the target shifts rather than using raw values (Bool)
        opt_override - Override default parameters for optimization (Bool)
        clip_val - Clip values parameter for optimizer (Float)
        clip_norm - Clip norm parameter for optimizer (Float)
        reject_outliers - Eliminate entries for which the targets differ by more than this number of standard deviations (Float or None)
        noise - Dictionary of noise magnitudes for each column if using fit_generator and
                want to inject noise (Dict or Float/Int)
        noise_type - Argument to be fed to noise generator (Str) -- See res_level_generator()
        noise_dist - Distribution from which to draw noise values -- "uniform" or "normal" (Str)
        batch_size - Number of examples per batch (Int)
        lsuv_batch - Number of examples used for LSUV initialization
        
    returns:
        
    '''
    num_shifts=len(atom)
    # Predicting only center residue requires odd window
    if rolling and center_only:
        if window % 2 is 0:
            window += 1

    data_gen,steps,num_feats,shifts_mean,shifts_std=rnn_data_prep(data,atom,window,batch_size,norm_shifts,val_split=val_split if es_data is None else None)
    
    opt = make_optimizer(opt_type, lrate, mom, dec, nest, clip_norm=clip_norm, clip_val=clip_val, opt_override=opt_override)
    
    regularizer = make_reg(reg, reg_type)
    # Build model
    input_layer=keras.layers.Input((window,num_feats))
    block1=conv_1d_block([150,100,100],4,1,input_layer,activ)
    block2=conv_1d_block([100,50,50],2,2,block1,activ)
    flattened=keras.layers.Flatten()(block2)
    fc=keras.layers.Dense(50)(flattened)
    if activ=="relu":
            fc=keras.layers.ReLU()(fc)
    elif activ=="prelu":
        fc=keras.layers.PReLU()(fc)
    elif activ=="elu":
        fc=keras.layers.ELU()(fc)
    head_layers=[]
    for _ in atom:
        head_fc=fc
        head_fc=keras.layers.Dense(32)(head_fc)
        if activ=="relu":
            head_fc=keras.layers.ReLU()(head_fc)
        elif activ=="prelu":
            head_fc=keras.layers.PReLU()(head_fc)
        elif activ=="elu":
            head_fc=keras.layers.ELU()(head_fc)
        output=keras.layers.Dense(1)(head_fc)
        head_layers.append(output)
    mod=keras.models.Model(inputs=input_layer,outputs=head_layers)
    mod.compile(loss='mean_squared_error', optimizer=opt)
    print(mod.summary())
    if lsuv:
        mod = LSUVinit(mod, feat_train[:lsuv_batch])
        
    # Get initial weights to reset model after pretraining
    weights = mod.get_weights()
    if early_stop is not None:
        if es_data is None:
            train_gen,val_gen,full_gen=data_gen
            train_steps,val_steps,full_steps=steps
            val_epochs, hist_list, val_list, param_list=generator_early_stopping(mod,atom,shifts_std,early_stop,tol,per,epochs,min_epochs,batch_size,train_gen,train_steps,val_gen,val_steps)
        else:
            val_gen,val_steps,_,_,_=rnn_data_prep(es_data,atom,window,batch_size,norm_stats=(shifts_mean,shifts_std))
            val_epochs, hist_list, val_list, param_list=generator_early_stopping(mod,atom,shifts_std,early_stop,tol,per,epochs,min_epochs,batch_size,data_gen,steps,val_gen,val_steps)
    else:
        val_epochs = epochs
        val_list = []
        hist_list = []
        param_list = []
        retrain=True
    if retrain:
        mod.set_weights(weights)
        if es_data is None:
            mod.fit_generator(full_gen, steps_per_epoch=full_steps, epochs=val_epochs,verbose=VERBOSITY0,use_multiprocessing=USE_MULTIPROCESSING,workers=1)
        else:
            data_all=pd.concat([data,es_data],ignore_index=True)
            data_gen,steps,num_feats,shifts_mean,shifts_std = rnn_data_prep(data_all, atom,window,batch_size,norm_shifts)
            mod.fit_generator(data_gen, steps_per_epoch=steps, epochs=val_epochs,verbose=VERBOSITY,use_multiprocessing=USE_MULTIPROCESSING,workers=1)
    return shifts_mean, shifts_std, val_list, hist_list, param_list, mod
    

def hard_mtl_fc(data, atoms, arch, activ='prelu', lrate=0.001, mom=0, dec=10**-6, epochs=100, min_epochs=5, per=5, tol=1.0, opt_type='sgd', do=0.0, drop_last_only=False, reg=0.0, reg_type=None,
                   early_stop=None, bnorm=False, lsuv=False, nest=False, norm_shifts=True, opt_override=False, clip_val=0.0, clip_norm=0.0, reject_outliers=None, noise=None, 
                   noise_type='angle', noise_dist='uniform', batch_size=64, lsuv_batch=64):
    '''Constructs a model from the given features and shifts for
    the requested atom.  The model is trained for the given number
    of epochs with the loss being checked every per epochs.  Training
    stops when this loss increases by more than tol. The arch argument
    is a list specifying the number of hidden units at each layer.
    
    args:
        data - Feature and target data (Pandas DataFrame)
        atom - Name of atom to predict (Str)
        arch - List of the number of neurons per layer (List)
        activ - Activation function to use (Str - prelu, relu, tanh, etc.)
        lrate - Learning rate for SGD (Float)
        mom - Momentum for SGD (Float)
        dec - Decay for SGD learning rate (Float)
        epochs - Maximum number of epochs to train (Int)
        min_epochs - Minimum number of epochs to train (Int)
        per - Number of epochs in a test strip for early-stopping (Int)
        tol - Early-stopping parameter (Float)
        opt_type - Optimization procedure to use (Str - sgd, rmsprop, adam, etc.)
        do - Dropout percentage between 0 and 1 (Float)
        drop_last_only - Apply dropout only at last layer (Bool)
        reg - Parameter for weight regularization of dense layers (Float)
        reg_type - Type of weight regularization to use (Str - L1 or L2)
        early-stop - Whether or not and how to do early-stopping (None or Str - 'GL', 'PQ', or 'UP')
        bnorm - Use batch normalization (Bool)
        lsuv - Use layer-wise sequential unit variance initialization (Bool)
        nest - Use Nesterov momentum (Bool)
        norm_shifts - Normalize the target shifts rather than using raw values (Bool)
        opt_override - Override default parameters for optimization (Bool)
        clip_val - Clip values parameter for optimizer (Float)
        clip_norm - Clip norm parameter for optimizer (Float)
        reject_outliers - Eliminate entries for which the targets differ by more than this number of standard deviations (Float or None)
        noise - Dictionary of noise magnitudes for each column if using fit_generator and
                want to inject noise (Dict or Float/Int)
        noise_type - Argument to be fed to noise generator (Str) -- See res_level_generator()
        noise_dist - Distribution from which to draw noise values -- "uniform" or "normal" (Str)
        batch_size - Number of examples per batch (Int)
        lsuv_batch - Number of examples used for LSUV initialization
        
    returns:
        
    '''
    dat = data.copy()
    for atom in atoms:
        dat = dat[dat[atom].notnull()]
        
    
    feats, shifts_list, shifts_means, shifts_stds = mtl_data_prep(dat, atoms, reject_outliers=reject_outliers, norm_shifts=norm_shifts)

    # Split up the data into train and validation
    seed = np.random.randint(1, 100)
    train_idxs, val_idxs = next(skl.model_selection.ShuffleSplit(n_splits=1, random_state=seed, test_size=0.2, train_size=None).split(feats))
    feat_train, feat_val = feats[train_idxs], feats[val_idxs]
    shifts_train_list = []
    shifts_val_list = []
    for i in range(len(atoms)):
        shifts_train_list.append(shifts_list[0][train_idxs])
        shifts_val_list.append(shifts_list[0][val_idxs])
    dim_in = feats.shape[1]
    
    # If using noise, make data generators -- Not Implemented for MTL
    if noise:
        raise ValueError('Noise is not yet implemented for MTL so this kwarg must have value None')
#        train_dat, val_dat = skl.model_selection.train_test_split(data, test_size=0.2, random_state=seed)
#        train_gen = res_level_generator(train_dat, atom, noise=noise, noise_type=noise_type, noise_dist=noise_dist, batch_size=batch_size, norm_shifts=(shifts_mean, shifts_std))
#        full_gen = res_level_generator(train_dat, atom, noise=noise, noise_type=noise_type, noise_dist=noise_dist, batch_size=batch_size, norm_shifts=(shifts_mean, shifts_std))
#        n_train = len(train_dat) // batch_size # Number of full batches per epoch
#        rem_train = len(train_dat) % batch_size # Size of extra batch
#        n_full = len(dat) // batch_size
#        rem_full = len(dat) % batch_size
#        train_steps = min(rem_train, 1) + n_train
#        full_steps = min(rem_full, 1) + n_full

    # Set a boolean variable for presence of dropout layers
    if (float(do) > 0) and (not drop_last_only):
        dropout = True
    else:
        dropout = False

    opt = make_optimizer(opt_type, lrate, mom, dec, nest, clip_norm=clip_norm, clip_val=clip_val, opt_override=opt_override)
    
    regularizer = make_reg(reg, reg_type)
    
    # Build model
    # Process all features together first then give separate caps for each output
    in_feats = keras.layers.Input(shape=(dim_in,), name='feature_input')
    inp = keras.layers.Lambda(lambda x: x)(in_feats)
    # Define a dictionary to hold the intermediate layers for each separate cap
    atom_layer_dict = {}
    for i in arch[0]:
        if activ == 'prelu':
            inp = keras.layers.Dense(i, activation='linear', kernel_regularizer=regularizer)(inp)
            inp = keras.layers.advanced_activations.PReLU()(inp)
        else:
            inp = keras.layers.Dense(i, activation=activ, kernel_regularizer=regularizer)(inp)
        if bnorm:
            inp = keras.layers.BatchNormalization()(inp)
        if dropout:
            inp = keras.layers.Dropout(do)(inp)

    for atom in atoms:
        atom_layer_dict[atom] = keras.layers.Lambda(lambda x: x)(inp)
        for i in arch[1]:
            if activ == 'prelu':
                atom_layer_dict[atom] = keras.layers.Dense(i, activation='linear', kernel_regularizer=regularizer)(atom_layer_dict[atom])
                atom_layer_dict[atom] = keras.layers.advanced_activations.PReLU()(atom_layer_dict[atom])
            else:
                atom_layer_dict[atom] = keras.layers.Dense(i, activation=activ, kernel_regularizer=regularizer)(atom_layer_dict[atom])
            if bnorm:
                atom_layer_dict[atom] = keras.layers.BatchNormalization()(atom_layer_dict[atom])
            if dropout:
                atom_layer_dict[atom] = keras.layers.Dropout(do)(atom_layer_dict[atom])
            
        atom_layer_dict[atom] = keras.layers.Dense(1, activation='linear', name=atom+'_out')(atom_layer_dict[atom])

    # Define model and compile
    mod = keras.models.Model(inputs=in_feats, outputs=[atom_layer_dict[atom] for atom in atoms])
    mod.compile(loss='mean_squared_error', optimizer=opt)
    
       
#    if lsuv:
#        mod = LSUVinit(mod, feat_train[:lsuv_batch])
        
    # Get initial weights to reset model after pretraining
    weights = mod.get_weights()

    # Do early-stopping routine if desired 
    if early_stop is not None:
        if noise is not None:
            raise ValueError('Noise not implemented so value for this kwarg must be None')
#            val_epochs, hist_list, val_list, param_list = early_stopping(mod, early_stop, tol, per, epochs, min_epochs, batch_size, feat_train=None, shift_train=None, feat_val=feat_val, shift_val=shift_val, train_gen=train_gen, train_steps=train_steps, noise=noise)
        else:
            val_epochs, hist_list, val_list, param_list = early_stopping(mod, early_stop, tol, per, epochs, min_epochs, batch_size, feat_train=feat_train, shift_train=shifts_train_list, feat_val=feat_val, shift_val=shifts_val_list, train_gen=None, train_steps=None, noise=noise, mtl=True)
    else:
        hist_list = []
        val_list = []
        param_list = []
        val_epochs = epochs

    # Retrain model on full dataset for same number of epochs as was
    # needed to obtain min validation loss
    mod.set_weights(weights)
    if noise:
        pass
#        mod.fit_generator(full_gen, steps_per_epoch=full_steps, epochs=val_epochs)
    else:
        #mod.fit(feats, shift_norm, batch_size=batch_size, epochs=val_epochs)
        mod.fit(feats, shifts_list, batch_size=batch_size, epochs=val_epochs)

    return shifts_means, shifts_stds, val_list, hist_list, param_list, mod



def stitch_mtl_fc(data, atoms, arch, activ='prelu', lrate=[0.001, 0.001], mom=0, dec=10**-6, epochs=[100, 100, 50], min_epochs=5, per=5, tol=1.0, opt_type='sgd', do=0.0, drop_last_only=False, reg=0.0, reg_type=None, early_stop=None, bnorm=False, lsuv=False, nest=False, norm_shifts=True, opt_override=False, clip_val=0.0, clip_norm=0.0, opt2_kwargs={}, cs_mult=100, reject_outliers=None, noise=None, cs_mode='affine_combo', hard_burnin=0, noise_type='angle', noise_dist='uniform', batch_size=64, lsuv_batch=2048, residual=False, masked=False):
    '''Constructs a model from the given features and shifts for
    the requested atom.  The model is trained for the given number
    of epochs with the loss being checked every per epochs.  Training
    stops when this loss increases by more than tol. The arch argument
    is a list specifying the number of hidden units at each layer.
    
    args:
        data - Feature and target data (Pandas DataFrame)
        atom - Name of atom to predict (Str)
        arch - List of the number of neurons per layer.  If residual, then all layers must be the same width (List)
        activ - Activation function to use (Str - prelu, relu, tanh, etc.)
        lrate - Learning rate for SGD (Float)
        mom - Momentum for SGD (Float)
        dec - Decay for SGD learning rate (Float)
        epochs - Maximum number of epochs to train (Int)
        min_epochs - Minimum number of epochs to train (Int)
        per - Number of epochs in a test strip for early-stopping (Int)
        tol - Early-stopping parameter (Float)
        opt_type - Optimization procedure to use (Str - sgd, rmsprop, adam, etc.)
        do - Dropout percentage between 0 and 1 (Float)
        drop_last_only - Apply dropout only at last layer (Bool)
        reg - Parameter for weight regularization of dense layers (Float)
        reg_type - Type of weight regularization to use (Str - L1 or L2)
        early-stop - Whether or not and how to do early-stopping (None or Str - 'GL', 'PQ', or 'UP')
        bnorm - Use batch normalization (Bool)
        lsuv - Use layer-wise sequential unit variance initialization (Bool)
        nest - Use Nesterov momentum (Bool)
        norm_shifts - Normalize the target shifts rather than using raw values (Bool)
        opt_override - Override default parameters for optimization (Bool)
        clip_val - Clip values parameter for optimizer (Float)
        clip_norm - Clip norm parameter for optimizer (Float)
        reject_outliers - Eliminate entries for which the targets differ by more than this number of standard deviations (Float or None)
        noise - Dictionary of noise magnitudes for each column if using fit_generator and
                want to inject noise (Dict or Float/Int)
        noise_type - Argument to be fed to noise generator (Str) -- See res_level_generator()
        noise_dist - Distribution from which to draw noise values -- "uniform" or "normal" (Str)
        batch_size - Number of examples per batch (Int)
        lsuv_batch - Number of examples used for LSUV initialization (Int)
        cs_mult - Factor by which the cross-stitch layers train faster than the sub-networks (Int)
        cs_mode - Method of combining sub-task layers via cross-stich layer.  Accepts "affine_combo" or "dense" (Str)
        hard_burnin - Number of epochs to pre-train a hard mtl network of same architecture as sub-tast networks (Int)
        residual - Add skip connections (Bool)
        masked - Use custom loss function masked_mse to allow training the cross-stitch network on all examples
        
    returns:
        
    '''
    dat = data.copy()

        
    
    feats, shifts_list, shifts_means, shifts_stds = mtl_data_prep(dat, atoms, reject_outliers=reject_outliers, norm_shifts=norm_shifts, mode='testing')
    if masked:
        combined_feats, combined_shifts, combined_means, combined_std = mtl_data_prep(dat, atoms, reject_outliers=reject_outliers, norm_shifts=norm_shifts, mode='masked')
        loss_fn = masked_mse
    else:
        combined_feats, combined_shifts, combined_means, combined_std = mtl_data_prep(dat, atoms, reject_outliers=reject_outliers, norm_shifts=norm_shifts)
        loss_fn = 'mean_squared_error'
    
    if norm_shifts:
        for i in range(len(shifts_list)):
            shifts_list[i] = (shifts_list[i] - combined_means[i]) / combined_std[i]

    # Split up the data into train and validation
#    seed = np.random.randint(1, 100)
#    train_idxs, val_idxs = next(skl.model_selection.ShuffleSplit(n_splits=1, random_state=seed, test_size=0.2, train_size=None).split(feats))
#    feat_train, feat_val = feats[train_idxs], feats[val_idxs]
#    shifts_train_list = []
#    shifts_val_list = []
#    for i in range(len(atoms)):
#        shifts_train_list.append(shifts_list[0][train_idxs])
#        shifts_val_list.append(shifts_list[0][val_idxs])
    
    dim_in = combined_feats.shape[1]
    
    # If using noise, make data generators -- Not Implemented for MTL
    if noise:
        raise ValueError('Noise is not yet implemented for MTL so this kwarg must have value None')
#        train_dat, val_dat = skl.model_selection.train_test_split(data, test_size=0.2, random_state=seed)
#        train_gen = res_level_generator(train_dat, atom, noise=noise, noise_type=noise_type, noise_dist=noise_dist, batch_size=batch_size, norm_shifts=(shifts_mean, shifts_std))
#        full_gen = res_level_generator(train_dat, atom, noise=noise, noise_type=noise_type, noise_dist=noise_dist, batch_size=batch_size, norm_shifts=(shifts_mean, shifts_std))
#        n_train = len(train_dat) // batch_size # Number of full batches per epoch
#        rem_train = len(train_dat) % batch_size # Size of extra batch
#        n_full = len(dat) // batch_size
#        rem_full = len(dat) % batch_size
#        train_steps = min(rem_train, 1) + n_train
#        full_steps = min(rem_full, 1) + n_full

    # Set a boolean variable for presence of dropout layers
    if (float(do) > 0) and (not drop_last_only):
        dropout = True
    else:
        dropout = False

    opt = make_optimizer(opt_type, lrate[0], mom, dec, nest, clip_norm=clip_norm, clip_val=clip_val, opt_override=opt_override)
    
    regularizer = make_reg(reg, reg_type)
    
    # Build model
    # First build separate models for each atom
    clayer_dict = {}
    layer_dict = {}
    mod_dict = {}
    pre_train_layers = []
    for at_idx, atom in enumerate(atoms):
        layer_dict['input_layer_' + atom] = keras.layers.Input(shape=(dim_in,), name='input_layer_'+atom)
        layer_dict['interout_-1_' + atom] = keras.layers.Lambda(lambda x: x)(layer_dict['input_layer_' + atom])
        for idx, i in enumerate(arch):
            if activ is 'prelu':
                layer_dict['dense_' + str(idx) + '_' + atom] = keras.layers.Dense(units=i, activation='linear', kernel_regularizer=regularizer, name='dense_' + str(idx) + '_' + atom)(layer_dict['interout_' + str(idx-1) + '_' + atom])
                layer_dict['prelu_' + str(idx) + '_' + atom] = keras.layers.advanced_activations.PReLU(name='prelu_' + str(idx) + '_' + atom)(layer_dict['dense_' + str(idx) + '_' + atom])
                pre_train_layers += ['dense_' + str(idx) + '_' + atom]
                pre_train_layers += ['prelu_' + str(idx) + '_' + atom]
            else:
                layer_dict['dense_' + str(idx) + '_' + atom] = keras.layers.Dense(units=i, activation=activ, kernel_regularizer=regularizer, name='dense_' + str(idx) + '_' + atom)(layer_dict['interout_' + str(idx-1) + '_' + atom])
                pre_train_layers += ['dense_' + str(idx) + '_' + atom]
            if bnorm:
                if activ is 'prelu':
                    layer_dict['bn_' + str(idx) + '_' + atom] = keras.layers.BatchNormalization(name='bn_' + str(idx) + '_' + atom)(layer_dict['prelu_' + str(idx) + '_' + atom])
                else:
                    layer_dict['bn_' + str(idx) + '_' + atom] = keras.layers.BatchNormalization(name='bn_' + str(idx) + '_' + atom)(layer_dict['dense_' + str(idx) + '_' + atom])
                pre_train_layers += ['bn_' + str(idx) + '_' + atom]
            if dropout:
                if bnorm:
                    layer_dict['drop_' + str(idx) + '_' + atom] = keras.layers.Dropout(do)(layer_dict['bn_' + str(idx) + '_' + atom])
                else:
                    if activ is 'prelu':
                        layer_dict['drop_' + str(idx) + '_' + atom] = keras.layers.Dropout(do)(layer_dict['prelu_' + str(idx) + '_' + atom])
                    else:
                        layer_dict['drop_' + str(idx) + '_' + atom] = keras.layers.Dropout(do)(layer_dict['dense_' + str(idx) + '_' + atom])
                layer_dict['interout_' + str(idx) + '_' + atom] = keras.layers.Lambda(lambda x: x)(layer_dict['drop_' + str(idx) + '_' + atom])
            else:
                if bnorm:
                    layer_dict['interout_' + str(idx) + '_' + atom] = keras.layers.Lambda(lambda x: x)(layer_dict['bn_' + str(idx) + '_' + atom])
                else:
                    if activ is 'prelu':
                        layer_dict['interout_' + str(idx) + '_' + atom] = keras.layers.Lambda(lambda x: x)(layer_dict['prelu_' + str(idx) + '_' + atom])
                    else:
                        layer_dict['interout_' + str(idx) + '_' + atom] = keras.layers.Lambda(lambda x: x)(layer_dict['dense_' + str(idx) + '_' + atom])
            if residual and (idx > 0 or i==dim_in):
                layer_dict['interout_' + str(idx) + '_' + atom] = keras.layers.Add()([layer_dict['interout_' + str(idx) + '_' + atom], layer_dict['interout_' + str(idx - 1) + '_' + atom]])
#        if drop_last_only:
#            mod.add(keras.layers.Dropout(do))
        layer_dict['output_layer_' + atom] = keras.layers.Dense(units=1, activation='linear', name='output_layer_' + atom)(layer_dict['interout_' + str(len(arch)-1) + '_' + atom])
        pre_train_layers += ['output_layer_' + atom]
                
        mod_dict['model_' + atom] = keras.models.Model(inputs=layer_dict['input_layer_' + atom], outputs=layer_dict['output_layer_' + atom])
        mod_dict['model_' + atom].compile(loss='mean_squared_error', optimizer=opt)


    # Now build the cross stitch model that combines sub-models
    clayer_dict['combined_input_layer'] = keras.layers.Input(shape=(dim_in,), name='combined_input_layer')
    for atom in atoms:
        clayer_dict['combined_input_-1' + '_' + atom] = keras.layers.Lambda(lambda x: x, name='combined_input_-1_'+atom)(clayer_dict['combined_input_layer'])
    for idx, i in enumerate(arch):
        for at_idx, atom in enumerate(atoms):
            if activ is 'prelu':
                clayer_dict['combined_dense_' + str(idx) + '_' + atom] = keras.layers.Dense(units=i, activation='linear', kernel_regularizer=regularizer, name='combined_dense_' + str(idx) + '_' + atom)(clayer_dict['combined_input_' + str(idx-1) + '_' + atom])
                clayer_dict['combined_prelu_' + str(idx) + '_' + atom] = keras.layers.advanced_activations.PReLU(name='combined_prelu_' + str(idx) + '_' + atom)(clayer_dict['combined_dense_' + str(idx) + '_' + atom])
            else:
                clayer_dict['combined_dense_' + str(idx) + '_' + atom] = keras.layers.Dense(units=i, activation=activ, kernel_regularizer=regularizer, name='combined_dense_' + str(idx) + '_' + atom)(clayer_dict['combined_input_' + str(idx-1) + '_' + atom])
            if bnorm:
                if activ is 'prelu':
                    clayer_dict['combined_bn_' + str(idx) + '_' + atom] = keras.layers.BatchNormalization(name='combined_bn_' + str(idx) + '_' + atom)(clayer_dict['combined_prelu_' + str(idx) + '_' + atom])
                else:
                    clayer_dict['combined_bn_' + str(idx) + '_' + atom] = keras.layers.BatchNormalization(name='combined_bn_' + str(idx) + '_' + atom)(clayer_dict['combined_dense_' + str(idx) + '_' + atom])
            if dropout:
                if bnorm:
                    clayer_dict['combined_drop_' + str(idx) + '_' + atom] = keras.layers.Dropout(do, name='combined_drop_' + str(idx) + '_' + atom)(clayer_dict['combined_bn_' + str(idx) + '_' + atom])
                else:
                    if activ is 'prelu':
                        clayer_dict['combined_drop_' + str(idx) + '_' + atom] = keras.layers.Dropout(do, name='combined_drop_' + str(idx) + '_' + atom)(clayer_dict['combined_prelu_' + str(idx) + '_' + atom])
                    else:
                        clayer_dict['combined_drop_' + str(idx) + '_' + atom] = keras.layers.Dropout(do, name='combined_drop_' + str(idx) + '_' + atom)(clayer_dict['combined_dense_' + str(idx) + '_' + atom])
                clayer_dict['combined_interout_' + str(idx) + '_' + atom] = keras.layers.Lambda(lambda x: x, name='combined_interout_' + str(idx) + '_' + atom)(clayer_dict['combined_drop_' + str(idx) + '_' + atom])
                        
            # Define an intermediate output layer for this index and for each atom
#            if dropout:
#                clayer_dict['combined_interout_' + str(idx) + '_' atom] = keras.layers.Dense(units=1, activation='linear', name='combined_interout_' + str(idx) + '_' atom)(clayer_dict['combined_drop_' + str(idx) + '_' + atom])
            else:
                if bnorm:
                    clayer_dict['combined_interout_' + str(idx) + '_' + atom] = keras.layers.Dense(units=1, activation='linear', name='combined_interout_' + str(idx) + '_' + atom)(clayer_dict['combined_bn_' + str(idx) + '_' + atom])
                else:
                    if activ is 'prelu':
                        clayer_dict['combined_interout_' + str(idx) + '_' + atom] = keras.layers.Dense(units=1, activation='linear', name='combined_interout_' + str(idx) + '_' + atom)(clayer_dict['combined_prelu_' + str(idx) + '_' + atom])
                    else:
                        clayer_dict['combined_interout_' + str(idx) + '_' + atom] = keras.layers.Dense(units=1, activation='linear', name='combined_interout_' + str(idx) + '_' + atom)(clayer_dict['combined_dense_' + str(idx) + '_' + atom])
            if residual == 1 and (idx > 0 or i==dim_in):
                clayer_dict['combined_interout_' + str(idx) + '_' + atom] = keras.layers.Add()([clayer_dict['combined_interout_' + str(idx) + '_' + atom], clayer_dict['combined_input_' + str(idx-1) + '_' + atom]])
        
        if cs_mode == 'dense':
        # Concatenate the intermediate outputs for cross-stitching
            clayer_dict['concat_' + str(idx)] = keras.layers.concatenate([clayer_dict['combined_interout_' + str(idx) + '_' + a] for a in atoms], name='concat_' + str(idx))
            concat_size = i * len(atoms)
            clayer_dict['cross_stitch_' + str(idx)] = keras.layers.Dense(units=concat_size, kernel_initializer='identity', use_bias=False, name='cross_stitch_' + str(idx))(clayer_dict['concat_' + str(idx)])
            # Split the resulting output from the cross_stitch
            for at_idx, atom in enumerate(atoms):
                clayer_dict['combined_input_' + str(idx) + '_' + atom] = keras.layers.Lambda(lambda x: x[:, at_idx*i : i*(at_idx + 1)])(clayer_dict['cross_stitch_' + str(idx)])
            
        # Can instead try restricted cross stitch that has far fewer parameters:
#       clayer_dict['same_stitch_' + idx] = keras.layersDense(units=1, kernel_initializer='identity') 
#        for at_idx, atom in enumerate(atoms):
#            for at_idx2 in range(at_idx+1, len(atoms)):
        if cs_mode == 'affine_combo':
            for atom in atoms:
                ordered_names = ['combined_interout_' + str(idx) + '_' + atom]
                ordered_names += ['combined_interout_' + str(idx) + '_' + at for at in atoms if at is not atom]
                clayer_dict['cross_stitch_' + str(idx) + '_' + atom] = ConvexCombination(name='cross_stitch_' + str(idx) + '_' + atom)([clayer_dict[o_name] for o_name in ordered_names])
                clayer_dict['combined_input_' + str(idx) + '_' + atom] = keras.layers.Lambda(lambda x: x)(clayer_dict['cross_stitch_' + str(idx) + '_' + atom])
#        ordered_names = ['combined_interout_' + str(idx) + '_' + atom for atom in atoms]
#        [clayer_dict['cross_stitch_' + str(idx) + '_' + atom] for atom in atom_names] = ConvexCombination()([clayer_dict[o_name] for o_name in ordered_names])
                
        if residual == 2 and (idx > 0 or i==dim_in):
            clayer_dict['combined_input_' + str(idx) + '_' + atom] = keras.layers.Add()([clayer_dict['combined_input_' + str(idx) + '_' + atom], clayer_dict['combined_input_' + str(idx-1) + '_' + atom]])
        

        
    for atom in atoms:
        clayer_dict['combined_output_layer_' + atom] = keras.layers.Dense(units=1, activation='linear', name='combined_output_layer_' + atom)(clayer_dict['combined_input_' + str(len(arch)-1) + '_' + atom])
        #clayer_dict['combined_output_layer_' + atom].set_weights(clayer_dict['output_layer_' + atom].get_weights())
                    
                    
#        if dropout:
#            layer_dict['combined_output_layer_' + atom] = keras.layers.Dense(units=1, activation='linear', name='combined_output_layer_' + atom)(layer_dict['combined_drop_' + str(len(arch)-1) + '_' + atom])
#        else:
#            if bnorm:
#                layer_dict['combined_output_layer_' + atom] = keras.layers.Dense(units=1, activation='linear', name='combined_output_layer_' + atom)(layer_dict['combined_bn_' + str(len(arch)-1) + '_' + atom])
#            else:
#                if activ is 'prelu':
#                    layer_dict['combined_output_layer_' + atom] = keras.layers.Dense(units=1, activation='linear', name='combined_output_layer_' + atom)(layer_dict['combined_prelu_' + str(len(arch)-1) + '_' + atom])
#                else:
#                    layer_dict['combined_output_layer_' + atom] = keras.layers.Dense(units=1, activation='linear', name='combined_output_layer_' + atom)(layer_dict['combined_dense_' + str(len(arch)-1) + '_' + atom])
#        
#        layer_dict['combined_output_layer_' + atom].set_weights(layer_dict['output_layer_' + atom].get_weights())
#    
    mod_dict['combined_mod'] = keras.models.Model(inputs=clayer_dict['combined_input_layer'], outputs=[clayer_dict['combined_output_layer_' + atom] for atom in atoms])
    
    # Need to make an optimizer with layer-dependent learning-rates so cross stitch layers learn faster than the pre-trained layers
    cs_names = ['cross_stitch_' + str(idx) + '_' + atom for idx in range(len(arch)) for atom in atoms]
    mults = {}
    for name in clayer_dict.keys():
        mults[name] = 1 / cs_mult
    for name in cs_names:
        mults[name] = 1
    opt2 = Adam_lr_mult(lr=lrate[1], multipliers=mults, **opt2_kwargs)
    
    mod_dict['combined_mod'].compile(loss=loss_fn, optimizer=opt2)
    
    if hard_burnin:
        # Make hard multi-task network for burn-in
        hlayer_dict = {}
        hlayer_dict['input_layer_' + 'hmtl'] = keras.layers.Input(shape=(dim_in,), name='input_layer_'+'hmtl')
        hlayer_dict['interout_-1_' + 'hmtl'] = keras.layers.Lambda(lambda x: x)(hlayer_dict['input_layer_' + 'hmtl'])
        for idx, i in enumerate(arch[:-1]):
            if activ is 'prelu':
                hlayer_dict['dense_' + str(idx) + '_' + 'hmtl'] = keras.layers.Dense(units=i, activation='linear', kernel_regularizer=regularizer, name='dense_' + str(idx) + '_' + 'hmtl')(hlayer_dict['interout_' + str(idx-1) + '_' + 'hmtl'])
                hlayer_dict['prelu_' + str(idx) + '_' + 'hmtl'] = keras.layers.advanced_activations.PReLU(name='prelu_' + str(idx) + '_' + 'hmtl')(hlayer_dict['dense_' + str(idx) + '_' + 'hmtl'])
            else:
                hlayer_dict['dense_' + str(idx) + '_' + 'hmtl'] = keras.layers.Dense(units=i, activation=activ, kernel_regularizer=regularizer, name='dense_' + str(idx) + '_' + 'hmtl')(hlayer_dict['interout_' + str(idx-1) + '_' + 'hmtl'])
            if bnorm:
                if activ is 'prelu':
                    hlayer_dict['bn_' + str(idx) + '_' + 'hmtl'] = keras.layers.BatchNormalization(name='bn_' + str(idx) + '_' + 'hmtl')(hlayer_dict['prelu_' + str(idx) + '_' + 'hmtl'])
                else:
                    hlayer_dict['bn_' + str(idx) + '_' + 'hmtl'] = keras.layers.BatchNormalization(name='bn_' + str(idx) + '_' + 'hmtl')(hlayer_dict['dense_' + str(idx) + '_' + 'hmtl'])
            if dropout:
                if bnorm:
                    hlayer_dict['drop_' + str(idx) + '_' + 'hmtl'] = keras.layers.Dropout(do)(hlayer_dict['bn_' + str(idx) + '_' + 'hmtl'])
                else:
                    if activ is 'prelu':
                        hlayer_dict['drop_' + str(idx) + '_' + 'hmtl'] = keras.layers.Dropout(do)(hlayer_dict['prelu_' + str(idx) + '_' + 'hmtl'])
                    else:
                        hlayer_dict['drop_' + str(idx) + '_' + 'hmtl'] = keras.layers.Dropout(do)(hlayer_dict['dense_' + str(idx) + '_' + 'hmtl'])
                hlayer_dict['interout_' + str(idx) + '_' + 'hmtl'] = keras.layers.Lambda(lambda x: x)(hlayer_dict['drop_' + str(idx) + '_' + 'hmtl'])
            else:
                if bnorm:
                    hlayer_dict['interout_' + str(idx) + '_' + 'hmtl'] = keras.layers.Lambda(lambda x: x)(hlayer_dict['bn_' + str(idx) + '_' + 'hmtl'])
                else:
                    if activ is 'prelu':
                        hlayer_dict['interout_' + str(idx) + '_' + 'hmtl'] = keras.layers.Lambda(lambda x: x)(hlayer_dict['prelu_' + str(idx) + '_' + 'hmtl'])
                    else:
                        hlayer_dict['interout_' + str(idx) + '_' + 'hmtl'] = keras.layers.Lambda(lambda x: x)(hlayer_dict['dense_' + str(idx) + '_' + 'hmtl'])
            if residual and (idx > 0 or i==dim_in):
                hlayer_dict['interout_' + str(idx) + '_' + 'hmtl'] = keras.layers.Add()([hlayer_dict['interout_' + str(idx) + '_' + 'hmtl'], hlayer_dict['interout_' + str(idx - 1) + '_' + 'hmtl']])
        # Use the last layer as an atom-dependent cap
        last_idx = len(arch) - 1
        for atom in atoms:
            if activ is 'prelu':
                hlayer_dict['dense_' + str(last_idx) + '_' + 'hmtl' + '_' + atom] = keras.layers.Dense(units=arch[last_idx], activation='linear', kernel_regularizer=regularizer, name='dense_' + str(last_idx) + '_' + 'hmtl' + '_' + atom)(hlayer_dict['interout_' + str(last_idx-1) + '_' + 'hmtl'])
                hlayer_dict['prelu_' + str(last_idx) + '_' + 'hmtl' + '_' + atom] = keras.layers.advanced_activations.PReLU(name='prelu_' + str(last_idx) + '_' + 'hmtl' + '_' + atom)(hlayer_dict['dense_' + str(last_idx) + '_' + 'hmtl' + '_' + atom])
            else:
                hlayer_dict['dense_' + str(last_idx) + '_' + 'hmtl' + '_' + atom] = keras.layers.Dense(units=arch[last_idx], activation=activ, kernel_regularizer=regularizer, name='dense_' + str(last_idx) + '_' + 'hmtl' + '_' + atom)(hlayer_dict['interout_' + str(last_idx-1) + '_' + 'hmtl'])
            if bnorm:
                if activ is 'prelu':
                    hlayer_dict['bn_' + str(last_idx) + '_' + 'hmtl' + '_' + atom] = keras.layers.BatchNormalization(name='bn_' + str(last_idx) + '_' + 'hmtl' + '_' + atom)(hlayer_dict['prelu_' + str(last_idx) + '_' + 'hmtl' + '_' + atom])
                else:
                    hlayer_dict['bn_' + str(last_idx) + '_' + 'hmtl' + '_' + atom] = keras.layers.BatchNormalization(name='bn_' + str(last_idx) + '_' + 'hmtl' + '_' + atom)(hlayer_dict['dense_' + str(last_idx) + '_' + 'hmtl' + '_' + atom])
            if dropout:
                if bnorm:
                    hlayer_dict['drop_' + str(last_idx) + '_' + 'hmtl' + '_' + atom] = keras.layers.Dropout(do)(hlayer_dict['bn_' + str(last_idx) + '_' + 'hmtl' + '_' + atom])
                else:
                    if activ is 'prelu':
                        hlayer_dict['drop_' + str(last_idx) + '_' + 'hmtl' + '_' + atom] = keras.layers.Dropout(do)(hlayer_dict['prelu_' + str(last_idx) + '_' + 'hmtl' + '_' + atom])
                    else:
                        hlayer_dict['drop_' + str(last_idx) + '_' + 'hmtl' + '_' + atom] = keras.layers.Dropout(do)(hlayer_dict['dense_' + str(last_idx) + '_' + 'hmtl' + '_' + atom])
                hlayer_dict['interout_' + str(last_idx) + '_' + 'hmtl' + '_' + atom] = keras.layers.Lambda(lambda x: x)(hlayer_dict['drop_' + str(last_idx) + '_' + 'hmtl' + '_' + atom])
            else:
                if bnorm:
                    hlayer_dict['interout_' + str(last_idx) + '_' + 'hmtl' + '_' + atom] = keras.layers.Lambda(lambda x: x)(hlayer_dict['bn_' + str(last_idx) + '_' + 'hmtl' + '_' + atom])
                else:
                    if activ is 'prelu':
                        hlayer_dict['interout_' + str(last_idx) + '_' + 'hmtl' + '_' + atom] = keras.layers.Lambda(lambda x: x)(hlayer_dict['prelu_' + str(last_idx) + '_' + 'hmtl' + '_' + atom])
                    else:
                        hlayer_dict['interout_' + str(last_idx) + '_' + 'hmtl' + '_' + atom] = keras.layers.Lambda(lambda x: x)(hlayer_dict['dense_' + str(last_idx) + '_' + 'hmtl' + '_' + atom])
            hlayer_dict['output_layer_' + 'hmtl' + '_' + atom] = keras.layers.Dense(units=1, activation='linear', name='output_layer_' + 'hmtl' + '_' + atom)(hlayer_dict['interout_' + str(last_idx) + '_' + 'hmtl' + '_' + atom])
        # Compile and train hard multi-task network
        mod_dict['model_' + 'hmtl'] = keras.models.Model(inputs=hlayer_dict['input_layer_' + 'hmtl'], outputs=[hlayer_dict['output_layer_' + 'hmtl' + '_' + at] for at in atoms])
        mod_dict['model_' + 'hmtl'].compile(loss=loss_fn, optimizer=opt)
        mod_dict['model_' + 'hmtl'].fit(combined_feats, combined_shifts, batch_size=batch_size, epochs=hard_burnin)
        
        # Use hard MTL network weights to initialize single-task sub networks
        
        for name in pre_train_layers:
            atom_id = name.split('_')[-1]
            idx_id = name.split('_')[-2]
            if idx_id in [str(i) for i in range(last_idx)]:
                hlname = name.split('_')[0] + '_' + idx_id + '_hmtl'
                mod_dict['model_' + atom_id].get_layer(name).set_weights(mod_dict['model_hmtl'].get_layer(hlname).get_weights())
            else:
                for atom in atoms:
                    hlname = name.split('_')[0] + '_' + idx_id + '_hmtl_' + atom
                    mod_dict['model_' + atom_id].get_layer(name).set_weights(mod_dict['model_hmtl'].get_layer(hlname).get_weights())
                
    else:
        pass
    
    # pre-train the sub-networks
    for at_idx, atom in enumerate(atoms):
        these_feats = feats[at_idx]
        these_shifts = shifts_list[at_idx][:, at_idx]
        mod_dict['model_' + atom].fit(these_feats, these_shifts, batch_size=batch_size, epochs=epochs[at_idx])
    # re-initialize weights of cross stitch network to pre-trained values
    for name in pre_train_layers:
        mod_dict['combined_mod'].get_layer('combined_' + name).set_weights(mod_dict['model_' + name.split('_')[-1]].get_layer(name).get_weights())
    
    # Freeze BN layers.  Not sure if this will help or not
    for layer in mod_dict['combined_mod'].layers:
        if isinstance(layer, keras.layers.normalization.BatchNormalization):
            layer._per_input_updates = {}

    mod_dict['combined_mod'].fit(combined_feats, combined_shifts, batch_size=batch_size, epochs=epochs[-1])
    
    
       
    # Get initial weights to reset model after pretraining
#    weights = mod.get_weights()

    # Do early-stopping routine if desired 
    if early_stop is not None:
        raise ValueError('Early-Stopping not yet implemented so value for this kwarg must be None')
        if noise is not None:
            raise ValueError('Noise not implemented so value for this kwarg must be None')
#            val_epochs, hist_list, val_list, param_list = early_stopping(mod, early_stop, tol, per, epochs, min_epochs, batch_size, feat_train=None, shift_train=None, feat_val=feat_val, shift_val=shift_val, train_gen=train_gen, train_steps=train_steps, noise=noise)
        else:
            val_epochs, hist_list, val_list, param_list = early_stopping(mod_dict['combined_mod'], early_stop, tol, per, epochs, min_epochs, batch_size, feat_train=feat_train, shift_train=shifts_train_list, feat_val=feat_val, shift_val=shifts_val_list, train_gen=None, train_steps=None, noise=noise, mtl=True)
    else:
        hist_list = []
        val_list = []
        param_list = []
        val_epochs = epochs

    # Retrain model on full dataset for same number of epochs as was
    # needed to obtain min validation loss
#    mod.set_weights(weights)
#    if noise:
#        pass
#        mod.fit_generator(full_gen, steps_per_epoch=full_steps, epochs=val_epochs)
#    else:
#        #mod.fit(feats, shift_norm, batch_size=batch_size, epochs=val_epochs)
#        mod.fit(feats, shifts_list, batch_size=batch_size, epochs=val_epochs)

    return shifts_means, shifts_stds, val_list, hist_list, param_list, mod_dict


def strucseq_branch_model(data, atom, activ, arch, lrate, mom, dec, epochs, min_epochs=0, per=5, tol=1.0, do=[0.0, 0.0, 0.0], reg=[0.0, 0.0, 0.0], seq_type='blosum', merge_mode='concat',
                          pretrain=None, bnorm=False, lsuv=False, nest=False, norm_shifts=True, reject_outliers=None):
    '''Constructs a model from the given features and shifts that seperately processes structure 
    and sequence information.  The arch argument is a list specifying the number of hidden units
    at each layer.  The architecture for this network is split into two branches that handle
    structural and sequential information respectively rather than being fully connected.  The 
    arch parameter is thus given as a list, the first two elements of which are themselves lists
    giving the neurons for each layer of each branch and the remaining elements giving the neurons
    for the remainder of the network, after the two branches meet. 
    
    data = Feature and target data (Pandas DataFrame)
    atom = Name of atom to predict (Str)
    activ = Activation function for dense layers - accepts "prelu" (Str)
    arch = List of the number of neurons per layer (List)
    lrate = Learning rate for SGD (Float)
    mom = Momentum for SGD (Float)
    dec = Decay for SGD learning rate (Float)
    max_epochs = Maximum number of epochs to train (Int)
    min_epochs = Minimum number of epochs to train (Int)
    per = Number of epochs in a test strip for pretraining (Int)
    tol = Pretraining parameter (Float)
    do = List of dropout percentages between 0 and 1.  1st, 2nd, and 3rd elements used for
          structure branch, sequence branch, and combined processing respectively (List of Floats)
    reg = List of parameters for L1 regularization of dense layers.  1st, 2nd, and 3rd elements used
          for structure branch, sequence branch, and combined processing respectively (List of Floats)
    seq_type = How sequence information is encoded.  Accepts "blosum" or "binary" (Str)
    pretrain = Whether or not and how to do pretraining.  Accepts None, GL, PQ, or UP (Str)
    bnorm = Use batch normalization (Bool)
    lsuv = Use layer-wise sequential unit variance initialization (Bool)
    nest = Use Nesterov momentum (Bool)
    rings = DataFrame contains ring-current columns (Bool)
    rcoil = DataFrame contains random-coil columns (Bool)
    norm_shifts = Normalize the target shifts rather than using raw values (Bool)
    reject_outliers = Eliminate entries for which the targets differ by more than this number of standard deviations (Float or None)
    '''
    
    dat = data[data[atom].notnull()]
    try:
        dat = dat.drop(['Unnamed: 0', 'FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM', 'CHAIN'], axis=1)
    except ValueError:
        pass
    try:
        dat = dat.drop(['RESNAME_ip1', 'RESNAME_im1'], axis=1)
    except ValueError:
        pass

    if reject_outliers is not None:
        n_sig = max(reject_outliers, 0.1) # Don't accept a cutoff less than 0.1 * std
        rej_mean = dat[atom].mean()
        rej_std = dat[atom].std()
        up_rej = rej_mean + n_sig * rej_std
        low_rej = rej_mean - n_sig * rej_std
        dat = dat[(dat[atom] > low_rej) & (dat[atom] < up_rej)]

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
    try:
        ring_col = atom + '_RC'
        rem1 = ring_cols.copy()
        rem1.remove(ring_col)
        feats = feats.drop(rem1, axis=1)
        feats[ring_col] = feats[ring_col].fillna(value=0)
        rings = True
    except KeyError:
        rings = False
    except ValueError:
        pass
    try:
        rcoil_col = 'RCOIL_' + atom
        rem2 = rcoil_cols.copy()
        rem2.remove(rcoil_col)
        feats = feats.drop(rem2, axis=1)
        rcoil = True
    except ValueError:
        rcoil = False

    # Separate features corresponding to structure and sequence information    
    structure_cols = struc_cols.copy()
    if seq_type == 'binary':
        sequence_cols = bin_seq_cols.copy()
    elif seq_type == 'blosum':
        sequence_cols = seq_cols.copy()
    if rings:
        structure_cols += [ring_col]
    if rcoil:
        structure_cols += [rcoil_col]
    struc_feats, seq_feats = feats[structure_cols].values, dat[sequence_cols].values
    feat_train, feat_val, shift_train, shift_val = skl.model_selection.train_test_split(feats, shift_norm, test_size=0.2)
    struc_feat_train, struc_feat_val = feat_train[structure_cols].values, feat_val[structure_cols].values
    seq_feat_train, seq_feat_val = feat_train[sequence_cols].values, feat_val[sequence_cols].values
#    shift_train, shift_val = shift_train.values, shift_val.values
    struc_dim_in = struc_feat_train.shape[1]
    seq_dim_in = seq_feat_train.shape[1]

    
    if max(do) > 0:
        dropout = True
    else:
        dropout = False

    # Define optimization procedure
    opt = keras.optimizers.SGD(lr=lrate, momentum=mom, decay=dec, nesterov=nest)
    
    # Build model
    # Process structure and sequence information seperately
    struc_in = keras.layers.Input(shape=(struc_dim_in,), name='struc_input')
    struc = keras.layers.Lambda(lambda x: x)(struc_in)
    seq_in = keras.layers.Input(shape=(seq_dim_in,), name='seq_input')
    seq = keras.layers.Lambda(lambda x: x)(seq_in)
    for i in arch[0]:
        if activ == 'prelu':
            struc = keras.layers.Dense(i, activation='linear', kernel_regularizer=keras.regularizers.l1(reg[0]))(struc)
            struc = keras.layers.advanced_activations.PReLU()(struc)
        else:
            struc = keras.layers.Dense(i, activation=activ, kernel_regularizer=keras.regularizers.l1(reg[0]))(struc)
        if bnorm:
            struc = keras.layers.BatchNormalization()(struc)
        if dropout:
            struc = keras.layers.Dropout(do[0])(struc)
    for i in arch[1]:
        if activ == 'prelu':
            seq = keras.layers.Dense(i, activation='linear', kernel_regularizer=keras.regularizers.l1(reg[1]))(seq)
            seq = keras.layers.advanced_activations.PReLU()(seq)
        else:
            seq = keras.layers.Dense(i, activation=activ, kernel_regularizer=keras.regularizers.l1(reg[1]))(seq)
        if bnorm:
            seq = keras.layers.BatchNormalization()(seq)
        if dropout:
            seq = keras.layers.Dropout(do[1])(seq)

    if merge_mode == 'concat':
        merge = keras.layers.concatenate([struc, seq])
    elif merge_mode == 'sum':
        merge = keras.layers.Add()([struc, seq])
    
    # Process merged info
    for i in arch[2:]:
        if activ == 'prelu':
            merge = keras.layers.Dense(i, activation='linear', kernel_regularizer=keras.regularizers.l1(reg[2]))(merge)
            merge = keras.layers.advanced_activations.PReLU()(merge)
        else:
            merge = keras.layers.Dense(i, activation=activ, kernel_regularizer=keras.regularizers.l1(reg[2]))(merge)
        if bnorm:
            merge = keras.layers.BatchNormalization()(merge)
        if dropout:
            merge = keras.layers.Dropout(do[2])(merge)
    
    # Define output
    predicted_shift = keras.layers.Dense(units=1, activation='linear')(merge)
    
    # Define model and compile
    mod = keras.models.Model(inputs=[struc_in, seq_in], outputs=predicted_shift)
    mod.compile(loss='mean_squared_error', optimizer=opt)

    if lsuv:
        mod = LSUVinit(mod, feat_train[:64])
    
    weights = mod.get_weights()
    
    # Initialize some outputs
    hist_list = []
    val_list = []
    param_list = []

    # Do pretraining to determine the best number of epochs for training
    val_min = 10 ** 10
    up_count = 0
    if pretrain is not None:
        for i in range(int(epochs/per)):
            pt1 = mod.evaluate([struc_feat_val, seq_feat_val], shift_val, verbose=0)
            val_list.append(pt1)
    
            if pt1 < val_min:
                val_min = pt1
    
            hist = mod.fit([struc_feat_train, seq_feat_train], shift_train, batch_size=64, epochs=per)
            hist_list += hist.history['loss']
            pt2 = mod.evaluate([struc_feat_val, seq_feat_val], shift_val, verbose=0)
            
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
    
    
    # Retrain model on full dataset for same number of epochs as was
    # needed to obtain min validation loss
    
    mod.set_weights(weights)

    mod.fit([struc_feats, seq_feats], shift_norm, batch_size=64, epochs=val_epochs)

    if reject_outliers is not None:
        return rej_mean, rej_std, val_list, hist_list, param_list, mod
    else:
        return shifts_mean, shifts_std, val_list, hist_list, param_list, mod



def residual_model(data, atom, arch, activ='prelu', lrate=0.001, mom=0, dec=10**-6, epochs=100, min_epochs=5, per=5, tol=1.0, opt_type='sgd', do=0.0, drop_last_only=False, reg=0.0, reg_type=None,
                   pretrain=None, bnorm=False, lsuv=False, nest=False, norm_shifts=True, opt_override=False, clip_val=0, clip_norm=0, skip_connection='residual', reject_outliers=None, batch_size=64, lsuv_batch=64, noise=None, noise_type='angle', noise_dist='uniform'):
    '''Constructs a residual neural network.  Essentially, a fully-connected network wherein all layers are the same width and that is
    endowed with "skip connections" that directly connect the output at various layers in the network (depending on skip_connection kwarg)
    that allow intervening layers to be bypassed during backpropagation.  Effectively allows the net to "unravel" as the training first
    learns representations from shallower sub-networks before more dramatically adjusting weights/biases of later layers.  Also can be 
    thought of as an ensemble of networks of various lengths.
    
    data = Feature and target data (Pandas DataFrame)
    atom = Name of atom to predict (Str)
    activ = Activation function for dense layers - accepts "prelu" (Str)
    arch = List [D,W] specifying the depth (number of layers) and width (neurons per layer) for the network (List)
    lrate = Learning rate for SGD (Float)
    mom = Momentum for SGD (Float)
    dec = Decay for SGD learning rate (Float)
    max_epochs = Maximum number of epochs to train (Int)
    min_epochs = Minimum number of epochs to train (Int)
    per = Number of epochs in a test strip for pretraining (Int)
    tol = Pretraining parameter (Float)
    do = Dropout percentage between 0 and 1 (Float)
    reg = Parameter for weight regularization (Float)
    pretrain = Whether or not and how to do pretraining.  Accepts None, GL, PQ, or UP (Str)
    bnorm = Use batch normalization (Bool)
    lsuv = Use layer-wise sequential unit variance initialization (Bool)
    nest = Use Nesterov momentum (Bool)
    rings = DataFrame contains ring-current columns (Bool)
    rcoil = DataFrame contains random-coil columns (Bool)
    norm_shifts = Normalize the target shifts rather than using raw values (Bool)
    opt_override = Override default optimization parameters (Bool)
    clip_val = Clip values of gradients in optimizer (Float)
    clip_norm = Clip norms of gradients in optimizer (Float)
    skip_connections = Mode of skip connection - accepts "residual", "coupled_highway", or "full_highway" (Str)
    reject_outliers = Eliminate entries for which the targets differ by more than this number of standard deviations (Float or None)
    '''
    
    dat = data[data[atom].notnull()]
    for col in cols_to_drop:
        try:
            dat = dat.drop(col, axis=1)
        except ValueError:
            pass

    if reject_outliers is not None:
        n_sig = max(reject_outliers, 0.1) # Don't accept a cutoff less than 0.1 * std
        rej_mean = dat[atom].mean()
        rej_std = dat[atom].std()
        up_rej = rej_mean + n_sig * rej_std
        low_rej = rej_mean - n_sig * rej_std
        dat = dat[(dat[atom] > low_rej) & (dat[atom] < up_rej)]

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
    try:
        ring_col = atom + '_RC'
        rem1 = ring_cols.copy()
        rem1.remove(ring_col)
        feats = feats.drop(rem1, axis=1)
        feats[ring_col] = feats[ring_col].fillna(value=0)
    except KeyError:
        pass
    except ValueError:
        pass
    try:
        rcoil_col = 'RCOIL_' + atom
        rem2 = rcoil_cols.copy()
        rem2.remove(rcoil_col)
        feats = feats.drop(rem2, axis=1)
    except ValueError:
        pass
    
    feats = feats.values
    feat_train, feat_val, shift_train, shift_val = skl.model_selection.train_test_split(feats, shift_norm, test_size=0.2)
    dim_in = feats.shape[1]
    
    if do > 0 and not drop_last_only:
        dropout = True
    else:
        dropout = False

    # Define optimization procedure
    if opt_type is 'sgd':
        opt = keras.optimizers.SGD(lr=lrate, momentum=mom, decay=dec, nesterov=nest, clipnorm=clip_norm, clipvalue=clip_val)
    elif opt_type is 'rmsprop':
        if opt_override:
            opt = keras.optimizers.RMSprop(lr=lrate, rho=mom, decay=dec, clipnorm=clip_norm, clipvalue=clip_val)
        else:
            opt = keras.optimizers.RMSprop(clipnorm=clip_norm, clipvalue=clip_val)
    elif opt_type is 'adagrad':
        if opt_override:
            opt = keras.optimizers.Adagrad(lr=lrate, decay=dec, clipnorm=clip_norm, clipvalue=clip_val)
        else:
            opt = keras.optimizers.Adagrad(clipnorm=clip_norm, clipvalue=clip_val)
    elif opt_type is 'adadelta':
        if opt_override:
            opt = keras.optimizers.Adadelta(lr=lrate, rho=mom, decay=dec, clipnorm=clip_norm, clipvalue=clip_val)
        else:
            opt = keras.optimizers.Adadelta(clipnorm=clip_norm, clipvalue=clip_val)
    elif opt_type is 'adam':
        if opt_override:
            try:
                beta1 = mom[0]
                beta2 = mom[1]
            except TypeError:
                beta1 = mom
                beta2 = mom
                print('Only one momentum given for adam-type optimizer.  Using this value for both beta1 and beta2')
            if nest:
                opt = keras.optimizers.Nadam(lr=lrate, beta_1=beta1, beta_2=beta2, schedule_decay=dec, clipnorm=clip_norm, clipvalue=clip_val)
            else:
                opt = keras.optimizers.Adam(lr=lrate, beta_1=beta1, beta_2=beta2, decay=dec, clipnorm=clip_norm, clipvalue=clip_val)
        else:
            if nest:
                opt = keras.optimizers.Nadam(clipnorm=clip_norm, clipvalue=clip_val)
            else:
                opt = keras.optimizers.Adam(clipnorm=clip_norm, clipvalue=clip_val)
    elif opt_type is 'adamax':
        if opt_override:
            try:
                beta1 = mom[0]
                beta2 = mom[1]
            except TypeError:
                beta1 = mom
                beta2 = mom
                print('Only one momentum given for adam-type optimizer.  Using this value for both beta1 and beta2')
            opt = keras.optimizers.Adamax(lr=lrate, beta_1=mom, beta_2=mom, decay=dec, clipnorm=clip_norm, clipvalue=clip_val)
        else:
            opt = keras.optimizers.Adamax(clipnorm=clip_norm, clipvalue=clip_val)
    
    # Define weight regularizers
    if reg_type is None:
        regularizer = keras.regularizers.l2(0)
    if reg_type == 'L1':
        regularizer = keras.regularizers.l1(reg)
    elif reg_type == 'L2':
        regularizer = keras.regularizers.l2(reg)
    elif reg_type == 'L1_L2':
        try:
            l1reg = reg[0]
            l2reg = reg[1]
        except TypeError:
            l1reg = reg
            l2reg = reg
            print('reg_type is L1_L2 but reg was not passed a list.  Using reg for both L1 and L2')
        regularizer = keras.regularizers.l1_l2(l1=l1reg, l2=l2reg)
    
    # Build model
    # Define depth and width
    depth = arch[1][0]
    width = arch[1][1]
    # Define input
    feat_in = keras.layers.Input(shape=(dim_in,), name='feature_input')
    # Define hidden layer initially as identity on input
    h_layer = keras.layers.Lambda(lambda x: x)(feat_in)
    for i in arch[0]:
        if activ == 'prelu':
            h_layer = keras.layers.Dense(i, activation='linear', kernel_regularizer=regularizer)(h_layer)
            h_layer = keras.layers.advanced_activations.PReLU()(h_layer)
        else:
            h_layer = keras.layers.Dense(i,activation=activ, kernel_regularizer=regularizer)(h_layer)
        if bnorm:
            h_layer = keras.layers.BatchNormalization()(h_layer)
        if dropout:
            h_layer = keras.layers.Dropout(do)(h_layer)
            
    
    if skip_connection == 'coupled_highway':
        carry = keras.layers.Dense(width, activation='sigmoid')
#        identity = keras.layers.Lambda(lambda x: x)
#        transfer = keras.layers.Subtract()([identity, carry])
    if skip_connection == 'full_highway':
        carry = keras.layers.Dense(width, activation='sigmoid')
        transfer = keras.layers.Dense(width, activation='sigmoid')
    # Iterate to build the network
    for i in range(depth):
        if dim_in == width or i > 0:
            old_h = h_layer
        if activ == 'prelu':
            h_layer = keras.layers.Dense(width, activation='linear', kernel_regularizer=regularizer)(h_layer)
            h_layer = keras.layers.advanced_activations.PReLU()(h_layer)
        else:
            h_layer = keras.layers.Dense(width, activation=activ, kernel_regularizer=regularizer)(h_layer)
        if bnorm:
            h_layer = keras.layers.BatchNormalization()(h_layer)
        if dropout:
            h_layer = keras.layers.Dropout(do)(h_layer)
        if dim_in == width or i > 0:
            if skip_connection == 'coupled_highway':
                this_carry = carry(old_h)
                this_transfer = keras.layers.Subtract()([old_h, this_carry])
                old_h = keras.layers.Multiply()([this_carry, old_h])
                h_layer = keras.layers.Multiply()([this_transfer, h_layer])
            elif skip_connection == 'full_highway':
                this_carry = carry(old_h)
                this_transfer = transfer(old_h)
                old_h = keras.layers.Multiply()([this_carry, old_h])
                h_layer = keras.layers.Multiply()([this_transfer, h_layer])
            h_layer = keras.layers.Add()([h_layer, old_h])
    
    for i in arch[2]:
        if activ == 'prelu':
            h_layer = keras.layers.Dense(i, activation='linear', kernel_regularizer=regularizer)(h_layer)
            h_layer = keras.layers.advanced_activations.PReLU()(h_layer)
        else:
            h_layer = keras.layers.Dense(i,activation=activ, kernel_regularizer=regularizer)(h_layer)
        if bnorm:
            h_layer = keras.layers.BatchNormalization()(h_layer)
        if dropout:
            h_layer = keras.layers.Dropout(do)(h_layer)
        
    if drop_last_only:
        h_layer = keras.layers.Dropout(do)(h_layer)
    # Define output
    predicted_shift = keras.layers.Dense(units=1, activation='linear')(h_layer)
    
    # Define model and compile
    mod = keras.models.Model(inputs=feat_in, outputs=predicted_shift)
    mod.compile(loss='mean_squared_error', optimizer=opt)

    if lsuv:
        mod = LSUVinit(mod, feat_train[:lsuv_batch])
    
    # Save initialized weights
    weights = mod.get_weights()
    
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
    
            hist = mod.fit(feat_train, shift_train, batch_size=batch_size, epochs=per)
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
        val_epochs = epochs
    
    
    # Retrain model on full dataset for same number of epochs as was
    # needed to obtain min validation loss
    
    mod.set_weights(weights)

    mod.fit(feats, shift_norm, batch_size=batch_size, epochs=val_epochs)

    if reject_outliers is not None:
        return rej_mean, rej_std, val_list, hist_list, param_list, mod
    else:
        return shifts_mean, shifts_std, val_list, hist_list, param_list, mod



def bidir_lstm_model(data, atom, shared_arch, shift_heads, activ='prelu', lrate=0.001, mom=0, 
                    dec=10**-6, epochs=100, min_epochs=5, per=5, tol=1.0, opt_type='adam', do=0, lstm_do=0.0, rec_do=0.0, reg=0.0,
                    reg_type=None,constrain=None, early_stop=None,es_data=None, bnorm=False, lsuv=False, nest=False, norm_shifts=True, 
                    opt_override=False, clip_val=0, clip_norm=0, val_split=0.2, window=5, rolling=True,
                    center_only=True, randomize=False,batch_size=32,sample_weights=True):
    '''Constructs a bidirectional recurrent net with LSTM cell.  Allows for deep bidirectional 
    recurrent nets and time-distributed dense layers that follow all recurrent cells.
    
    data = Feature and target data (Pandas DataFrame)
    atom = List containing name of atoms for predicting shifts at the same time (Str)
    activ = Activation function for dense layers - accepts "prelu" (Str)
    shared_arch = List with first element being a list of fully connected layers prior to the central LSTM piece, the second element specifying         the number of neurons in each of the one-or-more
            recurrent layers and all further elements being the number of neurons in each of the zero-or-more
            time-distributed dense layers, which will be the input for different shift heads (List))
            Example: [[192],[256, 128], 60, 30] means that the resulting network will have two bidirectional LSTM layers,
            the first with 256 neurons and the second with 128 connected from time-distributed dense layers of 192 neurons.
            These recurrent layers will be followed by two time-
            distributed layers, the first with 60 neurons and the second with 30.  This 30-neuron layer will
            connect different shift heads to produce shifts for different atoms.
    shift_heads = List of all the shift heads leading to the output of chemical shifts of different atoms. (List)
            Its length should be the same with the length of atom list
    lrate = Learning rate for SGD (Float)
    mom = Momentum for SGD (Float)
    dec = Decay for SGD learning rate (Float)
    max_epochs = Maximum number of epochs to train (Int)
    min_epochs = Minimum number of epochs to train (Int)
    per = Number of epochs in a test strip for early_stoping (Int)
    tol = early_stoping parameter (Float)
    do = Dropout percentage for LSTM cells, shared time distributed layers and shift heads (List of float)
    reg = Parameter for weight regularization (Float)
    early_stop = Whether or not and how to do early_stoping.  Accepts None, GL, PQ, or UP (Str) If early_stop=int, only do early_stop for such number without train on full data again
    es_data = Dataframe for doing early-stopping, used for fixed train/val/test split
    bnorm = Use batch normalization (Bool)
    lsuv = Use layer-wise sequential unit variance initialization (Bool)
    nest = Use Nesterov momentum (Bool)
    norm_shifts = Normalize the target shifts rather than using raw values (Bool)
    opt_override = Override default optimization parameters (Bool)
    clip_val = Clip values of gradients in optimizer (Float)
    clip_norm = Clip norms of gradients in optimizer (Float)
    val_split = Fraction of training data to set aside as validation data for early stopping (Float)
    window = Number of residues to include in a subsequence (Int)
    rolling = Use a rolling (overlapping) window rather than partitioning the chains into non-overlapping subsequences.
    center_only = Predict only for the central residue of each subsequence (requires odd window)
    randomize = Randomize the order of subsequences within a chain to confirm that state information isn't being (usefully) passed between such subsequences
    '''
    num_shifts=len(atom)
    if not len(shift_heads) == num_shifts:
        raise RuntimeError("Number of shift heads and number of atom types for prediction don't match!")
    # Predicting only center residue requires odd window
    if rolling and center_only:
        if window % 2 is 0:
            window += 1

    data_gen,steps,num_feats,shifts_mean,shifts_std=rnn_data_prep(data,atom,window,batch_size,norm_shifts,val_split=val_split if es_data is None else None)

    # Define optimization procedure
    opt = make_optimizer(opt_type, lrate, mom, dec, nest, clip_norm=clip_norm, clip_val=clip_val, opt_override=opt_override)
    
    # Define weight regularizers
    regularizer = make_reg(reg, reg_type)
    
    # Build model
    #mod = keras.models.Sequential()
    input_layer = keras.layers.Input((window,num_feats))
    layers=input_layer
    for num_nodes in shared_arch[0]:
        if activ is 'prelu':
            layers = keras.layers.TimeDistributed(keras.layers.Dense(units=num_nodes, activation='linear',
            kernel_initializer='random_uniform',bias_initializer='zeros',kernel_regularizer=regularizer,kernel_constraint=constrain))(layers)
            layers = keras.layers.TimeDistributed(keras.layers.advanced_activations.PReLU())(layers)
        else:
            layers = keras.layers.TimeDistributed(keras.layers.Dense(units=num_nodes, activation=activ,
             kernel_regularizer=regularizer,kernel_constraint=constrain))(layers)
        if do:
            layers = keras.layers.TimeDistributed(keras.layers.Dropout(do))(layers)
    for num_nodes in shared_arch[1][:-1]:
        layers = keras.layers.Bidirectional(keras.layers.LSTM(num_nodes, dropout=lstm_do, recurrent_dropout=rec_do, kernel_regularizer=regularizer,
         kernel_initializer='random_uniform',bias_initializer='zeros',kernel_constraint=constrain,return_sequences=True))(layers)
        if do:
            if not center_only:
                layers = keras.layers.TimeDistributed(keras.layers.Dropout(do))(layers)
            else:
                layers = keras.layers.Dropout(do)(layers)
    for num_nodes in shared_arch[1][-1:]:
        layers = keras.layers.Bidirectional(keras.layers.LSTM(num_nodes, dropout=lstm_do, recurrent_dropout=rec_do, kernel_regularizer=regularizer,
         kernel_initializer='random_uniform',bias_initializer='zeros',kernel_constraint=constrain,return_sequences=not center_only))(layers)
        if do:
            if not center_only:
                layers = keras.layers.TimeDistributed(keras.layers.Dropout(do))(layers)
            else:
                layers = keras.layers.Dropout(do)(layers)
    if not center_only:
        for num_nodes in shared_arch[2:]:
            if activ is 'prelu':
                layers = keras.layers.TimeDistributed(keras.layers.Dense(units=num_nodes, activation='linear',
                kernel_initializer='random_uniform',bias_initializer='zeros',kernel_regularizer=regularizer,kernel_constraint=constrain))(layers)
                layers = keras.layers.TimeDistributed(keras.layers.advanced_activations.PReLU())(layers)
            else:
                layers = keras.layers.TimeDistributed(keras.layers.Dense(units=num_nodes, activation=activ,
                kernel_regularizer=regularizer,kernel_constraint=constrain))(layers)
            if do:
                layers = keras.layers.TimeDistributed(keras.layers.Dropout(do))(layers)
        head_layers=[]
        for head in shift_heads:
            new_layers=layers
            for num_nodes in head:
                if activ is 'prelu':
                    new_layers = keras.layers.TimeDistributed(keras.layers.Dense(units=num_nodes, activation='linear',
                    kernel_regularizer=regularizer,kernel_constraint=constrain))(new_layers)
                    new_layers = keras.layers.TimeDistributed(keras.layers.advanced_activations.PReLU())(new_layers)
                else:
                    new_layers = keras.layers.TimeDistributed(keras.layers.Dense(units=num_nodes, activation=activ,
                    kernel_regularizer=regularizer,kernel_constraint=constrain))(new_layers)
                if do:
                    new_layers = keras.layers.TimeDistributed(keras.layers.Dropout(do))(new_layers)
            new_layers=keras.layers.TimeDistributed(keras.layers.Dense(1, activation='linear', 
                kernel_regularizer=regularizer,kernel_constraint=constrain))(new_layers)
            head_layers.append(new_layers)
    else:
        for num_nodes in shared_arch[2:]:
            if activ is 'prelu':
                layers = keras.layers.Dense(units=num_nodes, activation='linear',
                kernel_initializer='random_uniform',bias_initializer='zeros',kernel_regularizer=regularizer,kernel_constraint=constrain)(layers)
                layers = keras.layers.advanced_activations.PReLU()(layers)
            else:
                layers = keras.layers.Dense(units=num_nodes, activation=activ,
                kernel_regularizer=regularizer,kernel_constraint=constrain)(layers)
            if do:
                layers = keras.layers.Dropout(do)(layers)
        head_layers=[]
        for head in shift_heads:
            new_layers=layers
            for num_nodes in head:
                if activ is 'prelu':
                    new_layers = keras.layers.Dense(units=num_nodes, activation='linear',
                    kernel_regularizer=regularizer,kernel_constraint=constrain)(new_layers)
                    new_layers = keras.layers.advanced_activations.PReLU()(new_layers)
                else:
                    new_layers = keras.layers.Dense(units=num_nodes, activation=activ,
                    kernel_regularizer=regularizer,kernel_constraint=constrain)(new_layers)
                if do:
                    new_layers = keras.layers.Dropout(do)(new_layers)
            new_layers=keras.layers.Dense(1, activation='linear', 
                kernel_regularizer=regularizer,kernel_constraint=constrain)(new_layers)
            head_layers.append(new_layers)
    mod=keras.models.Model(inputs=input_layer,outputs=head_layers)
    mod.compile(loss='mse', optimizer=opt, sample_weight_mode=None if center_only else 'temporal')
    print(mod.summary())
    # print(mod.metrics_names)


    weights = mod.get_weights()
    if early_stop is not None:
        if es_data is None:
            train_gen,val_gen,full_gen=data_gen
            train_steps,val_steps,full_steps=steps
            val_epochs, hist_list, val_list, param_list=generator_early_stopping(mod,atom,shifts_std,early_stop,tol,per,epochs,min_epochs,batch_size,train_gen,train_steps,val_gen,val_steps)
        else:
            val_gen,val_steps,_,_,_=rnn_data_prep(es_data,atom,window,batch_size,norm_stats=(shifts_mean,shifts_std))
            val_epochs, hist_list, val_list, param_list=generator_early_stopping(mod,atom,shifts_std,early_stop,tol,per,epochs,min_epochs,batch_size,data_gen,steps,val_gen,val_steps)
    else:
        val_epochs = epochs
        val_list = []
        hist_list = []
        param_list = []
        retrain=True
    if retrain:
        mod.set_weights(weights)
        if es_data is None:
            mod.fit_generator(full_gen, steps_per_epoch=full_steps, epochs=val_epochs,verbose=VERBOSITY0,use_multiprocessing=USE_MULTIPROCESSING,workers=1)
        else:
            data_all=pd.concat([data,es_data],ignore_index=True)
            data_gen,steps,num_feats,shifts_mean,shifts_std = rnn_data_prep(data_all, atom,window,batch_size,norm_shifts)
            mod.fit_generator(data_gen, steps_per_epoch=steps, epochs=val_epochs,verbose=VERBOSITY,use_multiprocessing=USE_MULTIPROCESSING,workers=1)
    return shifts_mean, shifts_std, val_list, hist_list, param_list, mod
    
# Here, we build some evaluators to combine operations needed to get the rmsd of different types of models
def fc_eval(data, model, atom, mean=0, std=1, reject_outliers=None, split_numeric=False, split_class=False):
    '''Function for evaluating the performance of fully connected and residual networks.
    
    mean = Mean to use for standardizing the targets in data (Float)
    std = Standard deviation to use for standardizing the targets in data (Float)
    data = Feature and target data on which to evaluate the performance of the model (Pandas DataFrame)
    model = Model to evaluate (Keras Model)
    atom = Atom for which the model predicts shifts (Str)
    reject_outliers = Shifts that differ from mean by more than this multiple of std are dropped before evaluation (Float or None)
    split_numeric = If true, split the numeric columns in the dataset as separate input (bool)
    split_class = If true, split the classification bins as separate input. Only effective when split_numeric=True (bool)
    '''
    
    dat = data[data[atom].notnull()]
    for col in cols_to_drop:
        try:
            dat = dat.drop(col, axis=1)
        except KeyError:
            pass
        except ValueError:
            pass

    if reject_outliers is not None:
        n_sig = max(reject_outliers, 0.1) # Don't accept a cutoff less than 0.1 * std
        up_rej = mean + n_sig * std
        low_rej = mean - n_sig * std
        dat = dat[(dat[atom] > low_rej) & (dat[atom] < up_rej)]

    feats = dat.copy()
    for at in atom_names:
        try:
            feats = feats.drop(at, axis=1)
        except ValueError:
            pass

    
    # If ring currents or random coil shifts are in the data, drop 
    # those from the other atoms and fill any NaNs to 0
    try:
        ring_col = atom + '_RC'
        rem1 = ring_cols.copy()
        rem1.remove(ring_col)
        feats = feats.drop(rem1, axis=1)
        feats[ring_col] = feats[ring_col].fillna(value=0)
    except KeyError:
        pass
    except ValueError:
        pass
    try:
        rcoil_col = 'RCOIL_' + atom
        rem2 = rcoil_cols.copy()
        rem2.remove(rcoil_col)
        feats = feats.drop(rem2, axis=1)
    except KeyError:
        pass
    except ValueError:
        pass
    if split_numeric:
        feats_non_numeric = feats[non_numerical_cols].values
        feats_numeric=feats[[col for col in feats.columns if col not in non_numerical_cols]]
        if split_class:
            class_cols=[col for col in feats_numeric.columns if "CLASS_" in col]
            feats_class=feats[class_cols].values
            feats_numeric=feats_numeric[[col for col in feats_numeric.columns if col not in class_cols]].values
            feats=[feats_numeric,feats_non_numeric,feats_class]
        else:
            feats=[feats_numeric.values,feats_non_numeric]
    else:
        feats=feats.values
    shifts=dat[atom].values
    shifts_norm=(shifts-mean)/std
    mod_eval = model.evaluate(feats, shifts_norm, verbose=0)
    return np.sqrt(mod_eval) * std


def mtl_eval(data, model, atoms, mean=None, std=None, reject_outliers=None, testing=False):
    '''Function to evaluate performance of models that use multi-task learning
    
    '''
    feats, shifts_list, shifts_means, shifts_stds = mtl_data_prep(data, atoms, norm_shifts=False, testing=testing)
    
    norm_shifts_list = []

    if testing:
        mod_rmsd_list = []
        if mean is not None:
            for i in range(len(shifts_list)):
                norm_shifts = (shifts_list[i] - mean) / std
                norm_shifts_list.append(norm_shifts)
        else:
            norm_shifts_list = shifts_list
        for idx, shifts in enumerate(norm_shifts_list):
            mod_eval = model.evaluate(feats[idx], [shifts[:, 0], shifts[:, 1]], verbose=0)
            mod_rmsd = np.sqrt(mod_eval[1:]) * std
            mod_rmsd = mod_rmsd[idx]
            mod_rmsd_list.append(mod_rmsd)
        rmsd_dict = dict(zip(model.metrics_names[1:], mod_rmsd_list))
    else:
        if mean is not None:
            for i in range(len(shifts_list)):
                norm_shifts = (shifts_list[i] - mean[i]) / std[i]
                norm_shifts_list.append(norm_shifts)
        else:
            norm_shifts_list = shifts_list
        mod_eval = model.evaluate(feats, norm_shifts_list, verbose=0)
        mod_rmsd = np.sqrt(mod_eval[1:]) * std
        rmsd_dict = dict(zip(model.metrics_names[1:], mod_rmsd))

    return rmsd_dict
            

def strucseq_branch_eval(data, model, atom, mean=0, std=1):
    '''Function to evaluate the performance of the strucseq_branch model
    
    mean = Mean to use for standardizing the targets in data (Float)
    std = Standard deviation to use for standardizing the targets in data (Float)
    data = Feature and target data on which to evaluate the performance of the model (Pandas DataFrame)
    model = Model to evaluate (Keras Model)
    atom = Atom for which the model predicts shifts (Str)
    seq_type = Encoding for sequence information - accepts "blosum" or "binary" (Str)    
    '''
    
    dat = data[data[atom].notnull()]
    feats = dat.drop(atom_names, axis=1)
    
    # If ring currents or random coil shifts are in the data, drop 
    # those from the other atoms and fill any NaNs to 0
    try:
        ring_col = atom + '_RC'
        rem1 = ring_cols.copy()
        rem1.remove(ring_col)
        feats = feats.drop(rem1, axis=1)
        feats[ring_col] = feats[ring_col].fillna(value=0)
    except KeyError:
        pass
    except ValueError:
        pass
    try:
        rcoil_col = 'RCOIL_' + atom
        rem2 = rcoil_cols.copy()
        rem2.remove(rcoil_col)
        feats = feats.drop(rem2, axis=1)
    except ValueError:
        pass
    
    try:
        seq_feats = dat[seq_cols].values
        struc_feats = feats.drop(seq_cols, axis=1).values
    except KeyError:
        pass
    except ValueError:
        pass
    try:
        seq_feats = dat[bin_seq_cols].values
        struc_feats = feats.drop(bin_seq_cols, axis=1).values
    except KeyError:
        pass
    except ValueError:
        pass
    shifts = dat[atom].values
    shifts_norm = (shifts - mean) / std
    mod_eval = model.evaluate([struc_feats, seq_feats], shifts_norm, verbose=1)
    return np.sqrt(mod_eval) * std

    
def rnn_eval(data, model, atom, mean=0, std=1, window=5, rolling=True, center_only=True, randomize=False,batch_size=32,sample_weights=True,save_prefix=""):
    '''Function to evaluate the performance of recurrent neural network models
    
    data = Feature and target data on which to evaluate the performance of the model (Pandas DataFrame)
    model = Model to evaluate (Keras Model)
    atom = Atom for which the model predicts shifts (Str)
    mean = Mean to use for stadardizing the targets in the data (Float)
    std = Standard deviation to use for standardizing the targets in the data (Float)
    window = Size of subsequences taken from chains
    rolling = Use rolling window to generate overlapping subsequences (Bool)
    center_only = For rolling window, predict only the central residue (Bool)
    randomize = Randomize the order of subsequences within chains (Bool)
    '''
    num_shifts=len(atom)
    dat=data.copy()

    # Need to drop the random coil and ring current columns for the other atoms if 
    # such columns are in the data.
    try:
        ring_col = [single_atom + '_RC' for single_atom in atom]
        rem1 = ring_cols.copy()
        for column in ring_col:
            rem1.remove(column)
        dat.drop(rem1, axis=1, inplace=True)
        dat[ring_col] = dat[ring_col].fillna(value=0)
    except KeyError:
        pass
    except ValueError:
        pass
    try:
        rcoil_col = ['RCOIL_' + single_atom for single_atom in atom]
        rem2 = rcoil_cols.copy()
        for column in rcoil_col:
            rem1.remove(column)
        dat.drop(rem2, axis=1, inplace=True)
    except KeyError:
            pass
    except ValueError:
        pass

    idxs = sep_by_chains(dat, atom)
    for dcol in cols_to_drop:
        try:
            dat.drop(dcol, axis=1,inplace=True)
        except KeyError:
            pass
        except ValueError:
            pass
    # Form all pieces
    all_pieces=[]
    for chain in idxs:
        chain_length=len(chain)
        if chain_length<window:
            pass
        else:
            for i in range(chain_length-window+1):
                # record the distance from the head of the piece to the start of the chain
                dist_to_start=i
                all_pieces.append(chain[i:i+window])
    feats=dat.drop(atom_names,axis=1)
    if center_only:
        inputs=np.array([feats.loc[piece].values for piece in all_pieces])
        predictions=model.predict(inputs)
        if type(predictions) is not list:
            predictions=[predictions]
        analysis={"%s_%s"%(a,b):[] for a in atom for b in ["pred","real"]}
        for i in range(len(predictions[0])):
            for atom_idx,atom_ in enumerate(atom):
                analysis[atom_+"_pred"].append(predictions[atom_idx][i][0])
                analysis[atom_+"_real"].append(dat.loc[all_pieces[i][int((window-1)/2)]][atom_])
    else:
        # Record all prediction results into dictionary
        predictions=[dict() for atom_ in atom]
        inputs=np.array([feats.loc[piece].values for piece in all_pieces])
        outputs=model.predict(inputs)
        # Keep consistency for single/multiple atom predictions
        if type(outputs) is not list:
            outputs=[outputs]

        for atom_idx,result in enumerate(outputs):
            for piece,atom_pred in zip(all_pieces,outputs[atom_idx]):
                for line,shift in zip(piece,atom_pred):
                    if line in predictions[atom_idx]:
                        predictions[atom_idx][line].append(shift[0])
                    else:
                        predictions[atom_idx][line]=[shift[0]]
        # Calculate average predictions, and get real values
        analysis={"%s_%s"%(a,b):[] for a in atom for b in ["pred","real","pred_std"]}
        for i in predictions[0]:
            for atom_idx,atom_ in enumerate(atom):
                analysis[atom_+"_pred"].append(np.average(predictions[atom_idx][i]))
                analysis[atom_+"_pred_std"].append(np.std(predictions[atom_idx][i]))
                analysis[atom_+"_real"].append(dat.loc[i][atom_])
    analysis=pd.DataFrame(analysis)
    for atom_idx,atom_ in enumerate(atom):
        analysis[atom_+"_pred"]*=std[atom_idx]
        analysis[atom_+"_pred"]+=mean[atom_idx]
        analysis[atom_+"_err"]=analysis[atom_+"_pred"]-analysis[atom_+"_real"]
    analysis.to_csv(save_prefix+"analysis.csv")
    err=[]
    corr=[]
    for atom_ in atom:
        valid=analysis[atom_+"_err"].notnull()
        err.append(np.sqrt(np.average(analysis[valid][atom_+"_err"].values**2)))
        corr.append(np.corrcoef(analysis[valid][atom_+"_pred"],analysis[valid][atom_+"_real"])[1,0])
    return err,corr

def plot(history,val_list,per,fig_name,format_type):
    plt.cla()
    plt.plot(history,label='training loss')
    plt.plot(per*np.arange(len(val_list)),val_list,label='validation loss')
    plt.legend()
    plt.savefig(str(fig_name)+'.'+format_type,format=format_type)
    print("Saved training curve "+str(fig_name)+'.'+format_type)

# Need to write a function that does k-fold cross-validation 
def kfold_crossval(k, data, atom, feval, model, mod_args, mod_kwargs, per=5, out='summary', mod_type='fc', window=9, reuse_val_eps=False, omit_dat=None, one_round=False,save_plot=None):
    '''Function to perform k-fold cross validation on a given model with fixed
    hyper-parameters.
    
    k = Number of folds for the cross validation (Int)
    data = Features and targets (Pandas DataFrame)
    atom = Name of atom for which to predict shifts (Str)
    feval = Function to evaluate rmsd for the model (Function)
    model = Function that takes data and returns a trained model (Function)
    mod_args = List of arguments for model-generating function (List)
    mod_kwargs = Dictionary of keyword arguments for model-generating function (Dict)
    rnn = Assume model takes data like an rnn (by chain) rather than by residue (Bool)
    window = Number of residues in subsequences for rnn training
    reuse_val_eps = Use validation epochs obtained in the first fold of early-stopping for subsequent folds (Bool)
    omit_dat = Data that has been omitted from training but should be included in test (Pandas DataFrame)
    one_round = Use only a single round (Bool)
    '''
    dat = data.copy()
    test_rmsd_list = np.array([])
    train_rmsd_list = np.array([])
    epochs_list = np.array([])
    count = 0

    # Set a flag for multi-task learning
    mod_kwargsc = mod_kwargs.copy()
    if type(atom) == list:
        mtl = True
    else:
        mtl = False

    # Need to handle RNNs differently
    if mod_type == 'rnn': # Note - Currently only uses defaults for rnn_eval for rolling, center_only, and randomize -- Should implement later
        idxs = sep_by_chains(dat, atom)
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
            traindf, testdf = dat.iloc[full_train_idx], dat.iloc[full_test_idx]
            traindf.index = pd.RangeIndex(start=0, stop=len(traindf), step=1)
            testdf.index = pd.RangeIndex(start=0, stop=len(testdf), step=1)
            
            print('count is ' + str(count))
#            print('val_eps is ' + str(val_eps))
            if mod_kwargsc['early_stop'] is None:
                val_eps = mod_kwargsc['epochs']
            else:
                if reuse_val_eps and (count > 0):
                    print('triggered loop branch')
                    mod_kwargsc['early_stop'] = None
                    mod_kwargsc['epochs'] = val_eps
            mod_kwargsc['per']=per
            if window is not None:
                mod_kwargsc['window']=window
            mean, std, val_list, history, param_list, mod = model(traindf, atom, *mod_args, **mod_kwargsc)

            if save_plot:
                plot(history,val_list,per,count,save_plot)

            val_arr = np.array(val_list)
            try: 
                val_eps = 5 * (np.argmin(val_arr) + 1)
            except ValueError:
                pass
            # Append validation epochs to list if this info is available
            try:
                epochs_list = np.append(epochs_list, val_eps)
            except NameError:
                pass
            print("Analyzing testing error...")
            test_rmsd,test_corr = feval(testdf, mod, atom, mean=mean, std=std, window=window,center_only=False,save_prefix="test_")
            print("Analyzing training error...")
            train_rmsd,train_corr = feval(traindf, mod, atom, mean=mean, std=std, window=window,center_only=False,save_prefix="train_")
            test_rmsd_list = np.append(test_rmsd_list, test_rmsd)
            train_rmsd_list = np.append(train_rmsd_list, train_rmsd)
            try:
                print('Results this round:\n epochs:' + str(val_eps))
            except UnboundLocalError:
                print('Results this round: ' )
            print("Training error:")
            print(["%s:%f"%(shift_type,error) for shift_type,error in zip(atom,train_rmsd)] )
            print("Training correlation:")
            print(["%s:%f"%(shift_type,corr) for shift_type,corr in zip(atom,train_corr)] )
            print("Testing error:")
            print(["%s:%f"%(shift_type,error) for shift_type,error in zip(atom,test_rmsd)] )
            print("Testing correlation:")
            print(["%s:%f"%(shift_type,corr) for shift_type,corr in zip(atom,test_corr)] )
            count += 1
            if one_round:
                break
    
    if mod_type == 'fc' or mod_type == 'splitting':
        dat = dat[dat[atom].notnull()]

        kf = skl.model_selection.KFold(n_splits=k, shuffle=True)
        if omit_dat is not None:
            omit = omit_dat.copy()
            for col in cols_to_drop:
                try:
                    omit = omit.drop(col, axis=1)
                except ValueError:
                    pass
            omit_gen = kf.split(omit)

        for train_index, test_index in kf.split(dat):
            # Define this fold
            train_df, test_df = dat.iloc[train_index], dat.iloc[test_index]
            # Add ommitted data to test_df if there is any
            if omit_dat is not None:
                _, omit_test_idx = next(omit_gen)
                omit_test = omit.iloc[omit_test_idx]
                test_df = pd.concat([test_df, omit_test], ignore_index=True)
            # Modify the mod_kwargsc after the first round if want to reuse validation epochs obtained
            # in the first round as the number of epochs for subsequent rounds
            if mod_kwargsc['early_stop'] is None:
                val_eps = mod_kwargsc['epochs']
            else:
                if reuse_val_eps and (count > 0):
                    mod_kwargsc['early_stop'] = None
                    mod_kwargsc['epochs'] = val_eps

                
            # Train the model on this fold
            mean, std, val_list, history, params, mod = model(train_df, atom, *mod_args, **mod_kwargsc)
            
            # Get validation epochs if pretraining
            val_arr = np.array(val_list)
            try: 
                val_eps = 5 * (np.argmin(val_arr) + 1)
            except ValueError:
                pass
            # Append validation epochs to list if this info is available
            try:
                epochs_list = np.append(epochs_list, val_eps)
            except NameError:
                pass
            # Get the value of reject_outliers from mod_kwargsc
            try:
                rej_out = mod_kwargsc['reject_outliers']
            except KeyError:
                rej_out = None
            if mod_type == 'splitting':
                split_numeric=True
            else:
                split_numeric=False
            test_rmsd = feval(test_df, mod, atom, mean=mean, std=std, reject_outliers=rej_out,split_numeric=split_numeric)
            test_rmsd_list = np.append(test_rmsd_list, test_rmsd)
            train_rmsd = feval(train_df, mod, atom, mean=mean, std=std, reject_outliers=rej_out,split_numeric=split_numeric)
            train_rmsd_list = np.append(train_rmsd_list, train_rmsd)
            try:
                print('Results this round for atom ' + str(atom) + ' are ' + str([val_eps, train_rmsd, test_rmsd]))
            except UnboundLocalError:
                print('Results this round for atom ' + str(atom) + ' are ' + str([train_rmsd, test_rmsd]))
            count += 1
            if one_round:
                break
    
    if mod_type == 'hard_mtl':
        kf = skl.model_selection.KFold(n_splits=k, shuffle=True)
        idx_dict = {}
        gen_dict = {}
        comb_dat = dat.copy()
        for at in atom:
            dat[dat[at].notnull()].index
            comb_dat = comb_dat[comb_dat[at].notnull()]
            idx_dict[at] = list(dat[dat[at].notnull()].index)
        idx_dict['combined'] = list(comb_dat.index)
        gen_dict['combined'] = kf.split(idx_dict['combined'])
        for at in atom:
            idx_dict['diff_' + at] = list(set(idx_dict[at]).difference(idx_dict['combined']))
            gen_dict[at] = kf.split(idx_dict['diff_' + at])
        
        train_idx_dict = {}
        test_idx_dict = {}
        for i in range(k):
            train_idx_dict['combined'], test_idx_dict['combined'] = next(gen_dict['combined'])
            for at in atom:
                train_idx_dict[at], test_idx_dict[at] = next(gen_dict[at])
                train_idx_dict['full_' + at + '_' + str(i)] = train_idx_dict[at] + train_idx_dict['combined']
                test_idx_dict['full_' + at + '_' + str(i)] = test_idx_dict[at] + test_idx_dict['combined']
                
                # Modify the mod_kwargsc after the first round if want to reuse validation epochs obtained
                # in the first round as the number of epochs for subsequent rounds
                if mod_kwargsc['early_stop'] is None:
                    val_eps = mod_kwargsc['epochs']
                else:
                    if reuse_val_eps and (count > 0):
                        mod_kwargsc['early_stop'] = None
                        mod_kwargsc['epochs'] = val_eps
                
        
        loss_keys = list(train_rmsd_list[0].keys())
        mean_train_rmsd = {}
        std_train_rmsd = {}
        mean_test_rmsd = {}
        std_test_rmsd = {}
        for k in loss_keys:
            mean_train = np.array([rmsd[k] for rmsd in train_rmsd_list]).mean()
            std_train = np.array([rmsd[k] for rmsd in train_rmsd_list]).std()
            mean_train_rmsd[k], std_train_rmsd[k] = mean_train, std_train
            mean_test = np.array([rmsd[k] for rmsd in test_rmsd_list]).mean()
            std_test = np.array([rmsd[k] for rmsd in test_rmsd_list]).std()
            mean_test_rmsd[k], std_test_rmsd[k] = mean_test, std_test
            
            
            
        
    else:
        mean_train_rmsd = train_rmsd_list.mean()
        std_train_rmsd = train_rmsd_list.std()
        mean_test_rmsd = test_rmsd_list.mean()
        std_test_rmsd = test_rmsd_list.std()
    avg_epochs = epochs_list.mean()
    if out=='summary':
        return avg_epochs, train_rmsd, test_rmsd, std_train_rmsd, std_test_rmsd
    if out=='full':
        return epochs_list, train_rmsd_list, test_rmsd_list

def train_val_test(train_data, val_data, test_data, atom, feval, model, mod_args, mod_kwargs, per=5, out='summary', mod_type='fc',window=7):
    '''
    Function for training a model with specified training data and validation data, and evaluate the performance for the trained model on the testing data
    '''
    if mod_type == 'fc' or mod_type == 'splitting':
        train_dat=train_data.copy()
        train_dat=train_dat[train_dat[atom].notnull()]
        val_dat=val_data.copy()
        val_dat=val_dat[val_dat[atom].notnull()]
        mean, std, val_list, history, params, mod = model(train_dat, atom, *mod_args,es_data=val_dat, **mod_kwargs)
        # Get validation epochs if pretraining
        val_arr = np.array(val_list)
        try: 
            val_eps = per * (np.argmin(val_arr) + 1)
        except ValueError:
            val_eps = 0
        if mod_type == 'splitting':
            split_numeric=True
        else:
            split_numeric=False
        if mod_kwargs.get("has_class",False):
            has_class=True
        else:
            has_class=False
        try:
            rej_out = mod_kwargs['reject_outliers']
        except KeyError:
            rej_out = None
        test_rmsd = feval(test_data, mod, atom, mean=mean, std=std, reject_outliers=rej_out,split_numeric=split_numeric,split_class=has_class)
        train_rmsd = feval(pd.concat([train_data,val_data]), mod, atom, mean=mean, std=std, reject_outliers=rej_out,split_numeric=split_numeric,split_class=has_class)
        if len(val_arr)==0:
            val_arr=[0] 
        val_rmsd=np.sqrt(np.min(val_arr))*std
        print('Results this round for atom ' + str(atom) + ':')
        print("Training epochs:",val_eps)
        print("Training error:",train_rmsd)
        print("Validation error:",val_rmsd)
        print("Testing error:",test_rmsd)
    elif mod_type=="rnn":
        train_dat=train_data.copy()
        val_dat=val_data.copy()
        if mod_kwargs['early_stop'] is None:
            val_eps = mod_kwargs['epochs']
        mod_kwargs['per']=per
        if window is not None:
            mod_kwargs["window"]=window
        mean, std, val_list, history, param_list, mod = model(train_dat, atom, *mod_args,es_data=val_dat, **mod_kwargs)
        val_arr = np.array(val_list)
        if len(val_arr)==0:
            val_arr=[0] 
        val_rmsd=np.sqrt(np.min(val_arr))*std
        try: 
            val_eps = per * (np.argmin(val_arr) + 1)
        except ValueError:
            val_eps = 0
        center_only=mod_kwargs.get("center_only",True)
        print("Analyzing testing error...")
        test_rmsd,test_corr = feval(test_data, mod, atom, mean=mean, std=std, window=window,center_only=center_only,save_prefix="test_")
        print("Analyzing training error...")
        train_rmsd,train_corr = feval(pd.concat([train_data,val_data]), mod, atom, mean=mean, std=std, window=window,center_only=center_only,save_prefix="train_")

        print('Results:\n epochs:' + str(val_eps))
        print("Training error:")
        print(["%s:%f"%(shift_type,error) for shift_type,error in zip(atom,train_rmsd)] )
        print("Training correlation:")
        print(["%s:%f"%(shift_type,corr) for shift_type,corr in zip(atom,train_corr)] )
        print("Testing error:")
        print(["%s:%f"%(shift_type,error) for shift_type,error in zip(atom,test_rmsd)] )
        print("Testing correlation:")
        print(["%s:%f"%(shift_type,corr) for shift_type,corr in zip(atom,test_corr)] )
    return val_eps,train_rmsd,val_rmsd,test_rmsd,mod


def kfold_predictions(k, data, atom, model, mod_args, mod_kwargs, per=5, out='summary', rnn=False, window=10, reuse_val_eps=True): # THIS is to edit for showing distribution of predictions
    '''Function to perform k-fold splitting on the data and, for each split, get the distribution of predictions for
    test and training.  These are then combined for complete "training" and "testing" distributions of predictions.
    Function can also return associated errors and, for the training distribution, the standard deviations.
    --Note that as currently written doesn't accept to reject outliers
    
    k = Number of folds for the cross validation (Int)
    data = Features and targets (Pandas DataFrame)
    atom = Name of atom for which to predict shifts (Str)
    feval = Function to evaluate rmsd for the model (Function)
    model = Function that takes data and returns a trained model (Function)
    mod_args = List of arguments for model-generating function (List)
    mod_kwargs = Dictionary of keyword arguments for model-generating function (Dict)
    rnn = Assume model takes data like an rnn (by chain) rather than by residue (Bool)
    window = Number of residues in subsequences for rnn training
    reuse_val_eps = Use validation epochs obtained in the first fold of early-stopping for subsequent folds (Bool)
    '''
    dat = data[data[atom].notnull()]
    test_rmsd_list = np.array([])
    train_rmsd_list = np.array([])
    epochs_list = np.array([])
    # Need to handle RNNs differently -- NOT YET CORRECTLY IMPLEMENTED
    if rnn:
        raise ValueError('This function not yet implemented for RNNs')
        idxs = sep_by_chains(dat, atom)
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
            traindf, testdf = dat.iloc[full_train_idx], dat.iloc[full_test_idx]
            mean, std, val_list, history, param_list, mod = model(traindf, atom, **mod_kwargs)
            val_arr = np.array(val_list)
            val_eps = 5 * (np.argmin(val_arr) + 1)
            epochs_list = np.append(epochs_list, val_eps)
            test_rmsd = feval(mean, std, testdf, mod, atom, window=window)
            train_rmsd = feval(mean, std, traindf, mod, atom, window=window)
            test_rmsd_list = np.append(test_rmsd_list, test_rmsd)
            train_rmsd_list = np.append(train_rmsd_list, train_rmsd)
    
    else:
        for col in cols_to_drop:
            try:
                dat = dat.drop(col, axis=1)
            except ValueError:
                pass
            
        feats = dat.drop(atom_names, axis=1)
        shifts = dat[atom]
        
        # Need to drop the random coil and ring current columns for the other atoms if 
        # such columns are in the data.
        try:
            ring_col = atom + '_RC'
            rem1 = ring_cols.copy()
            rem1.remove(ring_col)
            feats = feats.drop(rem1, axis=1)
            feats[ring_col] = feats[ring_col].fillna(value=0)
        except KeyError:
            pass
        except ValueError:
            pass
        try:
            rcoil_col = 'RCOIL_' + atom
            rem2 = rcoil_cols.copy()
            rem2.remove(rcoil_col)
            feats = feats.drop(rem2, axis=1)
        except ValueError:
            pass
#        feats = feats.values
        
        train_cols = ['Training_Fold_' + str(i + 1) for i in range(k)]
        base_cols = train_cols + ['Testing']
        full_df = pd.DataFrame(np.nan, index=list(range(len(dat))), columns=base_cols) 

        kf = skl.model_selection.KFold(n_splits=k)
        count=0
        for train_index, test_index in kf.split(dat):
            # Define this fold
            train_df = dat.iloc[train_index]
            # Modify the mod_kwargs after the first round if want to reuse validation epochs obtained
            # in the first round as the number of epochs for subsequent rounds
            if reuse_val_eps and (count > 0):
                mod_kwargs['pretrain'] = None
                mod_kwargs['epochs'] = val_eps
            
            # Set the value of reject_outliers for mod_kwargs
            mod_kwargs['reject_outliers'] = None

            # Train the model on this fold
            mean, std, val_list, history, params, mod = model(train_df, atom, *mod_args, **mod_kwargs)
            
            # Get validation epochs if pretraining
            val_arr = np.array(val_list)
            try: 
                val_eps = 5 * (np.argmin(val_arr) + 1)
            except ValueError:
                pass
            # Append validation epochs to list if this info is available
            try:
                epochs_list = np.append(epochs_list, val_eps)
            except NameError:
                pass
                
            # Test/Train Feats for this fold
            train_feats = feats.iloc[train_index].values
            test_feats = feats.iloc[test_index].values
            
            # Predictions for this fold
            train_pred = mod.predict(train_feats)
            train_pred = train_pred.flatten() * std + mean
            test_pred = mod.predict(test_feats)
            test_pred = test_pred.flatten() * std + mean
            full_df.loc[train_index, 'Training_Fold_' + str(count + 1)] = train_pred
            full_df.loc[test_index, 'Testing'] = test_pred
            count += 1

    full_df['Empirical_Shifts'] = shifts.values
    full_df['Train_Mean'] = full_df[train_cols].mean(axis=1)
    full_df['Train_Std'] = full_df[train_cols].std(axis=1)
    full_df['Train_Error'] = abs(full_df['Train_Mean'] - shifts.values)
    full_df['Test_Error'] = abs(full_df['Testing'] - shifts.values)
    out_df = full_df.drop(train_cols, axis=1)
    train_rmsd = np.sqrt(pd.Series.mean(full_df['Train_Error'] ** 2))
    test_rmsd = np.sqrt(pd.Series.mean(full_df['Test_Error'] ** 2))
    if out=='summary':
        return out_df, train_rmsd, test_rmsd
    if out=='full':
        return out_df


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
