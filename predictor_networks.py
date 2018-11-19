#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:51:30 2018

@author: bennett
"""

import numpy as np
import random
import sklearn as skl
import keras
from Bio.SeqUtils import IUPACData
from chemshift_prediction.lsuv_init import LSUVinit

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
sparta_ring_cols = [atom + '_RING' for atom in atom_names]
sparta_rcoil_cols = [atom + '_RAND_COIL' for atom in atom_names]
sparta_rename_cols = sparta_ring_cols + sparta_rcoil_cols
sparta_rename_map = dict(zip(sparta_rename_cols, ring_cols + rcoil_cols))

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


# We begin with some data processing functions
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
            

def chain_batch_generator(data, idxs, atom, window, norm_shifts=(0, 1), sample_weights=True, randomize=False, rolling=True, center_only=True):
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
    rolling = Use rolling windows as opposed to non-overlapping windows for subsequences (Bool)
    center_only = predict only the central residue for rolling windows (Bool)
    '''
    dat = data.copy()
    
    try:
        dat = dat.drop(['Unnamed: 0', 'FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM', 'CHAIN'], axis=1)
    except ValueError:
        pass
    try:
        dat = dat.drop(['RESNAME_ip1', 'RESNAME_im1'], axis=1)
    except ValueError:
        pass
    
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
            num_feats = feats.shape[1]
            
            # Get weights as residues with non-null chemical shifts
            if sample_weights:
                weights = chain[atom].notnull()
                weights *= 1
                weights = weights.values
            
            if rolling:
                if window > l:
                    raise ValueError('Window is larger than chain length.  Padding not implemented for rolling sub-sequences.')
                # Implement rolling sub-sequences
                # Make sure that window is odd if only predicting the central residue
                if center_only:
                    if window % 2 is 0:
                        window += 1
                # Initialize outputs
                #print('Confirming window is ' + str(window))
                rolling_feats = np.zeros((l, window, num_feats))
                rolling_shifts = np.zeros((l, window))
                rolling_weights = np.zeros_like(rolling_shifts)
                #shape1 = rolling_shifts[4, :].shape
                #shape2 = shift_norm[4 : 4 + window].shape
                #print('shapes are ' + str(shape1) + ' and ' + str(shape2))
                #print('full shift shape is ' + str(rolling_shifts.shape))
                
                if sample_weights and center_only:
                    for i in range(l):
                        flank_length = (window - 1) // 2
                        if i in range(flank_length):
                            # Handle special case of first flank_length number of sub-sequences
                            rolling_weights[i, i] = 1
                            rolling_weights[i, :] = rolling_weights[i, :] * weights[: window]
                            rolling_shifts[i, :] = shift_norm[: window]
                            rolling_feats[i, :, :] = feats[: window, :]
                            #print('Ran first set')
                        elif i in range(l - flank_length, l):
                            # Handle special case of last flank_length number of sub-sequences
                            rolling_weights[i, window - l + i] = 1
                            rolling_weights[i, :] = rolling_weights[i, :] * weights[l - window :]
                            rolling_shifts[i, :] = shift_norm[l - window :]
                            rolling_feats[i, :, :] = feats[l - window :, :]
                            #print('Ran last set')
                        else:
                            # Handle all sub-sequences in the main body of the chain
                            rolling_weights[i, flank_length] = 1
                            #shape1 = rolling_shifts[i, :].shape
                            #shape2 = shift_norm[i - flank_length : i - flank_length + window].shape
                            #print('shapes are ' + str(shape1) + ' and ' + str(shape2) + ' and i is ' + str(i))
                            rolling_shifts[i, :] = shift_norm[i - flank_length : i - flank_length + window]
                            rolling_feats[i, :, :] = feats[i - flank_length: i - flank_length + window, :]
                            rolling_weights[i, :] = rolling_weights[i, :] * weights[i - flank_length : i - flank_length + window]
                elif sample_weights and not center_only:
                    for i in range(l - window + 1):
                        rolling_weights[i, :] = weights[i : i + window]
                        rolling_shifts[i, :] = shift_norm[i : i + window]
                        rolling_feats[i, :, :] = feats[i : i + window, :]
                
                rolling_shifts = rolling_shifts.reshape((l, window, 1))
                # If all weights are zero then pass to avoid NaN loss
                if np.array_equal(rolling_weights, np.zeros_like(rolling_weights)):
                    pass
                else:
                    if sample_weights:
                        if randomize:
                            np.random.seed(1)
                            np.random.shuffle(rolling_shifts)
                            np.random.seed(1)
                            np.random.shuffle(rolling_feats)
                            np.random.seed(1)
                            np.random.shuffle(rolling_weights)
                            yield rolling_feats, rolling_shifts, rolling_weights
                        else:
                            yield rolling_feats, rolling_shifts, rolling_weights
                    else:
                        yield rolling_feats, rolling_shifts
                
            elif not rolling:
                # Implement non-overlapping sub-sequences
                # Find the number of sub-sequences in this batch
                remainder = l % window
                if remainder == 0:
                    n = l // window
                else:
                    n = l // window + 1
                    # Fill out arrays to full size
                    padding = window - remainder
                    shift_norm = np.pad(shift_norm, (0, padding), mode='constant')
                    feats = np.pad(feats, ((0, padding), (0, 0)), mode='constant')
                    weights = np.pad(weights, (0, padding), mode='constant')
                
                # Reshape into subsequences for batch
                shift_norm = shift_norm.reshape((n, window, 1))
                feats = feats.reshape((n, window, num_feats))
                weights = weights.reshape((n, window))
                # If all weights are zero then pass to avoid NaN loss
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


def sparta_model(data, atom, epochs, per=5, tol=1.0, pretrain=None):
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

    # Build model
    mod = keras.models.Sequential()
    mod.add(keras.layers.Dense(units=30, activation='tanh', input_dim=dim_in))
    mod.add(keras.layers.Dense(units=1, activation='linear'))
    mod.compile(loss='mean_squared_error', optimizer='sgd')
    
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
    mod.set_weights(weights)
    mod = keras.models.Sequential()
    mod.add(keras.layers.Dense(units=30, activation='tanh', input_dim=dim_in))
    mod.add(keras.layers.Dense(units=1, activation='linear'))
    mod.compile(loss='mean_squared_error', optimizer='sgd')
    mod.fit(feats, shift_norm, batch_size=64, epochs=val_epochs)

    return shifts_mean, shifts_std, val_list, hist_list, param_list, mod


def deep_model(data, atom, arch, activ='prelu', lrate=0.001, mom=0, dec=10**-6, epochs=100, min_epochs=5, per=5, tol=1.0, opt_type='sgd', do=0.0, drop_last_only=False, reg=0.0, reg_type=None,
                   pretrain=None, bnorm=False, lsuv=False, nest=False, norm_shifts=True, opt_override=False, clip_val=0.0, clip_norm=0.0, reject_outliers=None):
    '''Constructs a model from the given features and shifts for
    the requested atom.  The model is trained for the given number
    of epochs with the loss being checked every per epochs.  Training
    stops when this loss increases by more than tol. The arch argument
    is a list specifying the number of hidden units at each layer.
    
    data = Feature and target data (Pandas DataFrame)
    atom = Name of atom to predict (Str)
    arch = List of the number of neurons per layer (List)
    activ = Activation function to use (Str - prelu, relu, tanh, etc.)
    lrate = Learning rate for SGD (Float)
    mom = Momentum for SGD (Float)
    dec = Decay for SGD learning rate (Float)
    epochs = Maximum number of epochs to train (Int)
    min_epochs = Minimum number of epochs to train (Int)
    per = Number of epochs in a test strip for pretraining (Int)
    tol = Pretraining parameter (Float)
    opt_type = Optimization procedure to use (Str - sgd, rmsprop, adam, etc.)
    do = Dropout percentage between 0 and 1 (Float)
    drop_last_only = Apply dropout only at last layer (Bool)
    reg = Parameter for weight regularization of dense layers (Float)
    reg_type = Type of weight regularization to use (Str - L1 or L2)
    pretrain = Whether or not and how to do pretraining.  Accepts None, GL, PQ, or UP (Str)
    bnorm = Use batch normalization (Bool)
    lsuv = Use layer-wise sequential unit variance initialization (Bool)
    nest = Use Nesterov momentum (Bool)
    norm_shifts = Normalize the target shifts rather than using raw values (Bool)
    opt_override = Override default parameters for optimization (Bool)
    clip_val = Clip values parameter for optimizer (Float)
    clip_norm = Clip norm parameter for optimizer (Float)
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

    
    if do > 0 and not drop_last_only:
        dropout = True
    else:
        dropout = False

    # Define optimization procedure
    if opt_type is 'sgd':
        if opt_override:
            opt = keras.optimizers.SGD(lr=lrate, momentum=mom, decay=dec, nesterov=nest, clipnorm=clip_norm, clipvalue=clip_val)
        else:
            opt = keras.optimizers.SGD()
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
        mod = LSUVinit(mod, feat_train[:64])
        
    # Get initial weights to reset model after pretraining
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
        val_epochs = epochs

    # Retrain model on full dataset for same number of epochs as was
    # needed to obtain min validation loss
    mod.set_weights(weights)
    mod.fit(feats, shift_norm, batch_size=64, epochs=val_epochs)
    
    if reject_outliers is not None:
        return rej_mean, rej_std, val_list, hist_list, param_list, mod
    else:
        return shifts_mean, shifts_std, val_list, hist_list, param_list, mod


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
    mod = keras.Model(inputs=[struc_in, seq_in], outputs=predicted_shift)
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
                   pretrain=None, bnorm=False, lsuv=False, nest=False, norm_shifts=True, opt_override=False, clip_val=0, clip_norm=0, skip_connection='residual', reject_outliers=None):
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
    depth = arch[0]
    width = arch[1]
    # Define input
    feat_in = keras.layers.Input(shape=(dim_in,), name='feature_input')
    # Define hidden layer initially as identity on input
    h_layer = keras.layers.Lambda(lambda x: x)(feat_in)
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
    
    if drop_last_only:
        h_layer = keras.layers.Dropout(do)(h_layer)
    # Define output
    predicted_shift = keras.layers.Dense(units=1, activation='linear')(h_layer)
    
    # Define model and compile
    mod = keras.Model(inputs=feat_in, outputs=predicted_shift)
    mod.compile(loss='mean_squared_error', optimizer=opt)

    if lsuv:
        mod = LSUVinit(mod, feat_train[:64])
    
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
        val_epochs = epochs
    
    
    # Retrain model on full dataset for same number of epochs as was
    # needed to obtain min validation loss
    
    mod.set_weights(weights)

    mod.fit(feats, shift_norm, batch_size=64, epochs=val_epochs)

    if reject_outliers is not None:
        return rej_mean, rej_std, val_list, hist_list, param_list, mod
    else:
        return shifts_mean, shifts_std, val_list, hist_list, param_list, mod


def bidir_lstm_model(data, atom, arch, activ='prelu', lrate=0.001, mom=0, dec=10**-6, epochs=100, min_epochs=5, per=5, tol=1.0, opt_type='adam', do=0.0, lstm_do=0.0, rec_do=0.0, reg=0.0,
                     reg_type=None, pretrain=None, bnorm=False, lsuv=False, nest=False, norm_shifts=True, opt_override=False, clip_val=0, clip_norm=0, val_split=0.2, window=9, rolling=True,
                     center_only=True, randomize=False):
    '''Constructs a bidirectional recurrent net with LSTM cell.  Allows for deep bidirectional 
    recurrent nets and time-distributed dense layers that follow all recurrent cells.
    
    data = Feature and target data (Pandas DataFrame)
    atom = Name of atom to predict (Str)
    activ = Activation function for dense layers - accepts "prelu" (Str)
    arch = List with first element being a list specifying the number of neurons in each of the one-or-more
            recurrent layers and all further elements being the number of neurons in each of the zero-or-more
            time-distributed dense layers, omitting the time-distriuted output layer which is always present (List))
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
    norm_shifts = Normalize the target shifts rather than using raw values (Bool)
    opt_override = Override default optimization parameters (Bool)
    clip_val = Clip values of gradients in optimizer (Float)
    clip_norm = Clip norms of gradients in optimizer (Float)
    val_split = Fraction of training data to set aside as validation data for early stopping
    window = Number of residues to include in a subsequence (Int)
    rolling = Use a rolling (overlapping) window rather than partitioning the chains into non-overlapping subsequences.
    center_only = Predict only for the central residue of each subsequence (requires odd window)
    randomize = Randomize the order of subsequences within a chain to confirm that state information isn't being (usefully) passed between such subsequences
    '''

    dat = data.copy()
    feats = dat.drop(atom_names, axis=1)
    drop_cols = ['Unnamed: 0', 'FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM', 'CHAIN']
    for dcol in drop_cols:
        try:
            feats = feats.drop(dcol, axis=1)
        except ValueError:
            pass
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
    num_feats = feats.shape[1]
    
    # Get shift statistics
    if norm_shifts:
        all_shifts = data[data[atom].notnull()]
        all_shifts = all_shifts[atom]
        shifts_mean = all_shifts.mean()
        shifts_std = all_shifts.std()

    # Split the data by chain and train/validation sets
    train_set, val_set = sep_by_chains(data, atom=atom, split=val_split)
    full_set = train_set + val_set

    # Predicting only center residue requires odd window
    if rolling and center_only:
        if window % 2 is 0:
            window += 1
    
    # Create generators for training and validation data as well as full data
    train_gen = chain_batch_generator(data, train_set, atom, window, norm_shifts=(shifts_mean, shifts_std), rolling=rolling, randomize=randomize)
    val_gen = chain_batch_generator(data, val_set, atom, window, norm_shifts=(shifts_mean, shifts_std), rolling=rolling, randomize=randomize)
    full_gen = chain_batch_generator(data, full_set, atom, window, norm_shifts=(shifts_mean, shifts_std), rolling=rolling, randomize=randomize)

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
    mod = keras.models.Sequential()
    for num_nodes in arch[0]:
        mod.add(keras.layers.Bidirectional(keras.layers.LSTM(num_nodes, dropout=lstm_do, recurrent_dropout=rec_do, kernel_regularizer=regularizer,
                                                             return_sequences=True), batch_input_shape=(None, window, num_feats)))
        if do[0]:
            mod.add(keras.layers.TimeDistributed(keras.layers.Dropout(do[0])))
    for num_nodes in arch[1:]:
        if activ is 'prelu':
            mod.add(keras.layers.TimeDistributed(keras.layers.Dense(units=num_nodes, activation='linear', kernel_regularizer=regularizer)))
            mod.add(keras.layers.TimeDistributed(keras.layers.advanced_activations.PReLU()))
        else:
            mod.add(keras.layers.TimeDistributed(keras.layers.Dense(units=num_nodes, activation=activ, kernel_regularizer=regularizer)))
        if do[1]:
            mod.add(keras.layers.TimeDistributed(keras.layers.Dropout(do[1])))
    mod.add(keras.layers.TimeDistributed(keras.layers.Dense(1, activation='linear', kernel_regularizer=regularizer)))
    mod.compile(loss='mean_squared_error', optimizer=opt, sample_weight_mode='temporal')
    
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

    mod.set_weights(weights)
    mod.fit_generator(full_gen, steps_per_epoch=full_steps, epochs=val_epochs)
    return shifts_mean, shifts_std, val_list, hist_list, param_list, mod
    
    
# Here, we build some evaluators to combine operations needed to get the rmsd of different types of models
def fc_eval(data, model, atom, mean=0, std=1, reject_outliers=None):
    '''Function for evaluating the performance of fully connected and residual networks.
    
    mean = Mean to use for standardizing the targets in data (Float)
    std = Standard deviation to use for standardizing the targets in data (Float)
    data = Feature and target data on which to evaluate the performance of the model (Pandas DataFrame)
    model = Model to evaluate (Keras Model)
    atom = Atom for which the model predicts shifts (Str)
    reject_outliers = Shifts that differ from mean by more than this multiple of std are dropped before evaluation (Float or None)
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
        up_rej = mean + n_sig * std
        low_rej = mean - n_sig * std
        dat = dat[(dat[atom] > low_rej) & (dat[atom] < up_rej)]

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

    feats = feats.values    
    shifts = dat[atom].values
    shifts_norm = (shifts - mean) / std
    mod_eval = model.evaluate(feats, shifts_norm, verbose=0)
    return np.sqrt(mod_eval) * std


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

    
def rnn_eval(data, model, atom, mean=0, std=1, window=9, rolling=True, center_only=True, randomize=False):
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
    
    idxs = sep_by_chains(data, atom)
    gen = chain_batch_generator(data, atom=atom, idxs=idxs, window=window, norm_shifts=(mean, std), rolling=rolling, center_only=center_only, randomize=randomize)
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
def kfold_crossval(k, data, atom, feval, model, mod_args, mod_kwargs, per=5, out='summary', rnn=False, window=10, reuse_val_eps=False):
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
    '''
    dat = data.copy()
    try:
        dat = dat.drop(['Unnamed: 0', 'FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM', 'CHAIN'], axis=1)
    except ValueError:
        pass
    try:
        dat = dat.drop(['RESNAME_ip1', 'RESNAME_im1'], axis=1)
    except ValueError:
        pass
    test_rmsd_list = np.array([])
    train_rmsd_list = np.array([])
    epochs_list = np.array([])
    # Need to handle RNNs differently
    if rnn:
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
        kf = skl.model_selection.KFold(n_splits=k)
        count=0
        for train_index, test_index in kf.split(dat):
            # Define this fold
            train_df, test_df = dat.iloc[train_index], dat.iloc[test_index]
            # Modify the mod_kwargs after the first round if want to reuse validation epochs obtained
            # in the first round as the number of epochs for subsequent rounds
            if reuse_val_eps and (count > 0):
                mod_kwargs['pretrain'] = None
                mod_kwargs['epochs'] = val_eps
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
            # Get the value of reject_outliers from mod_kwargs
            try:
                rej_out = mod_kwargs['reject_outliers']
            except KeyError:
                rej_out = None
            test_rmsd = feval(test_df, mod, atom, mean=mean, std=std, reject_outliers=rej_out)
            test_rmsd_list = np.append(test_rmsd_list, test_rmsd)
            train_rmsd = feval(train_df, mod, atom, mean=mean, std=std, reject_outliers=rej_out)
            train_rmsd_list = np.append(train_rmsd_list, train_rmsd)
            print('Results this round for atom ' + atom + ' are ' + str([val_eps, train_rmsd, test_rmsd]))
            count += 1

    train_rmsd = train_rmsd_list.mean()
    train_rmsd_spread = train_rmsd_list.std()
    test_rmsd = test_rmsd_list.mean()
    test_rmsd_spread = test_rmsd_list.std()
    avg_epochs = epochs_list.mean()
    if out=='summary':
        return avg_epochs, train_rmsd, test_rmsd, train_rmsd_spread, test_rmsd_spread
    if out=='full':
        return epochs_list, train_rmsd_list, test_rmsd_list
        

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


