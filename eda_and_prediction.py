#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 14:07:26 2018

@author: bennett
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from Bio.SeqUtils import IUPACData
import chemshift_prediction.predictor_networks as pn

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

# Lets try some nonlinear functions of these columns
# Define columns to be squared
phipsicos_cols = ['PSI_COS_i-1']
phipsicos_cols += [i + 'COS_i' for i in ['PHI_', 'PSI_']]
phipsicos_cols += ['PHI_COS_i+1']
chicos_cols = [i + 'COS' + k for k in ['_i-1', '_i', '_i+1'] for i in ['CHI1_', 'CHI2_']]
hbondcos_cols = ['O_'+i+'_i-1' for i in ['_COS_H', '_COS_A']]
hbondcos_cols += [i+j+'_i' for i in ['Ha_', 'HN_', 'O_'] for j in ['_COS_H', '_COS_A']]
hbondcos_cols += ['HN_'+i+'_i+1' for i in ['_COS_H', '_COS_A']]
hbondd_cols = ['O_d_HA_i-1']
hbondd_cols += [i+'d_HA_i' for i in ['Ha_', 'HN_', 'O_']]
hbondd_cols += ['HN_d_HA_i+1']
cos_cols = phipsicos_cols + chicos_cols + hbondcos_cols
square_cols = cos_cols + hbondd_cols
full_square_cols = square_cols + col_s2


# Load ShiftX2 data
try:
    shiftx2_train = pd.read_csv('/Users/kcbennett/Documents/data/ShiftX2/bmrb_clean/train_shiftx2_clean_rings.csv')
    shiftx2_test = pd.read_csv('/Users/kcbennett/Documents/data/ShiftX2/bmrb_clean/test_shiftx2_clean_rings.csv')
except FileNotFoundError:
    pass
try:
    shiftx2_train = pd.read_csv('/home/bennett/Documents/DataSets/bmrb_clean/clean_rings/train_shiftx2_clean_rings.csv')
    shiftx2_test = pd.read_csv('/home/bennett/Documents/DataSets/bmrb_clean/clean_rings/test_shiftx2_clean_rings.csv')
except FileNotFoundError:
    pass



# The following is to prepare a bunch of diferent DataFrames depending on normalization, sequence representation, differencing rcoils, etc
## Change blosum to binary sequence representation
#bin_shiftx2_train = pn.blosum_to_binary(shiftx2_train)
#bin_shiftx2_test = pn.blosum_to_binary(shiftx2_test)
#
## Add random coil shifts
#rcoil_shiftx2_train = pn.add_rand_coils(shiftx2_train)
#rcoil_shiftx2_test = pn.add_rand_coils(shiftx2_test)
#bin_rcoil_shiftx2_train = pn.add_rand_coils(bin_shiftx2_train)
#bin_rcoil_shiftx2_test = pn.add_rand_coils(bin_shiftx2_test)
#
## Take differences with random coil shifts
#rcoildiff_shiftx2_train = pn.diff_targets(rcoil_shiftx2_train, rings=False, coils=True)
#rcoildiff_shiftx2_test = pn.diff_targets(rcoil_shiftx2_test, rings=False, coils=True)
#bin_rcoildiff_shiftx2_train = pn.diff_targets(bin_rcoil_shiftx2_train, rings=False, coils=True)
#bin_rcoildiff_shiftx2_test = pn.diff_targets(bin_rcoil_shiftx2_test, rings=False, coils=True)
#
## Normalize all columns
#rcdiff_shiftx2_means, rcdiff_shiftx2_stds, rcoildiff_shiftx2_normtrain = pn.whitener(struc_cols + seq_cols + ring_cols, rcoildiff_shiftx2_train)
#_, _, rcoildiff_shiftx2_normtest = pn.whitener(struc_cols + seq_cols + ring_cols, rcoildiff_shiftx2_test, means=rcdiff_shiftx2_means, stds=rcdiff_shiftx2_stds, test_data=True)
#bin_rcdiff_shiftx2_means, bin_rcdiff_shiftx2_stds, bin_rcoildiff_shiftx2_normtrain = pn.whitener(struc_cols + ring_cols, bin_rcoildiff_shiftx2_train)
#_, _, bin_rcoildiff_shiftx2_normtest = pn.whitener(struc_cols + ring_cols, bin_rcoildiff_shiftx2_test, means=bin_rcdiff_shiftx2_means, stds=bin_rcdiff_shiftx2_stds, test_data=True)


# Combine ShiftX2 train/test data so we can do k-fold crossval instead
shiftx2 = pd.concat([shiftx2_train, shiftx2_test], ignore_index=True)
bin_shiftx2 = pn.blosum_to_binary(shiftx2)
rcoil_shiftx2 = pn.add_rand_coils(shiftx2)
bin_rcoil_shiftx2 = pn.add_rand_coils(bin_shiftx2)
rcoildiff_shiftx2 = pn.diff_targets(rcoil_shiftx2, rings=False, coils=True)
bin_rcoildiff_shiftx2 = pn.diff_targets(bin_rcoil_shiftx2, rings=False, coils=True)


# Read SPARTA+ data
try:
    #spartap_unfiltered = pd.read_csv('/home/bennett/Documents/DataSets/spartap/spartap.csv')
    spartap = pd.read_csv('/home/bennett/Documents/DataSets/spartap/spartap_yang_cleaned.csv')
except FileNotFoundError:
    pass
try:
    #spartap_unfiltered = pd.read_csv('/Users/kcbennett/Documents/data/SpartaP/spartap.csv')
    spartap = pd.read_csv('/Users/kcbennett/Documents/data/SpartaP/spartap_yang_data/spartap.csv')
except FileNotFoundError:
    pass

# Apply rename map
spartap = spartap.rename(index=str, columns=sparta_rename_map)
spartap.index = pd.RangeIndex(start=0, stop=len(spartap), step=1) # Needed to make indexing in blosum_to_binary function properly
spartap_diff = pn.diff_targets(spartap, rings=False, coils=True)
spartap_bin_diff = pn.blosum_to_binary(spartap_diff)


# Can try augmenting the Sparta+ feature set with non-linear transformations such as squaring
#spartap_bin_diff_cos2 = pn.featsq(spartap_bin_diff, cos_cols)

# Need to prepare some DataFrames for RNNs that drop the _i-1 and _i+1 columns
#reslevel_shiftx2 = rcoildiff_shiftx2.drop(im1_cols + ip1_cols, axis=1)
#reslevel_bin_shiftx2 = bin_rcoildiff_shiftx2.drop(im1_cols_bin + ip1_cols_bin, axis=1)
#reslevel_spartap_diff = spartap_diff.drop(im1_cols + ip1_cols, axis=1)
#reslevel_spartap_bin_diff = spartap_bin_diff.drop(im1_cols + ip1_cols, axis=1)


# Save results as csv
rcoildiff_shiftx2.to_csv('/home/bennett/Documents/DataSets/spartap/shiftx2_diff.csv')
bin_rcoildiff_shiftx2.to_csv('/home/bennett/Documents/DataSets/spartap/shiftx2_bindiff.csv')
spartap_diff.to_csv('/home/bennett/Documents/DataSets/spartap/spartap_diff.csv')
spartap_bin_diff.to_csv('/home/bennett/Documents/DataSets/spartap/spartap_bindiff.csv')





# Below is some code for plotting errors.  May be useful again

#rates = [0.005, 0.01, 0.05, 0.1]
#momenta = [0.7, 0.9, 0.95]
#for act in activs:
#    for dec in decays:
#        train_errs = resdf1.loc[(resdf1['Decay'] == dec) & (resdf1['Activation'] == act)]['Training_Error']
#        test_errs = resdf1.loc[(resdf1['Decay'] == dec) & (resdf1['Activation'] == act)]['Test_Error']
#        fig, ax = plt.subplots()
#        rects1 = ax.bar(ind, train_errs, width,
#                        color='SkyBlue', label='Training')
#        rects2 = ax.bar(ind, test_errs, width,
#                        color='IndianRed', label='Testing', alpha=0.5)
#        
#        # Add some text for labels, title and custom x-axis tick labels, etc.
#        ax.set_ylabel('Loss (MSE)')
#        ax.set_xlabel('Learning Rate (Momentum = [0.7, 0.9, 0.95])')
#        ax.set_title(act + ', Drop=20, Dec=' + str(dec))
#        ax.set_xticks([2, 6, 10, 14])
#        ax.set_xticklabels([str(i) for i in rates])
#        ax.axhline(1.37)
#        ax.axhline(.974)
#        path = '/home/bennett/Documents/Git_Collaborations_THG/shiftpred/performance_pics/'
#        file = act + 'Drop20' + 'Dec' + str(dec)
#        plt.savefig(path + file)
#
#
#
#
#
#
#
#resdf1 = pd.read_pickle('/home/bennett/Documents/Git_Collaborations_THG/shiftpred/hyper_search.pkl')
#train_errs = resdf1.loc[(resdf1['Decay'] == 10**-5) & (resdf1['Activation'] == 'relu')]['Training_Error']
#test_errs = resdf1.loc[(resdf1['Decay'] == 10**-5) & (resdf1['Activation'] == 'relu')]['Test_Error']
#
#fig, ax = plt.subplots()
#ind = np.array([int(i/3)+i+1 for i in range(len(train_errs))])  # the x locations for the groups
#width = 0.45  # the width of the bars
#
#rects1 = ax.bar(ind, train_errs, width,
#                color='SkyBlue', label='Training')
#rects2 = ax.bar(ind, test_errs, width,
#                color='IndianRed', label='Test', alpha=0.5)
#
## Add some text for labels, title and custom x-axis tick labels, etc.
#ax.set_ylabel('Loss (MSE)')
#ax.set_xlabel('Learning Rate (Momentum = [0.7, 0.9, 0.95])')
#ax.set_title('Relu, Drop=20%, Dec=10^-6')
#ax.set_xticks([2, 6, 10, 14])
#ax.set_xticklabels([str(i) for i in rates])
#ax.axhline(1.37)
#ax.axhline(.974)
#ax.
##path = '/home/bennett/Documents/Git_Collaborations_THG/shiftpred/performance_pics/'
##file = act + ', Drop=' + str(drop) + ', Dec=' + str(dec)
##plt.savefig(path + file)
#
#plt.show()
#
#
