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



# First define some prelimnary things

atom_names = ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE3', 'CE2', 'CG', 'CG1', 'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3', 'H', 'HA', 'HB', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE3', 'HE21', 'HE22', 'HG', 'HG1', 'HG12', 'HG13', 'HG2', 'HG3', 'HH2', 'HZ', 'HZ2', 'HZ3', 'N', 'ND2', 'NE1', 'NE2']
sparta_results = [0.25, 0.49, 0.94, 1.14, 1.09, 2.45] # Advertised performance of SPARTA+
paper_order = ['Ala', 'Cys','Asp','Glu','Phe','Gly','His','Ile','Lys','Leu','Met','Asn','Pro','Gln','Arg','Ser','Thr','Val','Trp','Tyr'] # The amino acid order used in Sparta+ data and blosum matrix (1-letter alphabetic).

# The dictionary storing the hydrophobicity for different residues
# Literature: Wimley WC & White SH (1996). Nature Struct. Biol. 3:842-848. 
hydrophobic_dict={'LYS': 1.81, 'GLN': 0.19, 'THR': 0.11, 'ASP': 0.5, 'GLU': 0.12, 'ARG': 1.0, 'LEU': -0.69, 'TRP': -0.24, 'VAL': -0.53, 
'ILE': -0.81, 'PRO': -0.31, 'MET': -0.44, 'ASN': 0.43, 'SER': 0.33, 'ALA': 0.33, 'GLY': 1.14, 'TYR': 0.23, 'HIS': -0.06, 'PHE': -0.58, 'CYS': 0.22}

# For easier access, define the names of different feature columns.
# These are the names that we assign in our own feature extraction routine.
col_phipsi = [i+j for i in ['PHI_', 'PSI_'] for j in ['COS_i-1', 'SIN_i-1']]
col_phipsi += [i+j for i in ['PHI_', 'PSI_'] for j in ['COS_i', 'SIN_i']]
col_phipsi += [i+j for i in ['PHI_', 'PSI_'] for j in ['COS_i+1', 'SIN_i+1']]
col_chi = [i+j+k for k in ['_i-1', '_i', '_i+1'] for i in ['CHI1_','ALTCHI1_', 'CHI2_','ALTCHI2_', 'CHI3_', 'CHI4_', 'CHI5_'] for j in ['COS', 'SIN', 'EXISTS']]
col_hbprev = ['O_'+i+'_i-1' for i in ['d_HA', '_COS_H', '_COS_A', '_EXISTS']]
col_hbond = [i+j+'_i' for i in ['Ha_', 'HN_', 'O_'] for j in ['d_HA', '_COS_H', '_COS_A', '_EXISTS']]
col_hbnext = ['HN_'+i+'_i+1' for i in ['d_HA', '_COS_H', '_COS_A', '_EXISTS']]
col_s2 = ['S2'+i for i in ['_i-1', '_i', '_i+1']]
struc_cols = col_phipsi + col_chi + col_hbprev + col_hbond + col_hbnext + col_s2
blosum_names = ['BLOSUM62_NUM_'+paper_order[i].upper() for i in range(20)]
col_blosum = [blosum_names[i]+j for j in ['_i-1', '_i', '_i+1'] for i in range(20)]
seq_cols = col_blosum
bin_seq_cols = ['BINSEQREP_'+ sorted(IUPACData.protein_letters_3to1.keys())[i].upper() + j for j in ['_i-1', '_i', '_i+1'] for i in range(20)]
rcoil_cols = ['RCOIL_' + atom for atom in atom_names]
ring_cols = [atom + '_RC' for atom in atom_names]
all_cols = struc_cols + seq_cols + rcoil_cols + ring_cols
all_cols_bin = struc_cols + bin_seq_cols + rcoil_cols + ring_cols

# These are the names of new columns that were not included in the SPARTA+ feature set
hsea_names = ['HSE_CA' + i  for i in ['_U', '_D', '_Angle']]
hseb_names = ['HSE_CB' + i  for i in ['_U', '_D']]
hse_cols = [name + i for i in ['_i-1', '_i', '_i+1'] for name in hsea_names + hseb_names]
dssp_ss_names = ['A_HELIX_SS', 'B_BRIDGE_SS', 'STRAND_SS', '3_10_HELIX_SS', 'PI_HELIX_SS', 'TURN_SS', 'BEND_SS', 'NONE_SS']
dssp_asa_names = ['REL_ASA', 'ABS_ASA']
dssp_pp_names = ['DSSP_PHI', 'DSSP_PSI']
dssp_hb_names = ['NH-O1_ENERGY', 'NH-O2_ENERGY', 'O-NH1_ENERGY', 'O-NH2_ENERGY']
dssp_cols = [name + i for i in ['_i-1', '_i', '_i+1'] for name in dssp_ss_names + dssp_asa_names + dssp_pp_names + dssp_hb_names]
dssp_energy_cols = [name + i for i in ['_i-1', '_i', '_i+1'] for name in dssp_hb_names]
dssp_expp_cols = [name + i for i in ['_i-1', '_i', '_i+1'] for name in dssp_ss_names + dssp_asa_names + dssp_hb_names]
dssp_ssi_cols = [name + '_i' for name in dssp_ss_names]
dssp_norm_cols = [name + i for i in ['_i-1', '_i', '_i+1'] for name in dssp_asa_names + dssp_hb_names]
dssp_pp_cols=[name + i for i in ['_i-1', '_i', '_i+1'] for name in dssp_pp_names]

# These are the names of the columns with sequence information beyond the tri-peptide level
ext_seq_cols = ['RESNAME_i' + i + str(j) for i in ['+', '-'] for j in range(1,21)]

# These column names are for renaming to make consistent between Michael's version
# of the naming and my own
sparta_ring_cols = [atom + '_RING' for atom in atom_names]
sparta_rcoil_cols = ["RC_" + atom  for atom in atom_names]
sparta_rename_cols = sparta_ring_cols + sparta_rcoil_cols
sparta_rename_map = dict(zip(sparta_rename_cols, ring_cols + rcoil_cols))
sx2_rcoil_cols = ['RC_' + atom for atom in atom_names]
sx2_rename_map = dict(zip(sparta_ring_cols + sx2_rcoil_cols, ring_cols + rcoil_cols))

# Define the names for the original Sparta+ Data Columns (same set as above but different order so we can assign column labels to data obtained directly from Yang)
orig_cols = col_blosum[:20]
orig_cols += [i + j for i in ['PHI_', 'PSI_'] for j in ['SIN_i-1', 'COS_i-1']]
orig_cols += [i + j + '_i-1' for i in ['CHI1_','ALTCHI1_', 'CHI2_','ALTCHI2_', 'CHI3_', 'CHI4_', 'CHI5_'] for j in ['SIN', 'COS', 'EXISTS']]
orig_cols += col_blosum[20:40]
orig_cols += [i + j for i in ['PHI_', 'PSI_'] for j in ['SIN_i', 'COS_i']]
orig_cols += [i + j + '_i' for i in ['CHI1_','ALTCHI1_', 'CHI2_','ALTCHI2_', 'CHI3_', 'CHI4_', 'CHI5_'] for j in ['SIN', 'COS', 'EXISTS']]
orig_cols += col_blosum[40:]
orig_cols += [i + j for i in ['PHI_', 'PSI_'] for j in ['SIN_i+1', 'COS_i+1']]
orig_cols += [i + j + '_i+1' for i in ['CHI1_','ALTCHI1_', 'CHI2_','ALTCHI2_', 'CHI3_', 'CHI4_', 'CHI5_'] for j in ['SIN', 'COS', 'EXISTS']]
orig_cols += ['O_'+i+'_i-1' for i in ['_EXISTS', 'd_HA', '_COS_A', '_COS_H']]
orig_cols += [i+j+'_i' for i in ['HN_', 'Ha_', 'O_'] for j in ['_EXISTS', 'd_HA', '_COS_A', '_COS_H']]
orig_cols += ['HN_'+i+'_i+1' for i in ['_EXISTS', 'd_HA', '_COS_A', '_COS_H']]
orig_cols += ['S2'+i for i in ['_i-1', '_i', '_i+1']]

# These are the names of the boolean columns in the Sparta+ features that indicate existance of chi1/chi2 angles as well as possible hydrogen bonds
exist_cols = [i + 'EXISTS' + k for k in ['_i-1', '_i', '_i+1'] for i in ['CHI1_','ALTCHI1_', 'CHI2_','ALTCHI2_', 'CHI3_', 'CHI4_', 'CHI5_']]
exist_cols += ['O_'+ '_EXISTS' +'_i-1', 'HN_' + '_EXISTS' + '_i+1']
exist_cols += [i + '_EXISTS' + '_i' for i in ['Ha_', 'HN_', 'O_']]
# For completeness, here are the remaining columns
non_exist_cols = orig_cols.copy()
for x in exist_cols:
    non_exist_cols.remove(x)

# These columns are multiplied by 10 in the original Sparta+ data
x10cols = [i + j for i in ['PHI_', 'PSI_'] for j in ['SIN_i-1', 'COS_i-1']]
x10cols += [i + j + '_i-1' for i in ['CHI1_','ALTCHI1_', 'CHI2_','ALTCHI2_', 'CHI3_', 'CHI4_', 'CHI5_'] for j in ['SIN', 'COS', 'EXISTS']]
x10cols += [i + j for i in ['PHI_', 'PSI_'] for j in ['SIN_i', 'COS_i']]
x10cols += [i + j + '_i' for i in ['CHI1_','ALTCHI1_', 'CHI2_','ALTCHI2_', 'CHI3_', 'CHI4_', 'CHI5_'] for j in ['SIN', 'COS', 'EXISTS']]
x10cols += [i + j for i in ['PHI_', 'PSI_'] for j in ['SIN_i+1', 'COS_i+1']]
x10cols += [i + j + '_i+1' for i in ['CHI1_','ALTCHI1_', 'CHI2_','ALTCHI2_', 'CHI3_', 'CHI4_', 'CHI5_'] for j in ['SIN', 'COS', 'EXISTS']]

# Make separate lists of cosine, sine, distance columns
phipsicos_cols = [i + 'COS_i-1' for i in ['PHI_', 'PSI_']]
phipsicos_cols += [i + 'COS_i' for i in ['PHI_', 'PSI_']]
phipsicos_cols += [i + 'COS_i+1' for i in ['PHI_', 'PSI_']]
chicos_cols = [i + 'COS' + k for k in ['_i-1', '_i', '_i+1'] for i in ['CHI1_','ALTCHI1_', 'CHI2_','ALTCHI2_', 'CHI3_', 'CHI4_', 'CHI5_']]
hbondcos_cols = ['O_'+i+'_i-1' for i in ['_COS_H', '_COS_A']]
hbondcos_cols += [i+j+'_i' for i in ['Ha_', 'HN_', 'O_'] for j in ['_COS_H', '_COS_A']]
hbondcos_cols += ['HN_'+i+'_i+1' for i in ['_COS_H', '_COS_A']]
cos_cols = phipsicos_cols + chicos_cols + hbondcos_cols
phipsisin_cols = [i + 'SIN_i-1' for i in ['PHI_', 'PSI_']]
phipsisin_cols += [i + 'SIN_i' for i in ['PHI_', 'PSI_']]
phipsisin_cols += [i + 'SIN_i-1' for i in ['PHI_', 'PSI_']]
chisin_cols = [i + 'SIN' + k for k in ['_i-1', '_i', '_i+1'] for i in ['CHI1_','ALTCHI1_', 'CHI2_','ALTCHI2_', 'CHI3_', 'CHI4_', 'CHI5_']]
sin_cols = phipsisin_cols + chisin_cols
# And a list for columns containing distance information
hbondd_cols = ['O_d_HA_i-1']
hbondd_cols += [i+'d_HA_i' for i in ['Ha_', 'HN_', 'O_']]
hbondd_cols += ['HN_d_HA_i+1']
hbondd_sidechain_cols = [i+j for i in ['HB', 'HB1', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE21', 'HE22', 'HG', 'HG1', 'HG12', 'HG13', 'HG2', 'HG3', 'HZ','HD11', 'HD12', 'HD13', 'HD23', 'HE3','HZ3','HH2','HZ2', 'HZ1', 'HG21', 'HG22', 'HG23']  for j in ['_dHA', '_COS_H', '_COS_A']]


# Need a list of columns that will be subject to noise if we wish to use noise injection
angle_cols = cos_cols + sin_cols
noisy_cols = angle_cols + hbondd_cols + col_s2

# Lets try some nonlinear functions of these columns
# Define columns to be squared
square_cols = cos_cols + hbondd_cols + col_s2
#square_cols_names = [x + '_sq' for x in square_cols]
square_cols_names = [x + '^2.0' for x in square_cols]
cols_pown1 = hbondd_cols
cols_pown2 = hbondd_cols
cols_pown3 = hbondd_cols
col_names_pown1 = [x + '^-1.0' for x in cols_pown1]
col_names_pown2 = [x + '^-2.0' for x in cols_pown2]
col_names_pown3 = [x + '^-3.0' for x in cols_pown3]

# Define the names for the _i+1 and _i-1 columns to allow dropping them for RNNs
im1_cols = ['PSI_'+i for i in ['COS_i-1', 'SIN_i-1']]
ip1_cols = ['PHI_'+i for i in ['COS_i+1', 'SIN_i+1']]
im1_cols += [i+j+'_i-1' for i in ['CHI1_','ALTCHI1_', 'CHI2_','ALTCHI2_', 'CHI3_', 'CHI4_', 'CHI5_'] for j in ['COS', 'SIN', 'EXISTS']]
ip1_cols += [i+j+'_i+1' for i in ['CHI1_','ALTCHI1_', 'CHI2_','ALTCHI2_', 'CHI3_', 'CHI4_', 'CHI5_'] for j in ['COS', 'SIN', 'EXISTS']]
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

# Some other columns
protein_letters=[code.upper() for code in IUPACData.protein_letters_3to1.keys()]
sp_feat_cols=['BLOSUM62_NUM_ALA_i-1', 'BLOSUM62_NUM_CYS_i-1', 'BLOSUM62_NUM_ASP_i-1', 'BLOSUM62_NUM_GLU_i-1', 'BLOSUM62_NUM_PHE_i-1', 
'BLOSUM62_NUM_GLY_i-1', 'BLOSUM62_NUM_HIS_i-1', 'BLOSUM62_NUM_ILE_i-1', 'BLOSUM62_NUM_LYS_i-1', 'BLOSUM62_NUM_LEU_i-1', 'BLOSUM62_NUM_MET_i-1', 
'BLOSUM62_NUM_ASN_i-1', 'BLOSUM62_NUM_PRO_i-1', 'BLOSUM62_NUM_GLN_i-1', 'BLOSUM62_NUM_ARG_i-1', 'BLOSUM62_NUM_SER_i-1', 'BLOSUM62_NUM_THR_i-1', 
'BLOSUM62_NUM_VAL_i-1', 'BLOSUM62_NUM_TRP_i-1', 'BLOSUM62_NUM_TYR_i-1', 'PHI_SIN_i-1', 'PHI_COS_i-1', 'PSI_SIN_i-1', 'PSI_COS_i-1', 'CHI1_SIN_i-1',
 'CHI1_COS_i-1', 'CHI1_EXISTS_i-1', 'ALTCHI1_SIN_i-1', 'ALTCHI1_COS_i-1', 'ALTCHI1_EXISTS_i-1', 
 'CHI2_SIN_i-1', 'CHI2_COS_i-1', 'CHI2_EXISTS_i-1', 'ALTCHI2_SIN_i-1', 'ALTCHI2_COS_i-1', 'ALTCHI2_EXISTS_i-1', 'CHI3_SIN_i-1', 'CHI3_COS_i-1', 'CHI3_EXISTS_i-1', 
 'CHI4_SIN_i-1', 'CHI4_COS_i-1', 'CHI4_EXISTS_i-1', 'CHI5_SIN_i-1', 'CHI5_COS_i-1', 'CHI5_EXISTS_i-1'
 'BLOSUM62_NUM_ALA_i', 'BLOSUM62_NUM_CYS_i', 'BLOSUM62_NUM_ASP_i',
 'BLOSUM62_NUM_GLU_i', 'BLOSUM62_NUM_PHE_i', 'BLOSUM62_NUM_GLY_i', 'BLOSUM62_NUM_HIS_i', 'BLOSUM62_NUM_ILE_i', 'BLOSUM62_NUM_LYS_i', 'BLOSUM62_NUM_LEU_i',
 'BLOSUM62_NUM_MET_i', 'BLOSUM62_NUM_ASN_i', 'BLOSUM62_NUM_PRO_i', 'BLOSUM62_NUM_GLN_i', 'BLOSUM62_NUM_ARG_i', 'BLOSUM62_NUM_SER_i', 'BLOSUM62_NUM_THR_i',
 'BLOSUM62_NUM_VAL_i', 'BLOSUM62_NUM_TRP_i', 'BLOSUM62_NUM_TYR_i', 'PHI_SIN_i', 'PHI_COS_i', 'PSI_SIN_i', 'PSI_COS_i', 'CHI1_SIN_i', 'CHI1_COS_i', 
 'CHI1_EXISTS_i', 'ALTCHI1_SIN_i', 'ALTCHI1_COS_i', 'ALTCHI1_EXISTS_i', 
 'CHI2_SIN_i', 'CHI2_COS_i', 'CHI2_EXISTS_i', 'ALTCHI2_SIN_i', 'ALTCHI2_COS_i', 'ALTCHI2_EXISTS_i', 'CHI3_SIN_i', 'CHI3_COS_i', 'CHI3_EXISTS_i', 
 'CHI4_SIN_i', 'CHI4_COS_i', 'CHI4_EXISTS_i', 'CHI5_SIN_i', 'CHI5_COS_i', 'CHI5_EXISTS_i', 'BLOSUM62_NUM_ALA_i+1', 'BLOSUM62_NUM_CYS_i+1', 'BLOSUM62_NUM_ASP_i+1', 
 'BLOSUM62_NUM_GLU_i+1', 'BLOSUM62_NUM_PHE_i+1', 'BLOSUM62_NUM_GLY_i+1', 'BLOSUM62_NUM_HIS_i+1', 'BLOSUM62_NUM_ILE_i+1', 'BLOSUM62_NUM_LYS_i+1', 
 'BLOSUM62_NUM_LEU_i+1', 'BLOSUM62_NUM_MET_i+1', 'BLOSUM62_NUM_ASN_i+1', 'BLOSUM62_NUM_PRO_i+1', 'BLOSUM62_NUM_GLN_i+1', 'BLOSUM62_NUM_ARG_i+1',
'BLOSUM62_NUM_SER_i+1', 'BLOSUM62_NUM_THR_i+1', 'BLOSUM62_NUM_VAL_i+1', 'BLOSUM62_NUM_TRP_i+1', 'BLOSUM62_NUM_TYR_i+1', 'PHI_SIN_i+1', 'PHI_COS_i+1',
'PSI_SIN_i+1', 'PSI_COS_i+1', 'CHI1_SIN_i+1', 'CHI1_COS_i+1', 'CHI1_EXISTS_i+1', 'ALTCHI1_SIN_i+1', 'ALTCHI1_COS_i+1', 'ALTCHI1_EXISTS_i+1', 
 'CHI2_SIN_i+1', 'CHI2_COS_i+1', 'CHI2_EXISTS_i+1', 'ALTCHI2_SIN_i+1', 'ALTCHI2_COS_i+1', 'ALTCHI2_EXISTS_i+1', 'CHI3_SIN_i+1', 'CHI3_COS_i+1', 'CHI3_EXISTS_i+1', 
 'CHI4_SIN_i+1', 'CHI4_COS_i+1', 'CHI4_EXISTS_i+1', 'CHI5_SIN_i+1', 'CHI5_COS_i+1', 'CHI5_EXISTS_i+1', 'O__EXISTS_i-1',
'O_d_HA_i-1', 'O__COS_A_i-1', 'O__COS_H_i-1', 'HN__EXISTS_i', 'HN_d_HA_i', 'HN__COS_A_i', 'HN__COS_H_i', 'Ha__EXISTS_i', 'Ha_d_HA_i', 'Ha__COS_A_i', 
'Ha__COS_H_i', 'O__EXISTS_i', 'O_d_HA_i', 'O__COS_A_i', 'O__COS_H_i', 'HN__EXISTS_i+1', 'HN_d_HA_i+1', 'HN__COS_A_i+1', 'HN__COS_H_i+1', 'S2_i-1', 'S2_i',
 'S2_i+1']
spartap_cols=sp_feat_cols+atom_names+[a+"_RC" for a in atom_names]+["FILE_ID","RESNAME","RES_NUM"]
col_square=["%s_%s_%s"%(a,b,c) for a in ['PHI','PSI'] for b in ['COS','SIN'] for c in ['i-1','i','i+1']]  
dropped_cols=["DSSP_%s_%s"%(a,b) for a in ["PHI","PSI"] for b in ['i-1','i','i+1']]+["BMRB_RES_NUM","MATCHED_BMRB","HA2_RING","HA3_RING","RCI_S2","identifier"]
col_lift=[col for col in sp_feat_cols if "BLOSUM" not in col and "_i-1" not in col and "_i+1" not in col]
non_numerical_cols=['3_10_HELIX_SS_i',"A_HELIX_SS_i","BEND_SS_i","B_BRIDGE_SS_i","CHI1_EXISTS_i","ALTCHI1_EXISTS_i", "CHI2_EXISTS_i","ALTCHI2_EXISTS_i", "CHI3_EXISTS_i", "CHI4_EXISTS_i", "CHI5_EXISTS_i","HN__EXISTS_i","Ha__EXISTS_i","NONE_SS_i","O__EXISTS_i","PI_HELIX_SS_i","STRAND_SS_i","TURN_SS_i"]+protein_letters

# These are the names of the columns that are not in Sparta+ but are in the un-augmented features from our extraction
cols_notinsp = dssp_cols + hse_cols + ext_seq_cols


# Now some stuff to read in data

# Read in ShiftX2 data
# Athena -- 
try:
    train_sx2 = pd.read_csv('/home/bennett/Documents/DataSets/bmrb_clean/clean_rings/train_shiftx2_clean_rings.csv')
    test_sx2 = pd.read_csv('/home/bennett/Documents/DataSets/bmrb_clean/clean_rings/test_shiftx2_clean_rings.csv')
except FileNotFoundError:
    pass
# Hermes
try:
    train_sx2 = pd.read_csv('/home/kochise/data/shiftx2/clean_rings/train_shiftx2_clean_rings.csv')
    test_sx2 = pd.read_csv('/home/kochise/data/shiftx2/clean_rings/test_shiftx2_clean_rings.csv')
except FileNotFoundError:
    pass
# Office
try:
#    train_sx2 = pd.read_csv('/Users/kcbennett/Documents/data/ShiftX2/bmrb_clean/train_shiftx2_clean_rings.csv')
#    test_sx2 = pd.read_csv('/Users/kcbennett/Documents/data/ShiftX2/bmrb_clean/test_shiftx2_clean_rings.csv')
    shiftx2_train = pd.read_csv('/Users/kcbennett/Documents/data/ShiftX2/sx2train_dssp.csv')
    shiftx2_test = pd.read_csv('/Users/kcbennett/Documents/data/ShiftX2/sx2test_dssp.csv')
except FileNotFoundError:
    pass

# Need to write a function to get targets as differences between raw shifts and the
# random coil and ring current values
def diff_targets(data, rings=False, coils=True, drop_cols=True):
    '''Function that replaces the shifts column with the difference between the raw
    shifts and the values in the columns given
    
    data = Feature and target data (Pandas DataFrame)
    rings = Subtract ring current columns from shift columns (Bool)
    coils = Subtract random coil columns from shift columns (Bool)
    drop_cols = Whether or not drop corresponding columns after obtaining the target difference
    '''
    df = data.copy()
    if rings:
        df[atom_names] = df[atom_names].values - df[ring_cols].fillna(0).values
        if drop_cols:
            df.drop(ring_cols, axis=1, inplace=True)
    
    if coils:
        df[atom_names] = df[atom_names].values - df[rcoil_cols].values
        if drop_cols:
            df.drop(rcoil_cols, axis=1, inplace=True)
    
    return df

# Define feature augmentation functions

def feat_pwr(data, columns, pwrs):
    '''Function to augment the feature set by adding new columns corresponding to given powers of selected features.
    
    args:
        data = Contains the data to be augmented (Pandas DataFrame)
        columns = List of columns to be used for the feature augmentation (List of Str)
        pwrs = List of exponents to use for feature augmentation (List of Float)
    
    returns:
        dat - Augmented data (Pandas DataFrame)
    '''
    dat = data.copy()
    for col in columns:
        for power in pwrs:
            new_col_name = col + '^' + str(power)
            if power < 0:
                dat[new_col_name] = 0
                dat.loc[dat[col] > 0, new_col_name] = dat.loc[dat[col] > 0, col]**power
            else:
                dat[new_col_name] = dat[col]**power
    return dat

def Add_res_spec_feats(dataset,include_onehot=True):
    '''
    Adding residue specific features into the dataset (only for current residue), including one-hot representation of the residue,
    and the hydrophobicity of the residue
    '''
    if include_onehot:
        for code in protein_letters:
            dataset[code]=[int(res==code) for res in dataset['RESNAME']]
    dataset["HYDROPHOBICITY"]=[hydrophobic_dict[res] for res in dataset['RESNAME']]

# Feature space lifting
def Lift_Space(dataset,participating_cols,increased_dim,w,b):
    if w is None:
        w = np.random.normal(0, 0.1, (len(participating_cols), increased_dim))
        b = np.random.uniform(0, 2*np.pi, increased_dim)
    lifted_dat_mat=np.cos(dataset[participating_cols].values.dot(w)+b)
    for n in range(increased_dim):
        dataset["Lifted_%d"%n]=lifted_dat_mat[:,n]

def encode_onehot(sequence_feat):
    '''
    Encode the sequence features into one-hot representation.
    sequence_feat: The sequence features that has length of total number of examples (N) and each element
     is a list of residue names with length (L)  (List/DataFrame)
    Returns a three-dimensional numpy array that has shape (N,L,20) 
    '''
    RES_CODES=sorted([code.upper() for code in IUPACData.protein_letters_3to1.keys()])
    RES_CODES_DICT={code:idx for idx,code in enumerate(RES_CODES)}
    onehot=np.zeros((len(sequence_feat),len(sequence_feat[0]),20))
    for idx_seq,seq in enumerate(sequence_feat):
        for idx_res,amino_acid in enumerate(seq):
            if amino_acid in RES_CODES:
                onehot[idx_seq][idx_res][RES_CODES_DICT[amino_acid]]=1
    return onehot 

# Adding classification results into features
def Implant_classification(dataset,model,half_window=20):
    print("Predicting using classification model...")
    input_cols=["RESNAME_i-%d"%n for n in range(half_window,0,-1)]+["RESNAME"]+["RESNAME_i+%d"%n for n in range(1,half_window+1)]
    input_classification=encode_onehot(dataset[input_cols].values)
    pred=model.predict(input_classification,verbose=1)
    class_names=["CLASS_"+str(i+1) for i in range(pred[0].shape[1])]
    for col in class_names:
        dataset[col]=0
    dataset[class_names]=pred[0]

def filter_outlier(data,atom,outlier_path="./"):
    '''
    Function to filter outliers from training and validation data based on PDBID and RESNUM. The outliers were selected out by 10 fold cross validation with sparta+ network with out-of-std predictions
    '''
    data["identifier"]=data["FILE_ID"].astype(str)+data["RES_NUM"].map(str)
    outlier=pd.read_csv(outlier_path+"filtered_"+atom+".csv")
    for i in range(len(outlier)):
        identifier=outlier.iloc[i]["PDB"]+str(outlier.iloc[i]["RESNUM"])
        resname=outlier.iloc[i]["RESNAME"]
        filtered=data[data["identifier"]==identifier]
        assert len(filtered)<=1
        if len(filtered)==1:
            # this item need to be filtered
            idx=filtered.iloc[0].name
            assert data.loc[idx,"RESNAME"]==resname
            data.loc[idx,atom]=np.nan
        if((i+1)%100==0):
            print("%d/%d"%(i+1,len(outlier)),end="\r")
    data.drop("identifier",axis=1,inplace=True)


# Define purifier functions
def dihedral_purifier(data, tol=0.001, drop_cols=True, set_nans=True):
    '''Function to purify given data based on eliminating those residues for which Biopython and DSSP disagree on dihedral angles by more than tol in Cos value for Phi_i.
    
    args:
        data - Data to be purified (Pandas DataFrame)
        tol - Tolerance for dihedral angle Cos values (Float)
        drop_cols - Drop columns associated with DSSP PHI/PSI angles (Bool)
        set_nans - Replace all shifts with NaN in filtered rows rather than removing said rows (Bool)
    
    returns:
        dat - Data with examples removed if Biopython and DSSP disagree by too much (Pandas DataFrame)
    '''
    dat = data.copy()
    phipsidev = np.abs(dat['PHI_COS_i'] - np.cos(np.pi / 180 * dat['DSSP_PHI_i'].astype(np.float64).values))
    good_idxs = ((dat['PHI_COS_i'] == 0) & (dat['PHI_SIN_i'] == 0)) | (phipsidev == 1) | (phipsidev <= 0.001)
    if set_nans:
        dat.loc[~good_idxs, atom_names] = np.nan
    else:
        dat = dat.loc[good_idxs]
    if drop_cols:
        dat = dat.drop(['DSSP_' + i + j for i in ['PHI', 'PSI'] for j in ['_i-1', '_i', '_i+1']], axis=1)
    return dat

def dssp_purifier(data, set_nans=True):
    '''Function to remove examples where DSSP fails to classify the secondary structure.
    
    args:
        data - Given data to be filtered (Pandas DataFrame)
        set_nans - Replace all shifts with NaN in filtered rows rather than removing said rows (Bool)
    
    returns:
        dat - Filtered data (Pandas DataFrame)
    '''
    dat = data.copy()
    bad_idxs = dat[(dat[dssp_ssi_cols[0]]==0) & (dat[dssp_ssi_cols[1]]==0) & (dat[dssp_ssi_cols[2]]==0) & (dat[dssp_ssi_cols[3]]==0) & (dat[dssp_ssi_cols[4]]==0) & (dat[dssp_ssi_cols[5]]==0) & (dat[dssp_ssi_cols[6]]==0) & (dat[dssp_ssi_cols[7]]==0)].index
    if set_nans:
        dat.loc[bad_idxs, atom_names] = np.nan
    else:
        dat = dat.drop(bad_idxs) 
    return dat

def medianize(data, cols, medians=None):
    '''Function to replace zeros by medians for specified columnns.
    
    args:
        data - Data containing columns with zeros to be replaced (Pandas DataFrame)
        cols - Names of columns with zeros that are to be replaced (List)
        medians - Explicit list of medians to use for replacement in testing data (List)
    
    returns:
        meds - Medians used for replacement (List)
        dat - Data with zeros replaced by medians (Pandas DataFrame)
    '''
    dat = data.copy()
    if medians is not None:
        meds = medians
    else:
        meds = [dat.loc[dat[col] != 0, col].median() for col in cols]
    med_dict = dict(zip(cols, meds))
    for col in cols:
        dat.loc[(dat[col] == 0), col] = med_dict[col]
    return meds, dat

def rc_fix(data,use_null=False):
    '''
    Function to replace zeroes in random coils into some more reasonable values
    use_null - If true, set null to those values that have unknown random coil values
    '''
    if use_null:
        checklist={"N":["PRO",np.nan],"H":["PRO",np.nan],"CB":["GLY",np.nan]}
    else:
        checklist={"N":["PRO",119.5],"H":["PRO",8.23],"CB":["GLY",36.8]}
    for atom in atom_names:
        zero_indices=data[data["RCOIL_"+atom]==0].index
        for idx in zero_indices:
            assert data.loc[idx,"RESNAME"]==checklist[atom][0]
            data.loc[idx,"RCOIL_"+atom]=checklist[atom][1]

def ha23ambigfix(data, mode=0):
    '''Function to use the HA2/HA3 ring current calculations in a way that resolves ambiguity.
    
    args:
        data - Data to fix (Pandas DataFrame)
        mode - Method of resolving ambiguity, 0 -> use average, 1 -> use HA2, 2 -> use HA3 (Int)
    
    returns:
        dat - Data with ring current ambiguity resolved and extra columns dropped
    '''
    dat = data.copy()
    idx_ha = [i for i in list(dat[dat['HA2_RING'].notnull()].index)]
    if mode==0:
        for i in idx_ha:
            dat.loc[i, 'HA_RC'] = (dat.loc[i, 'HA2_RING'] + dat.loc[i, 'HA3_RING']) / 2
    elif mode==1:
        for i in idx_ha:
            dat.loc[i, 'HA_RC'] = dat.loc[i, 'HA2_RING']
    elif mode==2:
        for i in idx_ha:
            dat.loc[i, 'HA_RC'] = dat.loc[i, 'HA3_RING']
    dat = dat.drop(['HA2_RING', 'HA3_RING'], axis=1)
    return dat

def check_nan_shifts(data, thr, ends=False):
    '''Function to check the number of NaN shifts in each row and, for each residue with fewer than thr non-NaN shifts, sets all shifts for that residue as well as neighbors to be NaN so that those residues will be excluded from training/evaluation
    
    args:
        data - Data to be cleansed (Pandas DataFrame)
        thr - Number of non-NaN shifts required for a residue and neighbors to survive (Int)
        ends - Set shifts of atoms on first and last residue of each chain to be NaN as well (Bool)
    
    returns:
        dat - Cleaned data (Pandas DataFrame)
    '''
    dat = data.copy()
    for idx in range(len(dat)):
        snums = dat.loc[idx, atom_names].count()
        file = dat.loc[idx, 'PDB_FILE_NAME']
        chain = dat.loc[idx, 'CHAIN']
        if snums < thr:
            dat.loc[idx, atom_names] = np.nan
            if idx > 0:
                file_m1 = dat.loc[idx - 1, 'PDB_FILE_NAME']
                chain_m1 = dat.loc[idx - 1, 'CHAIN']
                if (file == file_m1) and (chain == chain_m1):
                    dat.loc[idx - 1, atom_names] = np.nan
            if idx < (len(dat) -1):
                file_p1 = dat.loc[idx + 1, 'PDB_FILE_NAME']
                chain_p1 = dat.loc[idx + 1, 'CHAIN']
                if (file == file_m1) and (chain == chain_m1):
                    dat.loc[idx + 1, atom_names] = np.nan
        if ends:
            if (idx == 0) or (idx == len(dat) -1):
                dat.loc[idx, atom_names] = np.nan
            elif (file != file_m1) or (chain == chain_m1):
                dat.loc[idx, atom_names] = np.nan
                dat.loc[idx - 1, atom_names] = np.nan
            elif (file != file_p1) or (chain == chain_p1):
                dat.loc[idx, atom_names] = np.nan
                dat.loc[idx + 1, atom_names] = np.nan
    return dat

def raw_dprep(data, ha23mode=0, power_dict={2.0 : square_cols, -1.0 : hbondd_cols, -2.0 : hbondd_cols, -3.0 : hbondd_cols}, diff_rings=False):
    '''Function to prepare freshly-extracted data.
    
    args:
        data - Data to be prepared (Pandas DataFrame)
        ha23mode - Method of resolving ambiguity for HA2/3; see ha23ambigfix function (Int, 0 or 1)
        power_dict - Dictionary where keys are powers and corresponding values are lists of the column names.  These columns are raised to the corresonding power and the result is stored as new columns, augmenting the feature set (Dict)
        diff_rings - Difference the ring current contributions from the shifts.  Only use this if not doing kfold with the result since it requires that we pre-medianize the ring current contributions to eliminate NaN values and in principle this should be done at each fold separately (Bool)
    
    returns:
        dat - Cleaned data ready to be inserted into a kfold_crossval or model-generating function
    
    '''
    dat = data.rename(index=str, columns=sx2_rename_map)
    dat.index = pd.RangeIndex(start=0, stop=len(dat), step=1)
    dat = ha23ambigfix(dat, mode=ha23mode)
    try: # If DSSP columns are available, use them to filter a few bad residues based on the criteria in dihedral_purifier and dssp_purifier
        dat = dihedral_purifier(dat, drop_cols=True)
        dat = dssp_purifier(dat)
    except KeyError:
        pass
    for pwr in power_dict.keys(): # Augment the feature set with a few non-linear transformations
        dat = feat_pwr(dat, power_dict[pwr], [pwr])
    # It is assumed that random coil columns are contained in the data
    if diff_rings:
        for col in ring_cols:
            null_idxs = [int(i) for i in list(dat[dat[col].isnull()].index)]
            dat.loc[null_idxs, col] = dat[col].median()
        dat = diff_targets(dat, rings=True, coils=True)
    else:
        dat = diff_targets(dat, rings=False, coils=True)
    dat.index = pd.RangeIndex(start=0, stop=len(dat), step=1)
    return dat

def sx2_dprep(train_data, test_data, ha23mode=0, med_cols=angle_cols+hbondd_cols, add_squares=square_cols, med_rings=False, diff_rings=False, medians=True, normalize=False):
    '''Function to completely prepare test and train data from shiftx2 extraction.
    
    args:
        train_data = Training data to prepare (Pandas DataFrame)
        test_data = Test data to prepare (Pandas DataFrame)
        ha23mode = Mode for resolving HA2/HA3 ring current ambiguity (0, 1, or 2 -- See ha23ambigfix function)
        
    returns:
        train_dat - Prepared training data (Pandas DataFrame)
        test_dat - Prepared test data (Pandas DataFrame)
    '''
    train_dat = train_data.rename(index=str, columns=sx2_rename_map)
    test_dat = test_data.rename(index=str, columns=sx2_rename_map)
    train_dat.index = pd.RangeIndex(start=0, stop=len(train_dat), step=1)
    test_dat.index = pd.RangeIndex(start=0, stop=len(test_dat), step=1)
    train_dat = dihedral_purifier(train_dat)
    test_dat = dihedral_purifier(test_dat)
    train_dat = dssp_purifier(train_dat)
    test_dat = dssp_purifier(test_dat)
    if medians:
        train_medians, train_dat = medianize(train_dat, med_cols)
        _, test_dat = medianize(test_dat, med_cols, train_medians)
    train_dat = ha23ambigfix(train_dat, mode=ha23mode)
    test_dat = ha23ambigfix(test_dat, mode=ha23mode)
    # Fill in ring columns with medians
    if med_rings:
        for col in ring_cols:
            null_idxs_train = [int(i) for i in list(train_dat[train_dat[col].isnull()].index)]
            null_idxs_test = [int(i) for i in list(test_dat[test_dat[col].isnull()].index)]
            train_dat.loc[null_idxs_train, col] = train_dat[col].median()
            test_dat.loc[null_idxs_test, col] = train_dat[col].median()
    # Add square columns
    if add_squares is not None:
        train_dat = featsq(train_dat, add_squares)
        test_dat = featsq(test_dat, add_squares)
    
    # Difference the raw shifts with random coil (and maybe ring current) contributions 
    train_dat = diff_targets(train_dat, rings=diff_rings, coils=True)
    test_dat = diff_targets(test_dat, rings=diff_rings, coils=True)
    # Normalize all columns
    if normalize:
        sx2_means, sx2_stds, train_dat = whitener(struc_cols + col_blosum + ring_cols + hse_cols + dssp_norm_cols, train_dat)
        _, _, test_dat = whitener(struc_cols + col_blosum + ring_cols + hse_cols + dssp_norm_cols, test_dat, means=sx2_means, stds=sx2_stds, test_data=True)
    
    train_dat.index = pd.RangeIndex(start=0, stop=len(train_dat), step=1)
    test_dat.index = pd.RangeIndex(start=0, stop=len(test_dat), step=1)
    return train_dat, test_dat

def sp_dprep(data, ha23mode=0, spfeats_only=False, med_rings=True, add_squares=square_cols, diff_rings=False):
    '''Function to prepare sparta+ data.'''
    dat = data.rename(index=str, columns=sx2_rename_map)
    dat.index = pd.RangeIndex(start=0, stop=len(dat), step=1)
    dat = dihedral_purifier(dat, drop_cols=not spfeats_only)
    dat = dssp_purifier(dat)
    if spfeats_only:
        dat = dat.drop(dssp_cols + hse_cols + ext_seq_cols, axis=1)
    else:
        dat = featsq(dat, square_cols)
    dat = ha23ambigfix(dat, mode=ha23mode)
    if med_rings:
        for col in ring_cols:
            null_idxs = [int(i) for i in list(dat[dat[col].isnull()].index)]
            dat.loc[null_idxs, col] = dat[col].median()
    dat = diff_targets(dat, rings=diff_rings, coils=True)
    dat.index = pd.RangeIndex(start=0, stop=len(dat), step=1)
    return dat

def hbond_purifier(data, ang_tol=None, drop_phi=False):
    '''Function to filter data by H-bond angles or to drop Phi angles.
    
    args:
        data - Data to be filtered (Pandas DataFrame)
        ang_tol - Tolerance for cosine angle filtration about 180 degrees (None or Float)
        drop_phi - Whether or not to drop the Phi angles (Bool)
    
    returns:
        dat - Filtered data (Pandas DataFrame)'''
    dat = data.copy()
    if ang_tol is not None:
        cos_tol = np.cos((180 - ang_tol) * np.pi/180)
        hyd_angs = [i + '__COS_H_' + j for i in ['HN', 'Ha', 'O'] for j in ['i-1', 'i', 'i+1']]
        for col in hyd_angs:
            if col in dat.columns:
                dat = dat[(dat[col] < cos_tol) | (dat[col] == 0)]
    
    if drop_phi:
        phi_angs = [i + '__COS_A_' + j for i in ['HN', 'Ha', 'O'] for j in ['i-1', 'i', 'i+1']]
        for col in phi_angs:
            if col in dat.columns:
                dat = dat.drop(col, axis=1)
    
    return dat
                

# These few lines are to check chi angles existence for those residues with alt chi definitions
def get_chis(data, resname):
    #resname_dict = {'ASN' : [-2, -3, 1, 0, -3], 'HIS' : [-2, -3, 1, 0, -1], 'TRP' : [-3, -2, -4, -3, 1], 'ASP' : [-2, -3, 6, 2, -3], 'LEU' : [-1, -1, -4, -3, 0], 'PHE' : [-2, -2, -3, -3, 6], 'TYR' : [-2, -2, -3, -2, 3], 'VAL' : [0, -1, -3, -2, -1], 'ILE' : [-1, -1, -3, -3, 0]}
    blos = resname_blos_dict[resname]
    if resname == 'VAL':
        chi_name = 'ALTCHI1_EXISTS_i'
    else:
        chi_name = 'CHI2_EXISTS_i'
    dat = data[(data['BLOSUM62_NUM_ALA_i'] == blos[0]) & (data['BLOSUM62_NUM_CYS_i'] == blos[1]) & (data['BLOSUM62_NUM_ASP_i'] == blos[2]) & (data['BLOSUM62_NUM_GLU_i'] == blos[3]) & (data['BLOSUM62_NUM_PHE_i'] == blos[4])]
    return dat[chi_name]
        
def chi2_exist_nums(data, resname):
    '''returns total number of residues with given resname as well as number of these residues for which chi2 does not exist'''
    chis = get_chis(data, resname)
    nres = len(chis)
    nchis = len(chis[chis < 9])
    return (nres, nchis)


bestH_fckwargs = {'activ' : 'prelu', 'lrate' : 0.001, 'mom' : [0.9, 0.999], 'dec' : 1*10**-6, 'epochs' : 500, 'min_epochs' : 50, 'opt_type' : 'adam', 'do' : 0.4, 'reg' : 0*10**-5, 
                  'reg_type' : 'L2', 'tol' : 4, 'early_stop' : None, 'bnorm' : True, 'nest' : True, 'lsuv' : True, 'clip_norm' : 0.0, 'clip_val' : 0.0, 'reject_outliers' : None, 
                  'opt_override' : True, 'noise' : None, 'noise_type': 'angle', 'noise_dist' : 'uniform', 'batch_size' : 64, 'lsuv_batch' : 2048}

H_fckwargs = {'activ' : 'tanh', 'lrate' : 0.0001, 'mom' : [0.9, 0.999], 'dec' : 1*10**-6, 'epochs' : 450, 'min_epochs' : 50, 'opt_type' : 'sgd', 'do' : 0.0, 'reg' : 1*10**-6, 
              'reg_type' : 'L2', 'tol' : 3, 'pretrain' : 'PQ', 'bnorm' : False, 'nest' : True, 'lsuv' : False, 'clip_norm' : 0.0, 'clip_val' : 0.0, 'reject_outliers' : None,
              'opt_override' : False, 'noise' : None, 'noise_type': 'percent', 'noise_dist' : 'normal', 'batch_size' : 64, 'lsuv_batch' : 2048}



