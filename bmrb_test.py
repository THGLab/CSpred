#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:04:47 2018

@author: bennett
"""

import pynmrstar
import pandas as pd
import numpy as np
import glob
from Bio.SeqUtils import IUPACData


# The shiftx2 bmrb files are v2 nmr-star so set this module variable
pynmrstar.ALLOW_V2_ENTRIES = True

# Need to hard code the random coil chemical shifts. We use the values
# reported by Wishart et al. in J-Bio NMR, 5 (1995) 67-81.
# First when the residue in question is followed by Alanine (which will
# be our stand-in for a generic amino acid).
atom_names = ['HA', 'H', 'CA', 'CB', 'C', 'N']
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


# Paths at home
train_path_cs = '/home/bennett/Data/shiftx2-trainset-June2011/CS-corrected-training-addPDBresno'
train_path_pdb = '/home/bennett/Data/shiftx2-trainset-June2011/PDB-training-addHydrogens'
test_path_cs = '/home/bennett/Data/shiftx2-testset-June2011/CS-corrected-testset-addPDBresno'
test_path_pdb = '/home/bennett/Data/shiftx2-testset-June2011/PDB-testset-addHydrogens'
# Paths at work
train_path_cs = '/Users/kcbennett/Documents/data/ShiftX2/shiftx2-trainset-June2011/CS-corrected-training-addPDBresno'
train_path_pdb = '/Users/kcbennett/Documents/data/ShiftX2/shiftx2-trainset-June2011/PDB-training-addHydrogens'
test_path_cs = '/Users/kcbennett/Documents/data/ShiftX2/shiftx2-testset-June2011/CS-corrected-testset-addPDBresno'
test_path_pdb = '/Users/kcbennett/Documents/data/ShiftX2/shiftx2-testset-June2011/PDB-testset-addHydrogens'

def add_randcoil_shifts(df, atoms):
    out_df = df
    for atom in atoms:
        out_df['RC_' + atom] = np.nan
        for row in range(len(df)):
            res = df['RESNAME'].iloc[row]
            try:
                nxt_aa = df['RESNAME'].iloc[row+1]
            except IndexError:
                nxt_aa = 'ALA'
            if nxt_aa == 'PRO':
                rc_shift = randcoil_pro[atom][res]
            else:
                rc_shift = randcoil_ala[atom][res]
            out_df['RC_' + atom].iloc[row] = rc_shift
    return out_df
        

def df_from_tsv(path, atoms):
    '''takes path to a tsv file containing chemical shifts and returns a pandas
    dataframe containing the chemical shifts for the desired list of atoms.
    The returned dataframe is indexed by the residue number in the pdb file
    or the Residue_PDB_seq_code.'''

    # Read in data, label columns, and reshape
    df = pd.read_csv(path, comment='#', error_bad_lines=False, skiprows=1,
                     header=None, sep='\s+')
    df.columns = ['Index', 'Residue_PDB_seq_code', 'seq_code', 'Residue_label',
                  'Atom_name', 'Atom_type', 'Chem_shift_value', 'dblzeros',
                  'zeros']
    df = df.drop(['Index', 'seq_code', 'Atom_type', 'dblzeros', 'zeros',
                  'Residue_label'], axis=1)
    pvdf = df.pivot(index='Residue_PDB_seq_code', columns='Atom_name',
                    values='Chem_shift_value')
    pvdf = pvdf.reindex(columns=atoms)
    return pvdf


def df_from_bmrb(path, atoms):
    '''takes path to an nmr-star file, of the format found in the bmrb, and
    returns a pandas dataframe of the chemical shifts for the desired list
    of atoms.  Designed to work with shiftx2 dataset so uses version2 of
    nmr-star files (i.e., requires to set the pynmrstar module variable
    ALLOW_V2_ENTRIES=True). The returned dataframe is indexed by the
    residue number in the pdb file (i.e., Residue_PDB_seq_code).'''

    try:
        entry = pynmrstar.Entry.from_file(path)
        switch = 0
    except ValueError:
        bmr_num = path.split('/')[-1].split('.')[0].split('bmr')[1]
        entry = pynmrstar.Entry.from_file('http://rest.bmrb.wisc.edu/bmrb/NMR-STAR2/' + bmr_num)
        switch = 1

    # Find chemical shifts
    for i in entry:
        if i.category == 'assigned_chemical_shifts':
            for j in i:
                if '_Atom_name' in j.tags:
                    loop = j
            break

    if switch == 0:
        cs = loop.get_tag(['_Residue_PDB_seq_code', '_Atom_name',
                           '_Chem_shift_value'])
    else:
        cs = loop.get_tag(['_Residue_seq_code', '_Atom_name',
                           '_Chem_shift_value'])
    df = pd.DataFrame(cs, columns=['Residue_PDB_seq_code', 'Atom_name',
                                   'Chem_shift_value'])
    df['Chem_shift_value'] = df['Chem_shift_value'].astype(float)

    # IMPORTANT!!  Note that using pivot_table implies an averaging over the
    # shift values if any duplicate atom names are found on a given residue

    pvdf = pd.pivot_table(df, index='Residue_PDB_seq_code',
                          columns='Atom_name', values='Chem_shift_value')
    pvdf = pvdf.reindex(columns=atoms)
    pvdf.fillna(value=np.nan, inplace=True)
    return pvdf

# With above two functions and the sparta_features.py functions, can do


def build_full_df(pdb_path, shift_path, atoms, rcfeats=False):
    '''takes paths to directories of PDB files and tsv/nmr-star files
    and returns a pandas dataframe containing all the residues along
    with their identifying information, sparta+ features, and chemical
    shifts for the desired atoms.  File extensions are based on the
    shiftx2 dataset.'''

    pdb_files = glob.glob(pdb_path + '/*.pdbH')
    nmr_star_files = glob.glob(shift_path + '/*.str.corr.pdbresno')
    nmr_tsv_files = glob.glob(shift_path + '/*.shifty.corr.pdbresno')
    shift_files = nmr_star_files + nmr_tsv_files

    feature_reader = PDB_SPARTAp_DataReader()

    shift_ids = []
    pdb_ids = []
    shift_dict = {}
    pdb_dict = {}
    for i in range(len(shift_files)):
        fileID = shift_files[i].split('/')[-1].split('.')[0].split('_')[0]
        shift_ids.append(fileID)
        shift_dict[fileID] = shift_files[i]

    for i in range(len(pdb_files)):
        fileID = pdb_files[i].split('/')[-1].split('.')[0].split('_')[0]
        pdb_ids.append(fileID)
        pdb_dict[fileID] = pdb_files[i]

    shift_ids.sort()
    pdb_ids.sort()
    # Can sanity check by confirming that pdb_ids == shift_ids
    count = 0
#    errs = []
    for i in pdb_ids:
        pdb_file = pdb_dict[i]

        try:
            nmr_file = shift_dict[i]
        except KeyError:
            print('error on file ' + pdb_file)
            continue

        pdb_df = feature_reader.df_from_file_3res(pdb_file, rcshifts=rcfeats)
        if nmr_file in nmr_star_files:
            shift_df = df_from_bmrb(nmr_file, atoms)
            shift_df.index = shift_df.index.astype(int)
        else:
            shift_df = df_from_tsv(nmr_file, atoms)

        combined_df = pdb_df.join(shift_df, on='RES_NUM')

        if count == 0:
            full_df = combined_df
        else:
            full_df = pd.concat([full_df, combined_df])

        count += 1
        print('finished ' + i)
    return full_df


