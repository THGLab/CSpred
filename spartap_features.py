#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 12:11:25 2017

@author: kcbennett
"""



#import os
#import os.path
#import sys
#sys.path.append('/global/homes/m/mmartin8/.local/cori/2.7-anaconda-4.4/lib/python2.7/site-packages/')
import pandas as pd
import math
from Bio import PDB
from Bio.PDB.PDBParser import PDBParser
from Bio.SubsMat.MatrixInfo import blosum62
from Bio.SeqUtils import IUPACData
import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)
import numpy as np



atom_names = ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE3', 'CE2', 'CG', 'CG1', 'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3', 
'H', 'HA', 'HB', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE3', 'HE21', 'HE22', 'HG', 'HG1', 'HG12', 'HG13', 'HG2', 'HG3', 'HH2', 'HZ', 'HZ2', 'HZ3', 
'N', 'ND2', 'NE1', 'NE2']

#Wishart et al. in J-Bio NMR, 5 (1995) 67-81.
paper_order = ['Ala', 'Cys','Asp','Glu','Phe','Gly','His','Ile','Lys','Leu','Met','Asn','Pro','Gln','Arg','Ser','Thr','Val','Trp','Tyr']

paper_order = [i.upper() for i in paper_order]

# AAlist = [sorted(list(IUPACData.protein_letters_3to1.keys()))[i].upper()
#           for i in range(20)]

rc_ala = {}
rc_ala['N'] = [123.8, 118.8, 120.4, 120.2, 120.3, 108.8, 118.2, 119.9,
               120.4, 121.8, 119.6, 118.7, np.nan, 119.8, 120.5, 115.7,
               113.6, 119.2, 121.3, 120.3]
rc_ala['H'] = [8.24, 8.32, 8.34, 8.42, 8.30, 8.33, 8.42, 8.00,
               8.29, 8.16, 8.28, 8.40, np.nan, 8.32, 8.23, 8.31, 8.15, 8.03,
               8.25, 8.12]
rc_ala['HA'] = [4.32, 4.55, 4.71, 4.64, 4.35, 4.62, 3.96, 4.73, 4.17, 4.32,
                4.34, 4.48, 4.74, 4.42, 4.34, 4.3, 4.47, 4.35, 4.12, 4.66,
                4.55]
rc_ala['C'] = [177.8, 174.6, 176.3, 176.6, 175.8, 174.9, 174.1, 176.4, 176.6,
               177.6, 176.3, 175.2, 177.3, 176.0, 176.3, 174.6, 174.7, 176.3,
               176.1, 175.9]
rc_ala['CA'] = [52.5, 58.2, 54.2, 56.6, 57.7, 45.1, 55.0, 61.1,
                56.2, 55.1, 55.4, 53.1, 63.3, 55.7, 56.0, 58.3, 61.8, 62.2,
                57.5, 57.9]
rc_ala['CB'] = [19.1, 28, 41.1, 29.9, 39.6, np.nan, 29, 38.8, 33.1,
                42.4, 32.9, 38.9, 32.1, 29.4, 30.9, 63.8, 69.8, 32.9, 29.6,
                38.8]
rc_ala['CG'] =   [np.nan, np.nan, 179.7,  36.3,   138.8,  np.nan, 133.53, np.nan, 24.8,   27.0,   32.1 ,  177.0,  27.4,   33.8,   27.2,   np.nan, np.nan, np.nan, 111.7,  129.69]
rc_ala['CD'] =   [np.nan, np.nan, np.nan, 183.6,  np.nan, np.nan, np.nan, np.nan, 29.1,   np.nan, np.nan, np.nan, 50.6,   180.3,  43.4,   np.nan, np.nan, np.nan, np.nan, np.nan]
rc_ala['CD1'] =  [np.nan, np.nan, np.nan, np.nan, 131.7,  np.nan, np.nan, 14.1,   np.nan, 24.9,   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 127.1,  132.88]
rc_ala['CD2'] =  [np.nan, np.nan, np.nan, np.nan, 131.7,  np.nan, 119.9,  np.nan, np.nan, 23.5,   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 126.5, 132.85]
rc_ala['CE'] =   [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 42.1,   np.nan, 17.0,   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
rc_ala['CE1'] =  [np.nan, np.nan, np.nan, np.nan, 131.2,  np.nan, 138.5,  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 118.4]
rc_ala['CE2'] =  [np.nan, np.nan, np.nan, np.nan, 131.2,  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 138.9, 118.21]
rc_ala['CG1'] =  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 27.3,   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 21.2,   np.nan, np.nan]
rc_ala['CG2'] =  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 17.5,   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 21.6,   21.2,   np.nan, np.nan]
rc_ala['CZ'] =   [np.nan, np.nan, np.nan, np.nan, 129.3,  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 159.6,  np.nan, np.nan, np.nan, np.nan, 158.64]
rc_ala['HB'] =   [1.4,    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.8,    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 4.2,    2.0,    np.nan, np.nan]
rc_ala['HB2'] =  [np.nan, 2.9,    2.7,    2.0,    3.1,    np.nan, 3.2,    np.nan, 1.8,    1.6,    2.1,    2.8,    2.3,    2.1,    1.8,    3.9,    np.nan, np.nan, 3.3,    2.94]
rc_ala['HB3'] =  [np.nan, 2.9,    2.7,    2.0,    3.1,    np.nan, 3.1,    np.nan, 1.8,    1.6,    2.0,    2.8,    1.9,    2.1,    1.8,    3.9,    np.nan, np.nan, 3.2,    2.78]
rc_ala['HD1'] =  [np.nan, np.nan, np.nan, np.nan, 7.2,    np.nan, 10.69,  0.8,    np.nan, 0.9,    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7.2,    6.91]
rc_ala['HD2'] =  [np.nan, np.nan, np.nan, np.nan, 7.2,    np.nan, 7.1,    np.nan, 1.7,    0.8,    np.nan, np.nan, 3.6,    np.nan, 3.2,    np.nan, np.nan, np.nan, np.nan, 6.94]
rc_ala['HD21'] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7.6,    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
rc_ala['HD22'] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 6.9,    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
rc_ala['HD3'] =  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.7,    np.nan, np.nan, np.nan, 3.6,    np.nan, 3.2,    np.nan, np.nan, np.nan, np.nan, np.nan]
rc_ala['HE'] =   [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 2.1,    np.nan, np.nan, np.nan, 7.2,    np.nan, np.nan, np.nan, np.nan, np.nan]
rc_ala['HE1'] =  [np.nan, np.nan, np.nan, np.nan, 7.3,    np.nan, 7.8,    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 10.1,   6.69]
rc_ala['HE2'] =  [np.nan, np.nan, np.nan, np.nan, 7.3,    np.nan, 12.61,  np.nan, 3.0,    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 6.69]
rc_ala['HE21'] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7.3,    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
rc_ala['HE22'] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 6.9,    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
rc_ala['HE3'] =  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3.0,    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7.5,    np.nan]
rc_ala['HG'] =   [np.nan, 1.8,    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.6,    np.nan, np.nan, np.nan, np.nan, np.nan, 5.7,    np.nan, np.nan, np.nan, np.nan]
rc_ala['HG1'] =  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 5.6,    0.9,    np.nan, np.nan]
rc_ala['HG12'] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.4,    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
rc_ala['HG13'] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.2,    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
rc_ala['HG2'] =  [np.nan, np.nan, np.nan, 2.3,    np.nan, np.nan, np.nan, 0.9,    1.4,    np.nan, 2.5,    np.nan, 2.0,    2.4,    1.6,    np.nan, 1.2,    0.9,    np.nan, np.nan]
rc_ala['HG3'] =  [np.nan, np.nan, np.nan, 2.3,    np.nan, np.nan, np.nan, np.nan, 1.4,    np.nan, 2.5,    np.nan, 2.0,    2.4,    1.6,    np.nan, np.nan, np.nan, np.nan, np.nan]
rc_ala['HZ'] =   [np.nan, np.nan, np.nan, np.nan, 7.25,   np.nan, np.nan, np.nan, 7.5,    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
rc_ala['CE3'] =  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 120.8,  np.nan]
rc_ala['CZ3'] =  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 121.6,  np.nan]
rc_ala['HZ3'] =  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7.1,    np.nan]
rc_ala['CH2'] =  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 124.5,  np.nan]
rc_ala['HH2'] =  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7.2,    np.nan]
rc_ala['CZ2'] =  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 114.6,  np.nan]
rc_ala['HZ2'] =  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7.4,    np.nan]
rc_ala['ND2'] =  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 112.8,  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
rc_ala['NE1'] =  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 129.4,  np.nan]
rc_ala['NE2'] =  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 177.4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 112.3,  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]



randcoil_ala = {i: dict(zip(paper_order, rc_ala[i])) for i in atom_names}
# When the residue in question is followed by a Proline, we instead use:
rc_pro = {}
rc_pro['N'] = [125, 119.9, 121.4, 121.7, 120.9, 109.1, 118.2, 121.7, 121.6,
               122.6, 120.7, 119.0, np.nan, 120.6, 121.3, 116.6, 116.0, 120.5,
               122.2, 120.8]
rc_pro['H'] = [8.19, 8.30, 8.31, 8.34, 8.13, 8.21, 8.37, 8.06, 8.18,
               8.14, 8.25, 8.37, np.nan, 8.29, 8.2, 8.26, 8.15, 8.02, 8.09,
               8.1]
rc_pro['HA'] = [4.62, 4.81, 4.90, 4.64, 4.9, 4.13, 5.0, 4.47, 4.60, 4.63, 4.82,
                5.0, 4.73, 4.65, 4.65, 4.78, 4.61, 4.44, 4.99, 4.84]
rc_pro['C'] = [175.9, 173, 175, 174.9, 174.4, 174.5, 172.6, 175.0, 174.8,
               175.7, 174.6, 173.6, 171.4, 174.4, 174.5, 173.1, 173.2, 174.9,
               174.8, 174.8]
rc_pro['CA'] = [50.5, 56.4, 52.2, 54.2, 55.6, 44.5, 53.3, 58.7, 54.2, 53.1,
                53.3, 51.3, 61.5, 53.7, 54.0, 56.4, 59.8, 59.8, 55.7, 55.8]
rc_pro['CB'] = [18.1, 27.1, 40.9, 29.2, 39.1, np.nan, 29.0, 38.7, 32.6, 41.7,
                32.4, 38.7, 30.9, 28.8, 30.2, 63.3, 69.8, 32.6, 28.9, 38.3]

rc_pro['CG']  = rc_ala['CG']
rc_pro['CD']  = rc_ala['CD']
rc_pro['CD1'] = rc_ala['CD1']
rc_pro['CD2'] = rc_ala['CD2']
rc_pro['CE']  = rc_ala['CE']
rc_pro['CE1'] = rc_ala['CE1']
rc_pro['CE2'] = rc_ala['CE2']
rc_pro['CG1'] = rc_ala['CG1']
rc_pro['CG2'] = rc_ala['CG2']
rc_pro['CZ']  = rc_ala['CZ']
rc_pro['HB']  = rc_ala['HB']
rc_pro['HB2'] = rc_ala['HB2']
rc_pro['HB3'] = rc_ala['HB3']
rc_pro['HD1'] = rc_ala['HD1']
rc_pro['HD2'] = rc_ala['HD2']
rc_pro['HD21']= rc_ala['HD21']
rc_pro['HD22']= rc_ala['HD22']
rc_pro['HD3'] = rc_ala['HD3']
rc_pro['HE']  = rc_ala['HE']
rc_pro['HE1'] = rc_ala['HE1']
rc_pro['HE2'] = rc_ala['HE2']
rc_pro['HE21']= rc_ala['HE21']
rc_pro['HE22']= rc_ala['HE22']
rc_pro['HE3'] = rc_ala['HE3']
rc_pro['HG']  = rc_ala['HG']
rc_pro['HG1'] = rc_ala['HG1']
rc_pro['HG12']= rc_ala['HG12']
rc_pro['HG13']= rc_ala['HG13']
rc_pro['HG2'] = rc_ala['HG2']
rc_pro['HG3'] = rc_ala['HG3']
rc_pro['HZ']  = rc_ala['HZ']
rc_pro['CE3'] = rc_ala['CE3']
rc_pro['CZ3'] = rc_ala['CZ3']
rc_pro['HZ3'] = rc_ala['HZ3']
rc_pro['CH2'] = rc_ala['CH2']
rc_pro['HH2'] = rc_ala['HH2']
rc_pro['CZ2'] = rc_ala['CZ2']
rc_pro['HZ2'] = rc_ala['HZ2']
rc_pro['ND2']  = rc_ala['ND2']
rc_pro['NE1']  = rc_ala['NE1']
rc_pro['NE2']  = rc_ala['NE2']




randcoil_pro = {i: dict(zip(paper_order, rc_pro[i])) for i in atom_names}
oxidized_cys_correction = {"H": 0.11, "HA": 0.16, "C": 0, "CA": -2.8, "CB": 13.1, "N": -0.2, "CG": np.nan, 'CD': np.nan, 'CD1': np.nan, 'CD2': np.nan, 'CE': np.nan, 'CE1': np.nan, 'CE2': np.nan, 'CG1': np.nan, 'CG2': np.nan, 'CZ': np.nan, 'HB': np.nan, 'HB2': np.nan, 'HB3': np.nan, 'HD1': np.nan, 'HD2': np.nan, 'HD21': np.nan, 'HD22': np.nan, 'HD3': np.nan, 'HE': np.nan, 'HE1': np.nan, 'HE2': np.nan, 'HE21': np.nan, 'HE22': np.nan, 'HE3': np.nan, 'HG': np.nan, 'HG1': np.nan, 'HG12': np.nan, 'HG13': np.nan, 'HG2': np.nan, 'HG3': np.nan, 'HZ': np.nan, 'CE3': np.nan, 'CZ3': np.nan, 'HZ3': np.nan, 'CH2': np.nan, 'HH2': np.nan, 'CZ2': np.nan, 'HZ2': np.nan, 'ND2': np.nan,'NE': np.nan, 'NE1': np.nan,'NE2': np.nan}

secondary_struc_dict = dict(zip(['H', 'B', 'E', 'G', 'I', 'T', 'S', '-'], [list(np.identity(8)[i]) for i in range(8)]))
max_asa_dict = dict(zip(paper_order, [121.0, 148.0, 187.0, 214.0, 228.0, 97.0, 216.0, 195.0, 230.0, 191.0, 203.0, 187.0, 154.0, 214.0, 265.0, 143.0, 163.0, 165.0, 264.0, 255.0]))


class BaseDataReader(object):
    """ Base class for reading data files

    The main function of any child of this class is to read a text file
    in some relevant format and return a pandas DataFrame containing the
    data in that file.

    Args:
        columns (list or iterator): column labels of resulting DataFrame

    Attributes:
        columns_ (list or iterator): column labels of resulting DataFrame

    """

    def __init__(self, columns):
        self.columns_ = columns

    def df_from_file(self, fpath):
        """ Convert a file into a DataFrame

        Args:
            fpath (str): path to file

        Returns:
            pandas DataFrame with the column labels in columns_

        """
        raise NotImplementedError


class PDB_SPARTAp_DataReader(BaseDataReader):
    """ Data Reader class for PDB files. Child of BaseDataReader

    This class is designed to read PDB files and return DataFrame objects with all of
    the relevant data. Each row in the DataFrame that is output by df_from_file
    corresponds to a residue in the molecular structure described by the input PDB file.
    The relevant columns then correspond to the residue index (a unique identifier that
    connects the residue to the atoms it contains in the atom table) and a set of features to
    be described below

    Attributes:
        COLS_ (dict): dictionary mapping column names to data types

    """
#    COLS_ = {'FILE_ID': str,
#             'PDB_FILE_NAME': str,
#             'RES_NAME': str,
#             'PDB_RES_NUM': int,
#             'BLOSUM62_NUMS': list,
#             'DIHEDRAL_PHI': list,
#             'DIHEDRAL_PSI': list,
#             'DIHEDRAL_CHI1': list,
#             'DIHEDRAL_CHI2': list,
#             'HBOND_PATTERN': list,
#             'S2_ORDER_PARAM': float}

    COLS_ = {'FILE_ID': str,
             'PDB_FILE_NAME': str,
             'RES_NAME': str,
             'PDB_RES_NUM': int,
             'S2': float}
#
#
#    COLS_.update({'BLOSUM62_NUM_'+list(IUPACData.protein_letters_3to1.keys())[i].upper() + j: float)
#    for j in ['', '_i-1', '_i', '_i+i'] for i in range(20)]})
#    COLS_.update(dict([(i+j, float) for i in ['PHI_i-1', 'PSI_i+1', 'PHI_i', 'PSI_i', 'CHI1_i', 'CHI2_i'] for j in ['COS', 'SIN']]))
#    COLS_.update(dict([(i+j, bool) for i in ['CHI1_i', 'CHI2_i'] for j in ['EXISTS']]))
#    COLS_.update(dict([(i+j, float) for i in ['HN_i', 'HA_i', 'O_i', 'O_i-1', 'HN_i+1'] for j in ['COS_H', 'COS_A', 'd_HA']]))
#    COLS_.update(dict([(i+j, bool) for i in ['HN_i', 'HA_i', 'O_i', 'O_i-1', 'HN_i+1'] for j in ['EXISTS']]))

    def __init__(self):
        super(PDB_SPARTAp_DataReader, self).__init__(columns=PDB_SPARTAp_DataReader.COLS_.keys())

    @staticmethod
    def _fix_atom_type(start_atom_type):
        '''
        Fix improperly formatted atom types

        This means moving any number at the start of thpandase string to
        the end and changing Deuteriums to Hydrogens. E.g. if the
        input is '1DG2', this becomes 'HG12'.

        Args:
            start_atom_type (str): inital atom type

        Returns:
            atom_type - The fixed atom type (str)
        '''

        atom_type = [ch for ch in start_atom_type]
        if atom_type[0].isnumeric():
            atom_type.append(atom_type[0])
            atom_type.pop(0)
        if atom_type[0] == 'D':
            atom_type[0] = 'H'
        atom_type = ''.join(atom_type)
        return atom_type


    @staticmethod
    def _fix_res_name(start_res_name):
        '''
        Fix wrong residue names

        This is just removing extraneous capital letters at the front of some residue names

        Args:
            start_res_name - Intital residue name (Str)

        Returns:
            res_name - The fixed residue name (Str)
        '''

        res_name = start_res_name
        if len(start_res_name) > 3:
            res_name = start_res_name[1:]
        return res_name


    def get_bfactor(self, res_obj, atoms='all'):
        '''Function to get the b-factor for a given residue.  Can return the average of [C, CA, CB, H, HA, O] atoms or average all atoms in the residue.

        args:
            res_obj - Residue for which an average b-factor is desired (Biopython.PDB Residue object)
            atoms - Which atoms' b-factors to average. Accepts "all" or "set6" (Str)

        returns:
            bfact - Average b-factor (Float)
        '''
        btot = 0
        if atoms == 'all':
            at_list = list(res_obj.get_atoms())
            for atom in at_list:
                btot += atom.get_bfactor()
            bfact = btot / len(at_list)
        if atoms == 'set6':
            for at in ['H', '1H', 'H1', 'D', '1D', 'D1']:
                try:
                    HNatom = res_obj[at]
                    btot += HNatom.get_bfactor()
                    break
                except KeyError:
                    HNatom = None
            for at in ['HA', '1HA']:
                try:
                    HAatom = res_obj[at]
                    btot += HAatom.get_bfactor()
                    break
                except KeyError:
                    HAatom = None
            for at in ['N', 'C', 'CA', 'CB']:
                try:
                    this_atom = res_obj[at]
                    btot += this_atom.get_bfactor()
                except KeyError:
                    pass
            bfact = btot / 6
        return bfact


    def electric_field(self, nn_tree, res_object_list, rad=10.0):

    
        '''
        Function to calculate electric field for hydrogen and nitrogen atoms
    
        Args:
            rad = 10 - following the literature: DOI:10.1021/acs.biochem.7b00170
            nn_tree - A tree containing the distances between atoms in the set of interest (Bio.PDB.NeighborSearch)
    
        '''
    
        #Make sure all residues have unique numbers
        res_numbers = [i.get_id()[1] for i in res_object_list]  #list of res numbers
        if len(res_numbers)>len(set(res_numbers)):
            raise Exception('Not all residue numbers are unique in electric_field res_object_list')
    
        atom_names_target = ['H', 'HA', 'HB', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE3', 'HE21', 'HE22', 'HG', 'HG1', 'HG12', 'HG13', 'HG2', 'HG3', 'HH2', 'HZ', 'HZ2', 'HZ3', 'N', 'ND2', 'NE1', 'NE2']
    
    
        partners = {"GLY": {"H": "N", "HA2": "CA", "HA3": "CA", "N": "CA"},
        "ILE": {"H": "N", "HA": "CA", "HB": "CB", "HD1": "CD1", "HD2": "CD1", "HD3": "CD1", "HG12": "CG1", "HG13": "CG1", "HG21": "CG2", "HG22": "CG2", "HG23": "CG2", "N": "CA"},
        "VAL": {"H": "N", "HA": "CA", "HG12": "CG1", "HG13": "CG1", "HG21": "CG2", "HG22": "CG2", "HB": "CB", "HG11": "CG1", "HG23": "CG2", "N": "CA"},
        "ARG": {"H": "N", "HA": "CA", "HG2": "CG", "HB2": "CB", "HB3": "CB", "HD2": "CD", "HD3": "CD", "HE": "NE", "HG3": "CG","HH11": "NH1", "HH12": "NH1", "HH21": "NH2", "HH22": "NH2", "N": "CA", "NE": "CD", "NH1": "CZ", "NH2": "CZ"},
        "TRP": {"H": "N", "HA": "CA", "HH2": "CH2", "HB2": "CB", "HB3": "CB", "HD1": "CD1", "HE1": "NE1", "HE3": "CE3", "HZ2": "CZ2",  "HZ3": "CZ3", "N": "CA" , "NE1": "CE2"},
        "GLU": {"H": "N", "HA": "CA", "HG2": "CG", "HG2": "CG", "HB2": "CB", "HB3": "CB", "HE2": "OE2", "HG3": "CG", "N": "CA"},
        "GLN": {"H": "N", "HA": "CA", "HG2": "CG", "HB2": "CB", "HB3": "CB", "HE21": "NE2", "HE22": "NE2", "HG3": "CG", "N": "CA", "NE2": "CD"},
        "LYS": {"H": "N", "HA": "CA", "HG2": "CG", "HB2": "CB", "HB3": "CB", "HD2": "CD", "HD3": "CD", "HE2": "CE", "HE3": "CE", "HG3": "CG", "HZ1": "NZ", "HZ2": "NZ", "HZ3": "NZ", "N": "CA", "NZ": "CE"},
        "MET": {"H": "N", "HA": "CA", "HG2": "CG", "HB2": "CB", "HB3": "CB", "HE1": "CE", "HE2": "CE", "HE3": "CE", "HG3": "CG", "N": "CA"},
        "PRO": {"H": "N", "HA": "CA", "HG2": "CG", "HB2": "CB", "HB3": "CB", "HD2": "CD", "HD3": "CD", "HG3": "CG", "N": "CA"},
        "THR": {"H": "N", "HA": "CA", "HG21": "CG2", "HG22": "CG2", "HB": "CB", "HG1": "OG1", "HG23": "CG2", "N": "CA"},
        "ALA": {"H": "N", "HA": "CA", "HB1": "CB", "HB2": "CB", "HB3": "CB", "N": "CA"},
        "ASP": {"H": "N", "HA": "CA", "HB2": "CB", "HB3": "CB", "HD2": "OD2", "N": "CA"},
        "ASN": {"H": "N", "HA": "CA", "HB2": "CB", "HB3": "CB", "HD21": "ND2", "HD22": "ND2", "N": "CA", "ND2": "CG"},
        "CYS": {"H": "N", "HA": "CA", "HB2": "CB", "HB3": "CB", "HG": "SG", "N": "CA"},
        "HIS": {"H": "N", "HA": "CA", "HB2": "CB", "HB3": "CB", "HD1": "ND1", "HD2": "CD2", "HE1": "CE1", "HE2": "NE2", "N": "CA", "NE2": "CD2", "ND1": "CG"},
        "LEU": {"H": "N", "HA": "CA", "HB2": "CB", "HB3": "CB", "HD11": "CD1", "HD12": "CD1", "HD13": "CD1", "HD21": "CD2", "HD22": "CD2", "HD23": "CD2", "HG": "CG", "N": "CA"},
        "PHE": {"H": "N", "HA": "CA", "HB2": "CB", "HB3": "CB", "HD1": "CD1", "HD2": "CD2", "HE1": "CE1", "HE2": "CE2", "HZ": "CZ", "N": "CA"},
        "SER": {"H": "N", "HA": "CA", "HB2": "CB", "HB3": "CB", "HG": "OG", "N": "CA"},
        "TYR": {"H": "N", "HA": "CA", "HB2": "CB", "HB3": "CB", "HD1": "CD1", "HD2": "CD2", "HE1": "CE1", "HE2": "CE2", "HH": "OH", "N": "CA"}}
        source_atoms_dict = {'GLY': {'O': -0.568, 'N': -0.416, 'C': 0.597, 'H': 0.272, 'CA': -0.025, 'HA2': 0.07, 'HA2': 0.07},
    'ILE': {'O': -0.568, 'N': -0.416, 'C': 0.597, 'H': 0.275, 'CA':  0.060, 'CB':  0.130, 'HA': 0.087, 'HB': 0.019, 'CG2': -0.320, 'HG21': 0.088, 'HG22': 0.088, 'HG23': 0.088, 'CG1': -0.043, 'HG12': 0.024, 'HG13': 0.024, 'CD1': -0.066, 'HD11': 0.019, 'HD12': 0.019, 'HD13': 0.019},
    'VAL': {'O': -0.568, 'N': -0.416, 'C': 0.597, 'H': 0.272, 'CA': -0.088, 'CB':  0.299, 'HA': 0.097, 'HB': -0.030, 'CG1': -0.319, 'HG11': 0.079, 'HG12': 0.079, 'HG13': 0.079, 'CG2': -0.319, 'HG21': 0.079, 'HG22': 0.079, 'HG23': 0.079},
    'ARG': {'O': -0.589, 'N': -0.348, 'C': 0.734, 'H': 0.275, 'CA': -0.264, 'CB': -0.001, 'HA': 0.156, 'HB2': 0.033, 'HB3': 0.033, 'CG': 0.039, 'HG2': 0.029, 'HG3': 0.029, 'CD': 0.049, 'HD2': 0.069, 'HD3': 0.069, 'NE': -0.530, 'HE': 0.346, 'CZ': 0.808, 'NH1': -0.823, 'HH11': 0.448, 'HH12': 0.448, 'NH2': -0.823, 'HH21': 0.448, 'HH22': 0.448},
    'TRP': {'O': -0.568, 'N': -0.416, 'C': 0.597, 'H': 0.272, 'CA': -0.028, 'CB': -0.005, 'HA': 0.112, 'HB2': 0.034, 'HB3': 0.034, 'CG': -0.142, 'CD1': -0.164, 'HD1': 0.206, 'NE1': -0.342, 'HE1': 0.341, 'CE2': 0.138, 'CZ2': -0.260, 'HZ2': 0.157, 'CH2': -0.113, 'HH2': 0.142, 'CZ3': -0.197, 'HZ3': 0.145, 'CE3': -0.239, 'HE3': 0.170, 'CD2': 0.124},
    'GLU': {'O': -0.582, 'N': -0.416, 'C': 0.537, 'H': 0.272, 'CA':  0.040, 'CB':  0.056, 'HA': 0.085, 'HB2': 0.017, 'HB3': 0.017, 'CG': 0.014, 'HG2': 0.035, 'HG3': 0.035, 'CD': 0.805, 'OE1': -0.819, 'OE2': -0.819},
    'GLN': {'O': -0.568, 'N': -0.416, 'C': 0.597, 'H': 0.272, 'CA': -0.003, 'CB': -0.004, 'HA': 0.085, 'HB2': 0.017, 'HB3': 0.017, 'CG': -0.065, 'HG2': 0.035, 'HG3': 0.035, 'CD': 0.695, 'OE1': -0.609, 'NE2': -0.941, 'HE21': 0.425, 'HE22': 0.425},
    'LYS': {'O': -0.589, 'N': -0.416, 'C': 0.734, 'H': 0.272, 'CA': -0.240, 'CB': -0.009, 'HA': 0.143, 'HB2': 0.036, 'HB3': 0.036, 'CG': 0.019, 'HG2': 0.010, 'HG3': 0.010, 'CD': -0.048, 'HD2': 0.062, 'HD3': 0.062, 'CE': -0.014, 'HE2': 0.114, 'HE3': 0.114, 'NZ': -0.385, 'HZ1': 0.340, 'HZ2': 0.340, 'HZ3': 0.340},
    'MET': {'O': -0.568, 'N': -0.416, 'C': 0.597, 'H': 0.272, 'CA': -0.024, 'CB':  0.003, 'HA': 0.088, 'HB2': 0.024, 'HB3': 0.024, 'CG': 0.002, 'HG2': 0.044, 'HG3': 0.044, 'SD': -0.274, 'CE': -0.054, 'HE1': 0.068, 'HE2': 0.068, 'HE3': 0.068},
    'PRO': {'O': -0.548, 'N': -0.255, 'C': 0.590, 'H': 0.272, 'CA': -0.027, 'CB': -0.007, 'HA': 0.064, 'CD': 0.019, 'HD2': 0.039, 'HD3': 0.039, 'CG': 0.019, 'HG2': 0.019, 'HG3': 0.019, 'HB2': 0.025, 'HB3': 0.025},
    'THR': {'O': -0.568, 'N': -0.416, 'C': 0.597, 'H': 0.272, 'CA': -0.039, 'CB':  0.365, 'HA': 0.101, 'HB': 0.004, 'OG1': -0.676, 'HG1': 0.410, 'CG2': -0.244, 'HG21': 0.064, 'HG22': 0.064, 'HG23': 0.064},
    'ALA': {'O': -0.568, 'N': -0.416, 'C': 0.597, 'H': 0.272, 'CA':  0.034, 'CB': -0.183, 'HA': 0.082, 'HB1': 0.060, 'HB2': 0.060, 'HB3': 0.060},
    'ASP': {'O': -0.582, 'N': -0.516, 'C': 0.537, 'H': 0.294, 'CA':  0.038, 'CB': -0.030, 'HA': 0.088, 'HB2': -0.012, 'HB3': -0.012, 'CG': 0.799, 'OD1': -0.801, 'OD2': -0.801},
    'ASN': {'O': -0.568, 'N': -0.416, 'C': 0.597, 'H': 0.272, 'CA':  0.014, 'CB': -0.204, 'HA': 0.105, 'HB2': 0.080, 'HB3': 0.080, 'CG': 0.713, 'OD1': -0.593, 'ND2': -0.919, 'HD21': 0.420, 'HD22': 0.420},
    'CYS': {'O': -0.568, 'N': -0.416, 'C': 0.597, 'H': 0.272, 'CA':  0.021, 'CB': -0.123, 'HA': 0.112, 'HB2': 0.112, 'HB3': 0.112, 'SG': -0.3112, 'HG': 0.193},
    'HIS': {'O': -0.568, 'N': -0.416, 'C': 0.597, 'H': 0.272, 'CA': -0.058, 'CB': -0.007, 'HA': 0.136, 'HB2': 0.037, 'HB3': 0.037, 'CG': 0.187, 'ND1': -0.543, 'HD1': 0.334, 'CE1': 0.164, 'HE1': 0.144, 'NE2': -0.280, 'HE2': 0.334, 'CD2': -0.221, 'HD2': 0.186},
    'LEU': {'O': -0.568, 'N': -0.416, 'C': 0.597, 'H': 0.272, 'CA': -0.052, 'CB': -0.110, 'HA': 0.092, 'HB2': 0.046, 'HB3': 0.046, 'CG': 0.353, 'HG': -0.036, 'CD1': -0.412, 'HD11': 0.1, 'HD12': 0.1, 'HD13': 0.1, 'CD2': -0.412, 'HD21': 0.1, 'HD22': 0.1, 'HD23': 0.1},
    'PHE': {'O': -0.568, 'N': -0.416, 'C': 0.597, 'H': 0.272, 'CA': -0.002, 'CB': -0.034, 'HA': 0.098, 'HB2': 0.03, 'HB3': 0.03, 'CG': 0.012, 'CD1': -0.126, 'HD1': 0.133, 'CE1': -0.170, 'HE1': 0.143, 'CZ': -0.107, 'HZ': 0.130, 'CE2': -0.170, 'HE2': 0.143, 'CD2': -0.126, 'HD2': 0.133},
    'SER': {'O': -0.568, 'N': -0.416, 'C': 0.597, 'H': 0.272, 'CA': -0.025, 'CB':  0.212, 'HA': 0.084, 'HB2': 0.035, 'HB3': 0.035, 'OG': -0.655, 'HG': 0.428},
    'TYR': {'O': -0.568, 'N': -0.416, 'C': 0.597, 'H': 0.272, 'CA': -0.001, 'CB': -0.015, 'HA': 0.088, 'HB2': 0.030, 'HB3': 0.030, 'CG': -0.001, 'CD1': -0.191, 'HD1': 0.170, 'CE1': -0.234, 'HE1': 0.166, 'CZ': 0.3226, 'OH': -0.558, 'HH': 0.399, 'CE2': -0.234, 'HE2': 0.166, 'CD2': -0.191, 'HD2': 0.170}}
    
    
        averaged_methyl = {'LYS':{'HZ_EFIELD':['HZ1_EFIELD', 'HZ2_EFIELD', 'HZ3_EFIELD']}, 'ALA':{'HB_EFIELD':['HB1_EFIELD','HB2_EFIELD','HB3_EFIELD']}, 'ILE':{'HD1_EFIELD':['HD11_EFIELD', 'HD12_EFIELD', 'HD13_EFIELD'], 'HG2_EFIELD':['HG21_EFIELD', 'HG22_EFIELD', 'HG23_EFIELD']}, 'LEU':{'HD1_EFIELD':['HD11_EFIELD', 'HD12_EFIELD', 'HD13_EFIELD'], 'HD2_EFIELD':['HD21_EFIELD', 'HD22_EFIELD', 'HD23_EFIELD']}, 'MET':{'HE_EFIELD':['HE1_EFIELD','HE2_EFIELD','HE3_EFIELD']}, 'THR':{'HG2_EFIELD':['HG21_EFIELD','HG22_EFIELD','HG23_EFIELD']}, 'VAL':{'HG1_EFIELD':['HG11_EFIELD','HG12_EFIELD','HG13_EFIELD'], 'HG2_EFIELD':['HG21_EFIELD','HG22_EFIELD','HG23_EFIELD']}}
        
        
        res_electricfield_dict = {}
        for res in res_object_list:
            resnum = res.get_id()[1]
            target_atoms = [i for i in res.get_atoms() if i.get_id() in atom_names_target]
            for atom in target_atoms:
    
                electricfield_at = 0

                #get coords and IDs of atoms within rad
                source_coords = [[j.get_id(), j.get_coord(), j.get_parent().get_resname()] for j in nn_tree.search(atom.get_coord(), rad)]
                # remove those that are not source atoms
                source_atoms = list({key for sub_dict in source_atoms_dict.values() if isinstance(sub_dict, dict) for key in sub_dict.keys()})
                source_coords = [sublist for sublist in source_coords if sublist[0] in source_atoms]
     
                #get coords of the targetatom
                target_coords = atom.get_coord()
    
                # get coords of its partner
                resname_str = res.get_resname()
    
                if atom.get_id() in partners[resname_str]:
                    partner = partners[resname_str][atom.get_id()]
                    try:
                        partner_coords = [j.get_coord() for j in res.get_atoms() if j.get_id() == partner][0]
                    except IndexError:
                        electricfield_at += 0
                        pass

                    for j in source_coords:
                        ba = partner_coords - target_coords
                        bc = j[1] - target_coords
                        if np.linalg.norm(bc) > 0.5:
                            try:
                                cos_theta = np.dot (ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                                d2 = (np.linalg.norm(bc))**2
                                q = source_atoms_dict[j[2]][j[0]]
                                electricfield_at += cos_theta*q/d2
                            except KeyError:
                                electricfield_at += 0
                else:
                    electricfield_at += 0
    
                if resnum in res_electricfield_dict.keys():
                    res_electricfield_dict[resnum][atom.get_id()+'_EFIELD']=electricfield_at
                else:
                    res_electricfield_dict[resnum]={}
                    res_electricfield_dict[resnum][atom.get_id()+'_EFIELD']=electricfield_at
    
            if res.get_resname() in averaged_methyl:                
                for res_met in averaged_methyl.keys():
                    if res_met == res.get_resname():
                        for atom_met, atoms_met in averaged_methyl[res_met].items():
                            try:
                                average_electricfield = (res_electricfield_dict[resnum][atoms_met[0]] + res_electricfield_dict[resnum][atoms_met[1]] + res_electricfield_dict[resnum][atoms_met[2]])/3
                                res_electricfield_dict[resnum][atom_met] = average_electricfield
                                del res_electricfield_dict[resnum][atoms_met[0]]
                                del res_electricfield_dict[resnum][atoms_met[1]]
                                del res_electricfield_dict[resnum][atoms_met[2]]
                            except KeyError:
                                pass
    
            if len(res_electricfield_dict) != 0:
                included_atoms = []
                for i in res_electricfield_dict.get(resnum, {}).keys():
                    if '_' in i:
                        included_atoms.append(i.split('_')[0])
                remaining_atoms = [i for i in atom_names_target if i not in included_atoms]
                for a in remaining_atoms:
                    if resnum in res_electricfield_dict:
                        res_electricfield_dict[resnum][a+'_EFIELD'] = np.nan

        return res_electricfield_dict




    def blosum_nums(self, res_obj):
        '''
        Get blosum62 block substitution matrix comparison numbers

        Args:
            res_obj - The residue that is being compared (Bio.PDB.Residue)

        Returns:
            nums - The 20 blosum62 comparison numbers (list of floats) or [0]*20 if residue not found (np.Array of Int)
       '''

        # First need to convert blosum62 dictionary to 3-letter codes
        cap_1to3 = dict((i, j.upper()) for i, j in IUPACData.protein_letters_1to3.items())
        std_blosum62 = dict([(cap_1to3[i[0]], cap_1to3[i[1]]), j] for i,j in blosum62.items()
        if i[0] not in ['Z', 'B', 'X'] and i[1] not in ['Z', 'B', 'X'])


        res_name = self._fix_res_name(res_obj.resname)
        nums = []
        for key in sorted(std_blosum62):
            if key[0]==res_name or key[1]==res_name:
                nums.append(std_blosum62[key])
        return nums


    def binary_seqvecs(self, res_obj):
        '''
        Get a binary vector indicating which of the 20 standard amino acids the residue is

        Args:
            res_obj - The residue that is being classified (Bio.PDB.Residue)

        Returns:
            out - A list of 20 integers all but one of which are zeros and the remaining is a one. The position of the 1 element indicates which of the 20 standard amino acids the residue is.  If the residue is not one of the standard 20, returns [0]*20 (list of Int)
        '''

        res_name = self._fix_res_name(res_obj.resname)
        out = 0*[20]
        for i, aa in enumerate(paper_order):
            if res_name == aa:
                out[i] = 1
        return out


    @staticmethod
    def calc_phi_psi(chain_obj):
        '''
        Get list of dihedral phi and psi angles

        Args:
            chain_obj - A chain of residues for which to calculate the dihedral angles (Bio.PDB.Chain)

        Returns:
            fin_list - A list of angle parameters[cos(phi), sin(phi), cos(psi), sin(psi)] with [0]*4 for the absence of an angle (List of Float)
        '''

        polys = PDB.PPBuilder(radius=2.1).build_peptides(chain_obj)
        phi_psi_list = []
        full_res_list = []
        for poly in polys:
            phi_psi_list.append(poly.get_phi_psi_list())
            for res in poly:
                full_res_list.append(res)

        out_list = []
        for peps in phi_psi_list:
            for ang_pair in peps:
                if ang_pair[0] == None:
                    phi_dat = [0, 0]
                else:
                    phi_dat = [math.cos(ang_pair[0]), math.sin(ang_pair[0])]
                if ang_pair[1] == None:
                    psi_dat =[0, 0]
                else:
                    psi_dat = [math.cos(ang_pair[1]), math.sin(ang_pair[1])]
                out_list.append(phi_dat + psi_dat)
        fin_list = [out_list[i] for i in range(len(out_list)) if (full_res_list[i].get_id()[2] == ' ')]
        return fin_list


    def calc_torsion_angles(self, res_obj, chi=[1,2,3,4,5,6,7]):
        '''
        Get list of chi angles

        Args:
            res_obj - A residue for which to calculate (Bio.PDB.Residue)
            torsion angles
            chi - A list of integers in range(1,6) that specify which chi angles to calculate.  Default is [1, 2] (List of Int)

        Returns:
            chi_list - The sum of lists of torsion angle parameters [cos(chi), sin(chi), 0/1], with the boolean indicating existence of the angle, for each of the chi
        '''

        # First need to construct a dictionary mapping the residue
        # to the atoms defining each of the chi's

        chi_defs = dict(
            chi1=dict(
                ARG=['N', 'CA', 'CB', 'CG'],
                ASN=['N', 'CA', 'CB', 'CG'],
                ASP=['N', 'CA', 'CB', 'CG'],
                CYS=['N', 'CA', 'CB', 'SG'],
                GLN=['N', 'CA', 'CB', 'CG'],
                GLU=['N', 'CA', 'CB', 'CG'],
                HIS=['N', 'CA', 'CB', 'CG'],
                ILE=['N', 'CA', 'CB', 'CG1'],
                LEU=['N', 'CA', 'CB', 'CG'],
                LYS=['N', 'CA', 'CB', 'CG'],
                MET=['N', 'CA', 'CB', 'CG'],
                PHE=['N', 'CA', 'CB', 'CG'],
                PRO=['N', 'CA', 'CB', 'CG'],
                SER=['N', 'CA', 'CB', 'OG'],
                THR=['N', 'CA', 'CB', 'OG1'],
                TRP=['N', 'CA', 'CB', 'CG'],
                TYR=['N', 'CA', 'CB', 'CG'],
                VAL=['N', 'CA', 'CB', 'CG1'],
            ),
            altchi1=dict(
                VAL=['N', 'CA', 'CB', 'CG2'],
            ),
            chi2=dict(
                ARG=['CA', 'CB', 'CG', 'CD'],
                ASN=['CA', 'CB', 'CG', 'OD1'],
                ASP=['CA', 'CB', 'CG', 'OD1'],
                GLN=['CA', 'CB', 'CG', 'CD'],
                GLU=['CA', 'CB', 'CG', 'CD'],
                HIS=['CA', 'CB', 'CG', 'ND1'],
                ILE=['CA', 'CB', 'CG1', 'CD1'],
                LEU=['CA', 'CB', 'CG', 'CD1'],
                LYS=['CA', 'CB', 'CG', 'CD'],
                MET=['CA', 'CB', 'CG', 'SD'],
                PHE=['CA', 'CB', 'CG', 'CD1'],
                PRO=['CA', 'CB', 'CG', 'CD'],
                TRP=['CA', 'CB', 'CG', 'CD1'],
                TYR=['CA', 'CB', 'CG', 'CD1'],
            ),
            altchi2=dict(
                ASP=['CA', 'CB', 'CG', 'OD2'],
                LEU=['CA', 'CB', 'CG', 'CD2'],
                PHE=['CA', 'CB', 'CG', 'CD2'],
                TYR=['CA', 'CB', 'CG', 'CD2'],
            ),
            chi3=dict(
                ARG=['CB', 'CG', 'CD', 'NE'],
                GLN=['CB', 'CG', 'CD', 'OE1'],
                GLU=['CB', 'CG', 'CD', 'OE1'],
                LYS=['CB', 'CG', 'CD', 'CE'],
                MET=['CB', 'CG', 'SD', 'CE'],
            ),
            chi4=dict(
                ARG=['CG', 'CD', 'NE', 'CZ'],
                LYS=['CG', 'CD', 'CE', 'NZ'],
            ),
            chi5=dict(
                ARG=['CD', 'NE', 'CZ', 'NH1'],
            ),
        )

        # Next construct a list of the names of desired angles
        chi_names = []
        for x in chi:
            reg_chi = "chi%s" % x
            if reg_chi in chi_defs.keys():
                chi_names.append(reg_chi)
    #            This can be activated to implement alt_chi defs
                alt_chi = "altchi%s" % x
                if alt_chi in chi_defs.keys():
                    chi_names.append(alt_chi)
            else:
                pass


        # Construct list of triples that define the requested chi angles
        # The triples are cos, sin, existence with [0,0,0] convention for nan
        res_name = self._fix_res_name(res_obj.resname)
        chi_list = []
        for chis in chi_names:
            # Skip heteroatoms...this may need to be modified or expanded
            if res_obj.id[0] != " ":
                chi_list = [0]*3*len(chi)
                break
            chi_res = chi_defs[chis]
            try:
                atom_list = chi_res[res_name]
                try:
                    vec_atoms = [res_obj[i] for i in atom_list]
                    vectors = [i.get_vector() for i in vec_atoms]
                    angle = PDB.calc_dihedral(*vectors)
                    chi_list += [math.cos(angle), math.sin(angle), 1]
                except KeyError:
                    chi_list += [0, 0, 0]
            except KeyError:
                chi_list += [0, 0, 0]

        return chi_list








    @staticmethod
    def s2_param(nn_tree, res_obj, chain, rad=10.0, b_param=-0.1):
        '''
        Calculate S2 order parameter of N-H bond based on the contact model put forth in Zhang, Bruschweiler (2002) J. Am. Chem. Soc 124

        Args:
            nn_tree - A tree containing the distances between atoms in the set of interest (Bio.PDB.NeighborSearch)
            res_obj - The residue for which we are calculating S2 (Bio.PDB.Residue)
            prev_Oatom - The Oxygen atom for the previous residue in the chain (Bio.PDB.Atom)
            rad - The radius within which to search for heavy atoms to include in the contact model.  Note that, in principle, we could use different radii for the rOsum and rHsum in the model (Float)
            b_param - A parameter in the model (Float)

        Returns:
            S2 - The S2 order parameter for the N-H bond of res_obj (Float)
        '''

        rOsum = 0
        rHsum = 0

        try:
            prev_res = chain[res_obj.get_id()[1]-1]
        except KeyError:
            prev_res = None


        if prev_res != None:
            try:
                prev_Oatom = prev_res['O']
            except KeyError:
                prev_Oatom = None
        else:
            prev_Oatom = None


        if prev_Oatom != None:
            nn_atoms = nn_tree.search(prev_Oatom.get_coord(), rad, level='A')
            for atom in nn_atoms:
                if ('H' not in atom.get_name()) and (atom - prev_Oatom) > 0.1:
                    at_res = atom.get_parent()
                    if (at_res != res_obj) and (at_res != prev_res):
                        rOsum += math.exp(-(atom - prev_Oatom - 1.2))
                else:
                    pass
        else:
            pass

        for at in ['H', '1H']:
            try:
                Hatom = res_obj[at]
                nn_atoms = nn_tree.search(Hatom.get_coord(), rad, level='A')
                for atom in nn_atoms:
                    if ('H' not in atom.get_name()) and (atom - Hatom) > 0.1:
                        at_res = atom.get_parent()
                        if (at_res != res_obj) and (at_res != prev_res):
                            rHsum += math.exp(-(atom - Hatom - 1.2))
                    else:
                        pass
                break
            except KeyError:
                pass

        S2 = math.tanh(0.8 * (rOsum + rHsum)) + b_param
        return S2


    @staticmethod
    def find_nearest_atom(nn_tree, catom, rad=3.5, inrad=0.5, atom_type='Any', excl=[], atom_name = False):
        '''
        Find the nearest atom of type atom_type within radius rad of a central atom catom. Note that the function finds the closest atom that contains the atom_type string in its name rather than relying on an exact match.

        Args:
            nn_tree - A tree containing the distances between atoms in the set of interest (Bio.PDB.NeighborSearch)
            catom - Central atom for which we wish to find the nearest neighbor (Bio.PDB.Atom)
            rad - The distance within which to search (Float)
            atom_type - The type of atom for which to search. Default is to search for any atom (Str)
            excl - List of atom objects to exclude from the search(List of Bio.PDB.Atom)

        Returns:
            nn_atom - The nearest atom_type atom within rad of catom.  Returns None if no such atom found within rad of catom (Bio.PDB.Atom)
        '''

        coord = catom.get_coord()
        atoms = nn_tree.search(coord, rad)
        dists = []
        for atom in atoms:
            dists.append(atom-catom)
        # dists = [atom - catom for atom in atoms]
        
        # If only a single atom within rad, return that info now
        if len(dists) == 1:
        #            print('There are no other atoms within rad of catom')
            return None
        
        # Initializae min distance and corresponding index with upper bounds
        # Will use the possible IndexError to indicate no atoms found
        
        d_min = rad
        loc = 10**10
        
        # If no specified type, just find the atom with min distance but
        # remember to exclude the atom at coord (we use a 0.1 exclusion ball)
        if atom_type == 'Any':
            for idx, dist in enumerate(dists):
                if dist < d_min and dist > inrad and dist > 0.1:
                    at = atoms[idx]
                    if at not in excl:
                        d_min = dist
                        loc = idx
            try:
                nn_atom = atoms[loc]
                return nn_atom
            except IndexError:
        #                print('There are no atoms within rad of coord')
                return None
        # If type specified, need to check that it is in atom name
        # If none found, return None
        elif type(atom_type) == list:
            for idx, dist in enumerate(dists):
                at = atoms[idx]
                #print(at.get_name())
                atom_elem = at.element
                if dist < d_min and dist > inrad and dist > 0.1:
                    for atype in atom_type:
                        if atom_name:
                            if (atype == at.get_name()) and (at not in excl):
                                d_min = dist
                                loc = idx
                        else:
                            if (atype == atom_elem) and (at not in excl):
                                d_min = dist
                                loc = idx
            try:
                nn_atom = atoms[loc]
                return nn_atom
            except IndexError:
        #                print('There are no atoms within rad of coord')
                return None
            
        else:
            for idx, dist in enumerate(dists):
                atom_elem = atoms[idx].element
                if dist < d_min and dist > inrad and dist > 0.1 and (atom_type == atom_elem):
                    at = atoms[idx]
                    if at not in excl:
                        d_min = dist
                        loc = idx
            try:
                nn_atom = atoms[loc]
                return nn_atom
            except IndexError:
        #                print('There are no atoms of atom_type within rad of coord')
                return None


    @staticmethod
    def find_nxtnearest_Atom(nn_tree, catom, rad, atom_type='Any', excl=[]):
        '''
        Find the next nearest atom of type atom_type within radius rad of a central atom catom. Note that the function finds the closest atom that
        contains the atom_type string in its name rather than relying on an exact match.

        Args:
            nn_tree - A tree containing the distances between atoms in the set of interest (Bio.PDB.NeighborSearch)
            catom - Central atom for which we wish to find the next nearest neighbor (Bio.PDB.Atom)
            rad - The distance within which to search (Float)
            atom_type - The type of atom for which to search. Default is to search for any atom (Str)
            excl - List of atom objects to exclude from the search (List of Bio.PDB.Atom)

        Returns:
            nxtnn_atom - The next nearest atom_type atom within rad of catom.  Returns None if no such atom found within rad of catom (Bio.PDB.Atom)
        '''
# SHOULD UPDATE THIS FUNCTION TO LOOK AT ATOMIC ELEMENTS RATHER THAN ATOM NAMES AS WELL AS TO ACCEPT LISTS FOR ATOM_TYPE KWARG
        coord = catom.get_coord()
        atoms = nn_tree.search(coord, rad)
        dists = []
        for atom in atoms:
            dists.append(atom-catom)

    #    If only a single atom within rad, return that info now
        if len(dists) < 3:
#            print('There are fewer than two atoms within rad of coord')
            return None#
        d_min, d_2min = rad, rad
        min_loc, min_loc2 = 10**10, 10**10

        if atom_type == 'Any':
            for idx, dist in enumerate(dists):
                if dist < d_min and dist > 0.1:
                    d_2min = d_min
                    d_min = dist
                    min_loc2 = min_loc
                    min_loc = idx
                elif dist < d_2min and dist > d_min:
                    d_2min = dist
                    min_loc2 = idx
            nxtnn_atom = atoms[min_loc2]
            return nxtnn_atom
        else:
            for idx, dist in enumerate(dists):
                atom_name = atoms[idx].name
                if dist < d_min and dist > 0.1 and (atom_type in atom_name):
                    d_2min = d_min
                    d_min = dist
                    min_loc2 = min_loc
                    min_loc = idx
                elif dist < d_2min and dist > d_min and (atom_type in atom_name):
                    d_2min = dist
                    min_loc2 = idx
            try:
                nxtnn_atom = atoms[min_loc2]
                return nxtnn_atom
            except IndexError:
#                print('There are fewer than two atoms of atom_type within rad of coord')
                return None



    def calc_ring_currents(self,res_object_list):

        '''
        Takes list of residues and returns a dictionary of atom specific ring current shifts with residue numbers as keys. Note that this assumes that residues in the list have unique numbers.

        Args:
            res_object_list - List of Residue objects that may contain rings (List of Bio.PDB.Residue)

        Returns:
            res_ring_shift_dict - Dictionary where keys are atoms in the residues in the given list and values are corresponding ring current contributions to the chemical shifts (Dict)
        '''

        #Make sure all residues have unique numbers
        res_numbers = [i.get_id()[1] for i in res_object_list]
        if len(res_numbers)>len(set(res_numbers)):
            raise Exception('Not all residue numbers are unique in calc_ring_currents res_object_list')


        #Parameters from shiftX
        atom_names_target = ['N', 'HA', 'HA2', 'HA3', 'H', '1H', '1HA', '2HA','CG','CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CG1', 'CG2', 'CZ','HB', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE21', 'HE22', 'HE3', 'HG', 'HG1', 'HG12', 'HG13', 'HG2', 'HG3', 'HZ','CE3','CZ3','HZ3','CH2','HH2','CZ2','HZ2', 'HB1', 'HD11', 'HD12', 'HD13', 'HD23', 'HG11', 'HZ1', 'HG21', 'HG22', 'HG23','ND2','NE1','NE2','C','CA','CB']

        intensity_factors = {'PHE': 1.05, 'TYR': 0.92, 'TRP1': 1.04, 'TRP2': 0.90, 'HIS': 0.43}
        target_factors = {'HA': 5.45, 'HA2': 5.45, 'HA3': 5.45, 'H': 7.06, 'CA': 1.5, 'CB': 1.00, 'N': 1.00, 'C': 1.00, '1HA' : 5.13, '2HA' : 5.13, '1H' : 7.06,'CG':1, 'CD': 1, 'CD1': 1, 'CD2': 1, 'CE': 1, 'CE1': 1, 'CE2': 1, 'CG1': 1, 'CG2': 1, 'CZ': 1, 'HB': 5.45, 'HB2': 5.45, 'HB3': 5.45, 'HD1': 5.45, 'HD2': 5.45, 'HD21': 5.45, 'HD22': 5.45, 'HD3': 5.45, 'HE': 5.45, 'HE1': 5.45, 'HE2': 5.45, 'HE21': 5.45, 'HE22': 5.45, 'HE3': 5.45, 'HG': 5.45, 'HG1': 5.45, 'HG12': 5.45, 'HG13': 5.45, 'HG2': 5.45, 'HG3': 5.45, 'HZ': 5.45, 'CE3': 5.45, 'CZ3': 5.45, 'HZ3': 5.45, 'CH2': 5.45, 'HH2': 5.45, 'CZ2': 5.45, 'HZ2': 5.45, 'HB1': 5.45, 'HD11': 5.45, 'HD12': 5.45, 'HD13': 5.45, 'HD23': 5.45, 'HG11': 5.45, 'HZ1': 5.45, 'HG21': 5.45, 'HG22': 5.45, 'HG23': 5.45, 'ND2':1.0, 'NE1':1.0 ,'NE2':1.0}

        #list of atoms that can be a part of the ring
        ring_atoms = {'PHE': ['CB','CD1','CD2','CE1','CE2','CG','CZ','HD1','HD2','HE1','HE2','HZ'], 'TYR': ['CB','CD1','CD2','CE1','CE2','CG','CZ','HD1','HD2','HE1','HE2'],
              'TRP': ['CB','CD1','CD2','CE2','CG','HD1','HE1','HE3','CE3','CZ3','HZ3','CH2','HH2','CZ2','HZ2', 'NE1'], 'HIS': ['CB','CD1','CD2','CE1','CE2','CG','HD1','HE1','HE2','CE3','CZ3','HZ3','CH2','HH2','CZ2','HZ2', 'ND1', 'NE2']}


        averaged_methyl = {'LYS':{'HZ_RING':['HZ1_RING', 'HZ2_RING', 'HZ3_RING']}, 'ALA':{'HB_RING':['HB1_RING','HB2_RING','HB3_RING']}, 'ILE':{'HD1_RING':['HD11_RING', 'HD12_RING', 'HD13_RING'], 'HG2_RING':['HG21_RING', 'HG22_RING', 'HG23_RING']}, 'LEU':{'HD1_RING':['HD11_RING', 'HD12_RING', 'HD13_RING'], 'HD2_RING':['HD21_RING', 'HD22_RING', 'HD23_RING']}, 'MET':{'HE_RING':['HE1_RING','HE2_RING','HE3_RING']}, 'THR':{'HG2_RING':['HG21_RING','HG22_RING','HG23_RING']}, 'VAL':{'HG1_RING':['HG11_RING','HG12_RING','HG13_RING'], 'HG2_RING':['HG21_RING','HG22_RING','HG23_RING']}}


        #Get list of rings and associated coordinates
        rings = []
        for res in res_object_list:
            try:
                if (res.get_resname() == 'PHE'):
                    rings += [['PHE',(res['CG'].get_coord(), res['CD2'].get_coord(), res['CE2'].get_coord(), res['CZ'].get_coord(), res['CE1'].get_coord(), res['CD1'].get_coord())]]
                elif (res.get_resname()=='TYR'):
                    rings += [['TYR',(res['CG'].get_coord(), res['CD2'].get_coord(), res['CE2'].get_coord(), res['CZ'].get_coord(), res['CE1'].get_coord(), res['CD1'].get_coord())]]
                elif (res.get_resname() == 'TRP'):
                    rings += [['TRP1',(res['CD2'].get_coord(), res['CE3'].get_coord(), res['CZ3'].get_coord(), res['CH2'].get_coord(), res['CZ2'].get_coord(), res['CE2'].get_coord())]]
                    rings += [['TRP2',(res['CG'].get_coord(), res['CD2'].get_coord(), res['CE2'].get_coord(), res['NE1'].get_coord(), res['CD1'].get_coord())]]
                elif (res.get_resname() == 'HIS'):
                    rings += [['HIS',(res['CG'].get_coord(), res['ND1'].get_coord(), res['CE1'].get_coord(), res['NE2'].get_coord(), res['CD2'].get_coord())]]
            except:
                #Every once in a while a sidechain will be missing. These res will be excluded in the end.
                print('error on ring')


        #Calculate contribution from each ring on each atom
        res_ring_shift_dict = {}
        for res in res_object_list:
            resnum = res.get_id()[1]
            target_atoms = [i for i in res.get_atoms() if i.get_id() in atom_names_target]
            for atom in target_atoms:
                shift = 0
                for ring in rings:
                    if ((ring[0] == res.get_resname()) and (atom.get_id() in ring_atoms[res.get_resname()])):
                        shift = 0
                    elif ((ring[0][:-1] == res.get_resname()) and (atom.get_id() in ring_atoms[res.get_resname()])): #for Tyr
                        shift = 0
                    else:
                        G = 0
                        ring_coords = ring[1]
                        normal = np.cross(ring_coords[1]-ring_coords[0], ring_coords[-1]-ring_coords[0])
                        normal = normal/np.linalg.norm(normal)
                        o = atom.get_coord() + np.dot(normal, ring_coords[0]-atom.get_coord())*normal
                        for i in range(len(ring_coords)):
                            if (i == (len(ring_coords)-1)):
                                r_i = ring_coords[i] - o
                                r_j = ring_coords[0] - o
                                d_r_i = ring_coords[i] - atom.get_coord()
                                d_r_j = ring_coords[0] - atom.get_coord()
                                area_ij = np.linalg.norm(np.cross(r_i, r_j))/2
                                sign = np.sign( np.dot(np.cross(r_i, r_j), normal) )
                                area_ij = sign*area_ij
                                d_ij = 1/(np.linalg.norm(d_r_i)**3) + 1/(np.linalg.norm(d_r_j)**3)
                            else:
                                r_i = ring_coords[i] - o
                                r_j = ring_coords[i+1] - o
                                d_r_i = ring_coords[i] - atom.get_coord()
                                d_r_j = ring_coords[i+1] - atom.get_coord()
                                area_ij = np.linalg.norm(np.cross(r_i, r_j))/2
                                sign = np.sign( np.dot(np.cross(r_i, r_j), normal) )
                                area_ij = sign*area_ij
                                d_ij = 1/(np.linalg.norm(d_r_i)**3) + 1/(np.linalg.norm(d_r_j)**3)
                            G = G+d_ij*area_ij
                        I = intensity_factors[ring[0]]
                        F = target_factors[atom.get_id()]
                        shift += G*I*F

                if resnum in res_ring_shift_dict.keys():
                    res_ring_shift_dict[resnum][atom.get_id()+'_RING']=shift
                else:
                    res_ring_shift_dict[resnum]={}
                    res_ring_shift_dict[resnum][atom.get_id()+'_RING']=shift


            
            # methyl averaging
            if res.get_resname() in averaged_methyl:    
            
                for res_met in averaged_methyl.keys():
                    if res_met == res.get_resname():
                        for atom_met, atoms_met in averaged_methyl[res_met].items():
                            try:
                                average_ringcurrent = (res_ring_shift_dict[resnum][atoms_met[0]] + res_ring_shift_dict[resnum][atoms_met[1]] + res_ring_shift_dict[resnum][atoms_met[2]])/3
                                res_ring_shift_dict[resnum][atom_met] = average_ringcurrent
                                del res_ring_shift_dict[resnum][atoms_met[0]]
                                del res_ring_shift_dict[resnum][atoms_met[1]]
                                del res_ring_shift_dict[resnum][atoms_met[2]]  
                            except KeyError:
                                pass
            

            included_atoms = [i.split('_')[0] for i in res_ring_shift_dict[resnum].keys()]
            remaining_atoms = [i for i in atom_names_target if i not in included_atoms]
            for a in remaining_atoms:
                res_ring_shift_dict[resnum][a+'_RING'] = np.nan
            if res_ring_shift_dict[resnum]['H_RING'] is np.nan:
                res_ring_shift_dict[resnum]['H_RING'] = res_ring_shift_dict[resnum]['1H_RING']
            if res_ring_shift_dict[resnum]['HA_RING'] is np.nan:
                res_ring_shift_dict[resnum]['HA_RING'] = res_ring_shift_dict[resnum]['1HA_RING']

        
        return res_ring_shift_dict










    def NH_O_bond(self, nn_tree, res_obj, im1_atoms, ip1_atoms, rad, atom0, atom1, at_type, efilt=False):
        res_i_atoms = list(res_obj.get_atoms())
        angle2 = 0.0
        angle1 = 0.0
        excl_at = []
        while (angle2 < (math.pi / 2)) or (angle1 < (math.pi / 2)):
            if atom1.element == 'O': # Finding hydrogen bond for carboxyl oxygen
                full_excl = res_i_atoms + ip1_atoms + excl_at
            else: # Finding hydrogen bond for amide hydrogen
                full_excl = res_i_atoms + im1_atoms + excl_at
            atom2 = self.find_nearest_atom(nn_tree, atom1, rad, atom_type=at_type, excl=full_excl, atom_name = False)
            if atom2 is None:
                return 5*[0] #If there are no atoms of the specified type within rad of atom1 and accounting for exclusions, then return list of zeros for HNbond_params
            excl_at += [atom2]
            angle2 = PDB.calc_angle(atom0.get_vector(), atom1.get_vector(), atom2.get_vector())
            bond_dist = atom1 - atom2
#            Natom = res_obj['N']
#            HNangle2 = PDB.calc_angle(Natom.get_vector(), HNatom.get_vector(), HNOatom.get_vector())
            parent_tree = PDB.NeighborSearch(list(atom2.get_parent().get_atoms()))
            try:
                atom3 = self.find_nearest_atom(parent_tree, atom2, rad, atom_type=['N', 'C', 'O'], atom_name = False)
                #HNOCatom = HNOatom.get_parent()['C']
                angle1 = PDB.calc_angle(atom1.get_vector(), atom2.get_vector(), atom3.get_vector())
                energy = 0.084 * 332 * (1/(atom0 - atom2) + 1/(atom1 - atom3) - 1/(bond_dist) - 1/(atom0 - atom3))
            except AttributeError:
                angle1 = math.pi * 109.5 /180
                energy = 0
                if angle2 >= math.pi /2:
                    return [bond_dist, math.cos(angle1), math.cos(angle2), 1, energy] # Return the parameter list first so that it will not go through energy check
            HNbond_params = [bond_dist, math.cos(angle1), math.cos(angle2), 1, energy]
            if efilt and energy > -0.5:
                angle1 = 0.0 #If filtering by energy, reset one of the angles so that the while condition is not met and the loop will try again with a new atom2 if any candidates remain

        return HNbond_params


    def hbond_network(self, nn_tree, res_obj, rad=3*[5.0], ha_bond='restrictive', efilt=False, efilter_O2=True):
        '''
        Constructs parameters that encode the structure of 3 different Hydrogen bonds that may occur on a given residue. Each possible hydrogen bond is described by 4 parameters: The first is the distance from donor hydrogen to acceptor atom dHA. The second and third numbers are the cosines of the bond angle at the acceptor atom and the donor hydrogen respecctively. The final number is a boolean for the existence of the H-bond. The parameters are returned as a single list in the following order: alpha Hydrogen, Nitrogen Hydrogen, Oxygen.

        Args:
            nn_tree - A tree containing the distances between atoms in the set of interest (Bio.PDB.NeighborSearch)
            res_obj - Residue for which we wish to find the Hydrogen bond parameters (Bio.PDB.Residue)
            rad - The distance within which to search for each of the three types of bonds H_alpha, H_N, and O (List of Float)
            ha_bond - If restrictive, use definition from Sparta+ (definition given in the Wagner, Pardi, Wuthrich article).  If permissive, use a straight geometric definition (Str)
            efilter_O2 - When ha_bond == 'restrictive', use the DSSP energy of the secondary hydrogen bond to the Oxygen in question as a cutoff (Bool)

        Returns:
            output - A sum of lists [dHA, Cos(phi), Cos(psi), 1] containing the bond parameters or [0, 0, 0, 0] for each of 3 possible Hydrogen bonds. The order of parameters is alpha Hydrogen, Nitrogen Hydrogen, Oxygen (List of Float)
        '''

        # First initialize all bonds as nonexistent
        HAbond_Params = [0]*5
        HNbond_Params = [0]*5
        Obond_Params = [0]*5

        # Define atoms
        for at in ['HA', '1HA']:
            try:
                HAatom = res_obj[at]
                break
            except KeyError:
                HAatom = None
        for at in ['H', '1H', 'H1', 'D', '1D', 'D1']:
            try:
                HNatom = res_obj[at]
                break
            except KeyError:
                HNatom = None
        try:
            Oatom = res_obj['O']
        except KeyError:
            Oatom = None

        res_i_atoms = list(res_obj.get_atoms())
        all_res = list(res_obj.get_parent().get_parent().get_residues())
        idx_i = all_res.index(res_obj)
        try:
            res_ip1 = all_res[idx_i + 1]
            if res_obj.get_full_id()[2] == res_ip1.get_full_id()[2]:
                ip1_atoms = list(res_ip1.get_atoms())
            else:
                res_ip1 = None
                ip1_atoms = []
        except IndexError:
            res_ip1 = None
            ip1_atoms = []
        try:
            res_im1 = all_res[idx_i - 1]
            if res_obj.get_full_id()[2] == res_im1.get_full_id()[2]:
                im1_atoms = list(res_im1.get_atoms())
            else:
                res_im1 = None
                im1_atoms = []
        except IndexError:
            res_im1 = None
            im1_atoms = []
        # Use the above-defined functions to find the nearest Hydrogen/Oxygen to each atom
        # We only search within rad so if we get None, there is no H-bond.
        if HAatom is None:
            HAOatom = None
        else:
            ACatom = res_obj['CA']
            if ha_bond == 'permissive':
                HAOatom = self.find_nearest_atom(nn_tree, HAatom, rad[0], atom_type=['N', 'O'], excl=res_i_atoms, atom_name = False)
                if HAOatom is None:
                    pass
                else:
                    HAangle2 = PDB.calc_angle(ACatom.get_vector(), HAatom.get_vector(), HAOatom.get_vector())
                    HAbond_dist = HAatom - HAOatom
                    try:
                        HAOCatom = HAOatom.get_parent()['C']
                    except KeyError:
                        HAOCatom = None
                    if HAOCatom is None: #This should indicate that the Oxygen in question is in a water molecule
                        HAangle1 = math.pi * 109.5 /180 # Assume the water oxygen is in optimal geometry
                        HAenergy = 0
                        flag = 1
                    HAbond_Params = [HAbond_dist, math.cos(HAangle1), math.cos(HAangle2), flag, HAenergy]

            elif ha_bond == 'restrictive':
                excl_at_Ha = [] # A list of possible acceptor atoms that have been returned by nearest atom searches but that failed for some other reason so should be excluded from future searches
                flag = 0
                HAangle2 = 0
                HAangle1 = 0
                while (HAangle2 < (math.pi / 2)) or (HAangle1 < (math.pi / 2)):
                    flag=0
                    full_excl = res_i_atoms + ip1_atoms + im1_atoms + excl_at_Ha
                    HAOatom = self.find_nearest_atom(nn_tree, HAatom, rad[0], atom_type='O', excl=full_excl, atom_name = False)
                    if HAOatom is None: # No possible HA bond so break loop leaving HAbond_Params as 5*[0]
                        HAbond_Params=[0]*5
                        break
                    else:
                        excl_at_Ha += [HAOatom] # Add this acceptor to the exclusion list so that if we redo the loop it will not be included in the search again
                        HAangle2 = PDB.calc_angle(ACatom.get_vector(), HAatom.get_vector(), HAOatom.get_vector())
                        HAbond_dist = HAatom - HAOatom
                        try:
                            HAOCatom = HAOatom.get_parent()['C']
                        except KeyError:
                            HAOCatom = None
                        if HAOCatom is None: #This should indicate that the Oxygen in question is in a water molecule so set angle and energy to some default values but keep the HA bond
                            HAangle1 = math.pi * 109.5 /180
                            HAenergy = 0
                            flag = 1
                        else:
                            HAangle1 = PDB.calc_angle(HAatom.get_vector(), HAOatom.get_vector(), HAOCatom.get_vector())
                            HAenergy = 0.084 * 332 * (1/(ACatom - HAOatom) + 1/(HAatom - HAOCatom) - 1/(HAbond_dist) - 1/(ACatom - HAOCatom))
                            O_idx = all_res.index(HAOatom.get_parent()) # Get index of acceptor Oxygen in list of residues so we can exclude the correct neighbor residue when we check if this Oxygen has a secondary H-bond
                            try:
                                O_ip1_res = all_res[O_idx + 1]
                                O_ip1_atoms = list(O_ip1_res.get_atoms())
                            except IndexError:
                                O_ip1_atoms = []
                            O_ip1_atoms += [HAatom] #Need to exclude the H-alpha atom from the neighbor search but can't exclude all atoms on res_obj since the secondary H-bond on the Oxygen could be to the HN on res_obj
                            secondary_O_params = self.NH_O_bond(nn_tree, HAOatom.get_parent(), [], O_ip1_atoms, rad[2], HAOCatom, HAOatom, ['H', 'D'], efilt=efilter_O2) # Look for H-bond on acceptor Oxygen
                            if secondary_O_params[-2] == 1: # If acceptor Oxygen has an H-bond, keep candidate HA H-bond else loop will be repeated
                               flag = 1
                        HAbond_Params = [HAbond_dist, math.cos(HAangle1), math.cos(HAangle2), flag, HAenergy]

        if HNatom is None:
            pass
        else:
            Natom = res_obj['N']
            HNbond_Params = self.NH_O_bond(nn_tree, res_obj, im1_atoms, ip1_atoms, rad[1], Natom, HNatom, ['N', 'O'], efilt=efilt)
        if Oatom is None:
            pass
        else:
            Catom = res_obj['C']
            Obond_Params = self.NH_O_bond(nn_tree, res_obj, im1_atoms, ip1_atoms, rad[2], Catom, Oatom, ['H', 'D'], efilt=efilt)
        output = HAbond_Params + HNbond_Params + Obond_Params
        return output



    def hbond_network_sidechain(self, nn_tree, res_object_list, rad=[5.0], ha_bond='restrictive'):
        '''
        Side chain hydrogen bond features.
        
        Args:
            nn_tree - A tree containing the distances between atoms in the set of interest (Bio.PDB.NeighborSearch)
            res_obj - Residue for which we wish to find the Hydrogen bond parameters (Bio.PDB.Residue)
            rad - The distance within which to search for each of the three types of bonds H_alpha, H_N, and O (List of Float)
    
        Returns:
            [dHA, Cos(phi), Cos(psi), 1] 
            
        HDatom - h-bond donor
        HAatom - h-bond acceptor
        HACatom - C partner of h-bond acceptor
        '''
        
        partners = {
        "GLY": {"H": "N", "HA2": "CA", "HA3": "CA"}, 
        "ILE": {"H": "N", "HA": "CA", "HB": "CB", "HD1": "CD1", "HD2": "CD1", "HD3": "CD1", "HG12": "CG1", "HG13": "CG1", "HG21": "CG2", "HG22": "CG2", "HG23": "CG2"},
        "VAL": {"H": "N", "HA": "CA", "HG12": "CG1", "HG13": "CG1", "HG21": "CG2", "HG22": "CG2", "HB": "CB", "HG11": "CG1", "HG23": "CG2"}, 
        "ARG": {"H": "N", "HA": "CA", "HG2": "CG", "HB2": "CB", "HB3": "CB", "HD2": "CD", "HD3": "CD", "HE": "NE", "HG3": "CG","HH11": "NH1", "HH12": "NH1", "HH21": "NH2", "HH22": "NH2"},
        "TRP": {"H": "N", "HA": "CA", "HH2": "CH2", "HB2": "CB", "HB3": "CB", "HD1": "CD1", "HE1": "NE1", "HE3": "CE3", "HZ2": "CZ2",  "HZ3": "CZ3"},
        "GLU": {"H": "N", "HA": "CA", "HG2": "CG", "HG2": "CG", "HB2": "CB", "HB3": "CB", "HE2": "OE2", "HG3": "CG"},
        "GLN": {"H": "N", "HA": "CA", "HG2": "CG", "HB2": "CB", "HB3": "CB", "HE21": "NE2", "HE22": "NE2", "HG3": "CG"},
        "LYS": {"H": "N", "HA": "CA", "HG2": "CG", "HB2": "CB", "HB3": "CB", "HD2": "CD", "HD3": "CD", "HE2": "CE", "HE3": "CE", "HG3": "CG", "HZ1": "NZ", "HZ2": "NZ", "HZ3": "NZ"},
        "MET": {"H": "N", "HA": "CA", "HG2": "CG", "HB2": "CB", "HB3": "CB", "HE1": "CE", "HE2": "CE", "HE3": "CE", "HG3": "CG"}, 
        "PRO": {"H": "N", "HA": "CA", "HG2": "CG", "HB2": "CB", "HB3": "CB", "HD2": "CD", "HD3": "CD", "HG3": "CG"},
        "THR": {"H": "N", "HA": "CA", "HG21": "CG2", "HG22": "CG2", "HB": "CB", "HG1": "OG1", "HG23": "CG2"},
        "ALA": {"H": "N", "HA": "CA", "HB1": "CB", "HB2": "CB", "HB3": "CB"},
        "ASP": {"H": "N", "HA": "CA", "HB2": "CB", "HB3": "CB", "HD2": "OD2"},
        "ASN": {"H": "N", "HA": "CA", "HB2": "CB", "HB3": "CB", "HD21": "ND2", "HD22": "ND2"},
        "CYS": {"H": "N", "HA": "CA", "HB2": "CB", "HB3": "CB", "HG": "SG"}, 
        "HIS": {"H": "N", "HA": "CA", "HB2": "CB", "HB3": "CB", "HD1": "ND1", "HD2": "CD2", "HE1": "CE1", "HE2": "NE2"}, 
        "LEU": {"H": "N", "HA": "CA", "HB2": "CB", "HB3": "CB", "HD11": "CD1", "HD12": "CD1", "HD13": "CD1", "HD21": "CD2", "HD22": "CD2", "HD23": "CD2", "HG": "CG"}, 
        "PHE": {"H": "N", "HA": "CA", "HB2": "CB", "HB3": "CB", "HD1": "CD1", "HD2": "CD2", "HE1": "CE1", "HE2": "CE2", "HZ": "CZ"}, 
        "SER": {"H": "N", "HA": "CA", "HB2": "CB", "HB3": "CB", "HG": "OG"},  
        "TYR": {"H": "N", "HA": "CA", "HB2": "CB", "HB3": "CB", "HD1": "CD1", "HD2": "CD2", "HE1": "CE1", "HE2": "CE2", "HH": "OH"}}
        atom_names_target = ['HB', 'HB1', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE21', 'HE22', 'HG', 'HG1', 'HG12', 'HG13', 'HG2', 'HG3', 'HZ', 'HD11', 'HD12', 'HD13', 'HD23', 'HE3','HZ3','HH2','HZ2', 'HZ1', 'HG21', 'HG22', 'HG23', 'H', 'HA2', 'HA3']

        
        hbond_acceptors_partners = {'N': 'CA', 'O': 'C', 'OE1': 'CD', 'OE2': 'CD', 'OD1': 'CG', 'OD2': 'CG', 'OH': 'CZ', 'ND1': 'CG', 'OG': 'CB', 'OG1': 'CB', 'NE2': 'CD'}
        labels = ['_dHA', '_COS_H', '_COS_A', '_EXISTS', '_ENERGY']
        
        res_hbond_dict = {}
        
        for res in res_object_list:
            # First initialize all bonds as nonexistent
            Hsidechain_Params = [0]*5
            
            resnum = res.get_id()[1]

            target_atoms = [i for i in res.get_atoms() if i.get_id() in atom_names_target]
     
            res_i_atoms = list(res.get_atoms())
            all_res = list(res.get_parent().get_parent().get_residues())
            idx_i = all_res.index(res)
            try:
                res_ip1 = all_res[idx_i + 1]
                if res.get_full_id()[2] == res_ip1.get_full_id()[2]:
                    ip1_atoms = list(res_ip1.get_atoms())
                else:
                    res_ip1 = None
                    ip1_atoms = []
            except IndexError:
                res_ip1 = None
                ip1_atoms = []
            try:
                res_im1 = all_res[idx_i - 1]
                if res.get_full_id()[2] == res_im1.get_full_id()[2]:
                    im1_atoms = list(res_im1.get_atoms())
                else:
                    res_im1 = None
                    im1_atoms = []
            except IndexError:
                res_im1 = None
                im1_atoms = []
            # Use the above-defined functions to find the nearest Hydrogen/Oxygen to each atom
            # We only search within rad so if we get None, there is no H-bond.
    
            for atom in target_atoms:
                Hatom = res[atom.get_name()]
                
                if Hatom is None:
                    HAatom = None
                else:
                    
                    #find a partner
                    resname_str = res.get_resname()
                    if Hatom.get_id() in partners[resname_str]:
                        partner = partners[resname_str][Hatom.get_id()]
                        HDatom = res[partner]
                    else:
                        break
                    

                    if ha_bond == 'restrictive':
                        excl_at_Ha = [] # A list of possible acceptor atoms that have been returned by nearest atom searches but that failed for some other reason so should be excluded from future searches
                        flag = 0
                        HAangle2 = 0
                        HAangle1 = 0
                        while (HAangle2 < (math.pi / 2)) or (HAangle1 < (math.pi / 2)):
                            flag=0
                            full_excl = res_i_atoms + ip1_atoms + im1_atoms + excl_at_Ha
                            HAatom = self.find_nearest_atom(nn_tree, Hatom, rad=3.5, atom_type=['N', 'O', 'OE1', 'OE2', 'OD1', 'OD2', 'OH', 'ND1', 'OG', 'OG1', 'NE2'], excl=full_excl, atom_name = True)
                            if HAatom is None: # No possible HA bond so break loop leaving Hsidechain_Params as 5*[0]
                                Hsidechain_Params=[0]*5
                                break
                            elif (HAatom.get_name() in ['N', 'O', 'OE1', 'OE2', 'OD1', 'OD2', 'OH', 'ND1', 'OG', 'OG1', 'NE2']):                               
                                if (HAatom.get_name() == 'NE2') and (HAatom.get_parent().get_resname() == 'HIS'):
                                    Hsidechain_Params=[0]*5
                                    break
                                else:                   
                                    if HDatom:
                                        excl_at_Ha += [HAatom] 
                                        HAangle2 = PDB.calc_angle(HDatom.get_vector(), Hatom.get_vector(), HAatom.get_vector())
                                        HAbond_dist = Hatom - HAatom
                                        if HAatom.get_parent().get_resname() == 'HOH': #This should indicate that the Oxygen in question is in a water molecule so set angle and energy to some default values but keep the HA bond
                                            HAangle1 = math.pi * 109.5 /180
                                            HAenergy = 0
                                            flag = 1
                                        else:
                                            try:
                                                flag = 1
                                                HACatom = HAatom.get_parent()[hbond_acceptors_partners[HAatom.get_name()]]
                                                HAangle1 = PDB.calc_angle(Hatom.get_vector(), HAatom.get_vector(), HACatom.get_vector())
                                                HAenergy = 0.084 * 332 * (1/(HDatom - HAatom) + 1/(Hatom - HACatom) - 1/(HAbond_dist) - 1/(HDatom - HACatom))
                                            except:
                                                break
                            else:
                                break              
                            Hsidechain_Params = [HAbond_dist, math.cos(HAangle1), math.cos(HAangle2), flag, HAenergy]              
                        
                if resnum in res_hbond_dict.keys():
                    for i in range(5):
                        res_hbond_dict[resnum][atom.get_id()+labels[i]]=Hsidechain_Params[i]
                else:
                    res_hbond_dict[resnum]={}
                    for i in range(5):
                        res_hbond_dict[resnum][atom.get_id()+labels[i]]=Hsidechain_Params[i]
    
    
            if len(res_hbond_dict) != 0:
                included_atoms = []
                for i in res_hbond_dict.get(resnum, {}).keys():
                    if '_' in i:
                        included_atoms.append(i.split('_')[0])
                remaining_atoms = [i for i in atom_names_target if i not in included_atoms]
                for a in remaining_atoms:
                    if resnum in res_hbond_dict:
                        for i in range(5):
                            res_hbond_dict[resnum][a+labels[i]] = np.nan      
        
        return res_hbond_dict
    



    def check_disulfide(self, nn_tree, res_obj):
        '''
        Find out the oxidation state of the cysdiene residue of interest. i.e. determing whether the cysdiene forms a disulfide bond with other residues

        args:
            nn_tree - A tree containing the distances between atoms in the set of interest (Bio.PDB.NeighborSearch)
            res_obj - A CYS residue that we want to find out the oxidation state (Bio.PDB.Residue)

        returns:
            Bool - whether the CYS residue is oxidized or not
        '''
        try:
            S_atom = res_obj["SG"]
        except KeyError:
            S_atom = None
            for atom in res_obj.child_list:
                if 'S' in atom.name:
                    S_atom = atom
                    break
            if S_atom is None:
                return False
        nn_S = self.find_nearest_atom(nn_tree, S_atom, 2.5, atom_type='S', atom_name = True)
        if nn_S is None:
            return False
        else:
            # Check whether the atom occurs in a residue more than 4 amino acid away
            second_S_res_id = nn_S.parent.id[1]
            if abs(second_S_res_id - res_obj.id[1]) >= 4:
                return True
            return False



    def ionizable_residues(self, nn_tree, res_object_list):
        """
        atoms corresponding to ionisable residues: HG (Cys), HD2 (Asp), HE2 (Glu, His), HD1 (His)
        
        """
        
        protonation_dict = {}
        ionizable = ['HG', 'HD2', 'HE2', 'HD1']

        
        #Make sure all residues have unique numbers
        res_numbers = [i.get_id()[1] for i in res_object_list]  #list of res numbers
        if len(res_numbers)>len(set(res_numbers)):
            raise Exception('Not all residue numbers are unique in electric_field res_object_list')
    
    
        for res in res_object_list:
            res_dict = {'HG_PROTON': 0, 'HD2_PROTON': 0, 'HE2_PROTON': 0, 'HD1_PROTON': 0}
            resnum = res.get_id()[1]
            target_atoms = [i for i in res.get_atoms() if i.get_id() in ionizable]
            for atom in target_atoms:
                res_dict[atom.get_id()+'_PROTON'] = 1
                
            if resnum in protonation_dict.keys():
                protonation_dict[resnum]=res_dict
            else:
                protonation_dict[resnum]={}
                protonation_dict[resnum]=res_dict
        
        
        return protonation_dict








    def df_from_file_1res(self, fpath, hbrad=3*[5.0], ha_bond='restrictive', efilt=False, efilter_O2=True, s2rad=10.0, hse=True, first_chain_only=False, bfact_mode='all', sequence_columns=0):

        '''Function to create a pandas DataFrame of single-residue SPARTA+ features from a given PDB file.

        Args:
            fpath - path to PDB file (Str)
            hbrad - max length of hydrogen bonds (Float)
            ha_bond - restrictive or permissive, see hbond_network function (Str)
            efilt - use DSSP energy as cutoff for H-bonds, see hbond_network function (Bool)
            efilter_O2 - see hbond_network function (Bool)
            s2rad - distance within which to include heavy atoms for modeling S2 parameter (Float)
            hse - include a feature column for the half-sphere exposure (Bool)
            first_chain_only - only extract the first chain in the model (Bool)
            bfact_mode - Identifies the set of atoms used to obtain an average b-factor for a given residue.  Accepts "all" or "set6".  See get_bfactor (Str)
            sequence_columns - number of flanking residues on either side of the central residue to include for sequence matching columns (Int)

        Returns:
            df - DataFrame with number of rows equal to number of standard amino acids in the PDB and columns given by the SPARTA+ features as well as some preliminary file and residue ID info and additional features beyond the SPARTA+ set (Pandas DataFrame)
            '''

        parser = PDBParser(PERMISSIVE=1)
        structure = parser.get_structure('structure', fpath)
        if "H" not in set((atom.get_name() for atom in structure.get_atoms())):
            print("Warning! No hydrogen atoms found in the structure. Using a structure without hydrogen atoms will significantly detoriate the performance!!!")
        file_id = fpath.split('/')[-1].split('.')[0].split('_')[0]
        file_name = fpath.split('/')[-1]

        col_names = ['FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM', 'CHAIN']
        col_names +=[i+j for i in ['PHI_', 'PSI_'] for j in ['COS', 'SIN']]
        col_names += [i+j for i in ['CHI1_','ALTCHI1_', 'CHI2_','ALTCHI2_', 'CHI3_', 'CHI4_', 'CHI5_'] for j in ['COS', 'SIN', 'EXISTS']]
        col_names += [i+j for i in ['Ha_', 'HN_', 'O_'] for j in ['d_HA', 'COS_H', 'COS_A', 'EXISTS', 'ENERGY']]
        col_names += ['S2']
        AAlet3 = [i.upper() for i in sorted(IUPACData.protein_letters_3to1.keys())]
        col_names += ['BLOSUM62_NUM_'+AAlet3[i] for i in range(20)]
        if hse:
            col_names += ['HSE_CA' + i  for i in ['_U', '_D', '_Angle']]
            col_names += ['HSE_CB' + i for i in ['_U', '_D']]
        col_names += ['A_HELIX_SS', 'B_BRIDGE_SS', 'STRAND_SS', '3_10_HELIX_SS', 'PI_HELIX_SS', 'TURN_SS', 'BEND_SS', 'NONE_SS']
        col_names += ['REL_ASA', 'ABS_ASA']
        col_names += ['DSSP_PHI', 'DSSP_PSI']
        col_names += ['NH-O1_ENERGY', 'NH-O2_ENERGY', 'O-NH1_ENERGY', 'O-NH2_ENERGY']
        ring_column_names = ['C_RING', 'CA_RING', 'CB_RING', 'N_RING', 'H_RING', 'HA_RING', 'HA2_RING', 'HA3_RING','CG_RING','CD_RING', 'CD1_RING', 'CD2_RING', 'CE_RING', 'CE1_RING', 'CE2_RING', 'CG1_RING', 'CG2_RING', 'CZ_RING', 'HB_RING', 'HB2_RING', 'HB3_RING', 'HD1_RING', 'HD2_RING', 'HD21_RING', 'HD22_RING', 'HD3_RING', 'HE_RING', 'HE1_RING', 'HE2_RING', 'HE21_RING', 'HE22_RING', 'HE3_RING', 'HG_RING', 'HG1_RING', 'HG12_RING', 'HG13_RING', 'HG2_RING', 'HG3_RING', 'HZ_RING', 'CE3_RING', 'CZ3_RING', 'HZ3_RING', 'CH2_RING', 'HH2_RING', 'CZ2_RING', 'HZ2_RING', 'ND2_RING','NE1_RING','NE2_RING']
        efield_column_names = ['H_EFIELD', 'HA_EFIELD', 'HB_EFIELD', 'HB2_EFIELD', 'HB3_EFIELD', 'HD1_EFIELD', 'HD2_EFIELD', 'HD21_EFIELD', 'HD22_EFIELD', 'HD3_EFIELD', 'HE_EFIELD', 'HE1_EFIELD', 'HE2_EFIELD', 'HE3_EFIELD', 'HE21_EFIELD', 'HE22_EFIELD', 'HG_EFIELD', 'HG1_EFIELD', 'HG12_EFIELD', 'HG13_EFIELD', 'HG2_EFIELD', 'HG3_EFIELD', 'HH2_EFIELD', 'HZ_EFIELD', 'HZ2_EFIELD', 'HZ3_EFIELD', 'N_EFIELD', 'ND2_EFIELD', 'NE1_EFIELD', 'NE2_EFIELD']
        ionizable_column_names = ['HG_PROTON', 'HD2_PROTON', 'HE2_PROTON', 'HD1_PROTON']
        hbond_schain_column_names = [i+j for i in ['HB', 'HB1', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE21', 'HE22', 'HG', 'HG1', 'HG12', 'HG13', 'HG2', 'HG3', 'HZ','HD11', 'HD12', 'HD13', 'HD23', 'HE3','HZ3','HH2','HZ2', 'HZ1', 'HG21', 'HG22', 'HG23']  for j in ['_dHA', '_COS_H', '_COS_A', '_EXISTS', '_ENERGY']]

        col_bfact = ['AVG_B']
        col_names += ring_column_names
        col_names += efield_column_names
        col_names += ionizable_column_names
        col_names += hbond_schain_column_names
        col_names += col_bfact
        if sequence_columns > 0:
            seq_match_cols = ['RESNAME_i-' + str(i) for i in range(sequence_columns, 0, -1)] + ['RESNAME_i+' + str(i) for i in range(1, sequence_columns+1)]
            col_names += seq_match_cols
        col_names += ["CYS_OX"]
        data = []
        for model in structure:
            nn_tree = PDB.NeighborSearch(list(model.get_atoms()))
            try:
                dssp = PDB.DSSP(model, fpath)
            except:
                dssp = []
            if hse:
                hseca_calc = PDB.HSExposureCA(model)
                hsecb_calc = PDB.HSExposureCB(model)

            for chain in model:
                if sequence_columns > 0:
                    res_dict = {res.get_id() : idx for idx,res in enumerate(chain.get_unpacked_list())}
                    list_of_resnames = [res.resname for res in chain.get_unpacked_list()]
                    list_of_resnames = ['NONE'] * sequence_columns + list_of_resnames + ['NONE'] * sequence_columns
                dihedrals = self.calc_phi_psi(chain)
                polys = PDB.PPBuilder(radius=2.1).build_peptides(chain)
#                poly_residues = [j for i in polys for j in i]
#                insertion_resnums = []
#                #remove residues with insertion codes from poly residues.
#                for poly_res in poly_residues:
#                    if poly_res.get_full_id()[3][2] != ' ':
#                        insertion_resnums += [poly_res.get_full_id()[3][1]]
#                poly_residues = [i for i in poly_residues if i.get_full_id()[3][1] not in insertion_resnums]
                poly_residues = [res for poly in polys for res in poly if res.id[2] == ' ']
                seq = ''.join([PDB.Polypeptide.three_to_one(i.get_resname()) for i in poly_residues])
                chain_resnum_list = [i.get_id()[1] for i in poly_residues]
                rings = self.calc_ring_currents(poly_residues)
                efields = self.electric_field(nn_tree, poly_residues)
                ionizable = self.ionizable_residues(nn_tree, poly_residues)
                hbond_schain = self.hbond_network_sidechain(nn_tree, poly_residues, rad=[5.0], ha_bond='restrictive')


                for l, res in enumerate(poly_residues):
                    resname = self._fix_res_name(res.resname)

                    #A few defensive checks to make sure we are looping through peptide segments correctly
                    if resname not in AAlet3:
                        raise Exception('Something is wrong with peptide loop: '+fpath)
                    if PDB.Polypeptide.three_to_one(resname)!=seq[l]:
                        raise Exception('Something is wrong with peptide loop: '+fpath)

                    res_id = res.get_id()
                    resnum = res_id[1]
                    row_data = [file_id, file_name, resname, resnum, chain.get_id()]
                    row_data += dihedrals[l]
                    torsions = self.calc_torsion_angles(res)
                    row_data += torsions


                    HBonds = self.hbond_network(nn_tree, res, hbrad, ha_bond=ha_bond, efilt=efilt, efilter_O2=efilter_O2)
                    row_data += HBonds
                    row_data += [self.s2_param(nn_tree, res, chain, s2rad)]
                    row_data += self.blosum_nums(res)
                    if hse:
                        try:
                            hseca = hseca_calc[(chain.get_id(), res_id)]
                            hseca = list(hseca)
                            hsecb = hsecb_calc[(chain.get_id(), res_id)]
                            hsecb = list(hsecb)[:-1]
                        except KeyError:
                            hseca = 3 * [0]
                            hsecb = 2 * [0]
                        row_data += hseca
                        row_data += hsecb

                    try:
                        dssp_dat = dssp[(chain.id, res.id)]
                        row_data += secondary_struc_dict[dssp_dat[2]]
                        row_data += [dssp_dat[3], dssp_dat[3] * max_asa_dict[resname]]
                        row_data += [dssp_dat[4], dssp_dat[5]]
                        row_data += [dssp_dat[7], dssp_dat[9], dssp_dat[11], dssp_dat[13]]
                    except KeyError:
                        print('KeyError at ' + str((chain.id, res.id)) + '.  Skipping this residue')
                        row_data += 16*[0]
                        continue
                    except TypeError:
                        print('DSSP failed on ' + str((chain.id, res.id)) + '.  Skipping this residue')
                        row_data += 16*[0]
                        continue
                    row_data += [rings[resnum][i] for i in ring_column_names]
                    
                    try:
                        row_data += [efields[resnum][i] for i in efield_column_names]
                    except KeyError:
                        print('Electric field KeyError at ' + str((chain.id, res.id)) + '.  Skipping this residue')
                        row_data += 40*[0]
                        continue
                    
                    row_data += [ionizable[resnum][i] for i in ionizable_column_names]
                    
                    try:
                        row_data += [hbond_schain[resnum][i] for i in hbond_schain_column_names]
                    except KeyError:
                        print('Hbond KeyError at ' + str((chain.id, res.id)) + '.  Skipping this residue')
                        row_data += 185*[0]
                        continue                  
                    
                    
                    row_data += [self.get_bfactor(res, atoms=bfact_mode)]

                    if sequence_columns > 0:
                        central_res_idx = res_dict[res.get_id()] + sequence_columns # Obtain index of central residue, accounting for the 'NONE' padding of list_of_resnames
                        row_data += list_of_resnames[central_res_idx - sequence_columns : central_res_idx] + list_of_resnames[central_res_idx+1 : central_res_idx + 1 + sequence_columns]
                    if resname == "CYS":
                        # Check oxidation state of cysdiene (whether there are disulfide bonds)
                        row_data += [self.check_disulfide(nn_tree, res)]
                    else:
                        row_data += [False]
                    data.append(row_data)
                if first_chain_only:
                    break
        df = pd.DataFrame(data, columns=col_names)
        return df


    def df_from_file_3res(self, fpath, hbrad=3*[5.0], ha_bond='restrictive', efilt=False, efilter_O2=True, s2rad=10.0, rcshifts=True, hse=True, first_chain_only=False, bfact_mode='all', sequence_columns=0):

        '''Function to create a pandas DataFrame of SPARTA+ features from a given PDB file.

            Args:
            fpath - Path to PDB file (Str)
            hbrad - Max length of hydrogen bonds (Float)
            s2rad - Distance within which to include heavy atoms for modeling S2 parameter (Float)
            rcshifts - Include a column for random coil chemical shifts (Bool)
            hse - include a feature column for the half-sphere exposure (Bool)
            first_chain_only - only extract the first chain in the model (Bool)
            bfact_mode - Identifies the set of atoms used to obtain an average b-factor for a given residue.  Accepts "all" or "set6".  See get_bfactor (Str)
            sequence_columns - number of flanking residues on either side of the central residue to include for sequence matching columns (Int)

        Returns:
            df - DataFrame with number of rows equal to number of standard amino acids in the PDB and columns given by the SPARTA+ features as well as some preliminary file and residue ID info and additional features beyond the SPARTA+ set (Pandas DataFrame)
        '''

        #Names of columns from single-residue DataFrame for easier access
        phipsi_names = [i+j for i in ['PHI_', 'PSI_'] for j in ['COS', 'SIN']]
        chi_names = [i+j for i in ['CHI1_','ALTCHI1_', 'CHI2_','ALTCHI2_', 'CHI3_', 'CHI4_', 'CHI5_'] for j in ['COS', 'SIN', 'EXISTS']]
        hbprev_names = ['O_'+j for j in ['d_HA', 'COS_H', 'COS_A', 'EXISTS', 'ENERGY']]
        hb_names = [i+j for i in ['Ha_', 'HN_', 'O_'] for j in ['d_HA', 'COS_H', 'COS_A', 'EXISTS', 'ENERGY']]
        hbnext_names = ['HN_'+j for j in ['d_HA', 'COS_H', 'COS_A', 'EXISTS', 'ENERGY']]
        hse_names = ['HSE_CA' + i  for i in ['_U', '_D', '_Angle']]
        hse_names += ['HSE_CB' + i for i in ['_U', '_D']]
        ss_names = ['A_HELIX_SS', 'B_BRIDGE_SS', 'STRAND_SS', '3_10_HELIX_SS', 'PI_HELIX_SS', 'TURN_SS', 'BEND_SS', 'NONE_SS']
        asa_names = ['REL_ASA', 'ABS_ASA']
        dssp_phipsi_names = ['DSSP_PHI', 'DSSP_PSI']
        dssp_hbond_names = ['NH-O1_ENERGY', 'NH-O2_ENERGY', 'O-NH1_ENERGY', 'O-NH2_ENERGY']
        dssp_names = asa_names + ss_names + dssp_hbond_names + dssp_phipsi_names

        # Define column names for new DataFrame
        col_id = ['FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM', 'CHAIN']
        col_extra_resnames = ['RESNAME_im1', 'RESNAME_ip1']
        col_phipsi = [i+j+k for k in ['i-1', 'i', 'i+1'] for i in ['PHI_', 'PSI_'] for j in ['COS_', 'SIN_']]
        col_chi = [i+j+k for k in ['_i-1', '_i', '_i+1'] for i in ['CHI1_','ALTCHI1_', 'CHI2_','ALTCHI2_', 'CHI3_', 'CHI4_', 'CHI5_'] for j in ['COS', 'SIN', 'EXISTS']]
        #col_alpha = [i+j for i in ['ALPHA1_','ALPHA2_','ALPHA3_'] for j in ['ANGLE', 'EXISTS']]
        col_hbprev = ['O_'+i+'_i-1' for i in ['d_HA', '_COS_H', '_COS_A', '_EXISTS', '_ENERGY']]
        col_hbond = [i+j+'_i' for i in ['Ha_', 'HN_', 'O_'] for j in ['d_HA', '_COS_H', '_COS_A', '_EXISTS', '_ENERGY']]
        col_hbnext = ['HN_'+i+'_i+1' for i in ['d_HA', '_COS_H', '_COS_A', '_EXISTS', '_ENERGY']]
        col_s2 = ['S2'+i for i in ['_i-1', '_i', '_i+1']]
        blosum_names = ['BLOSUM62_NUM_'+sorted(list(IUPACData.protein_letters_3to1.keys()))[i].upper() for i in range(20)]
        col_blosum = [blosum_names[i]+j for j in ['_i-1', '_i', '_i+1'] for i in range(20)]
        col_names = col_id + col_extra_resnames + col_phipsi + col_chi + col_hbprev + col_hbond + col_hbnext + col_s2 + col_blosum
        if rcshifts:
            col_rccs = ['RC_' + i for i in atom_names]
            col_names += col_rccs
        if hse:
            col_hse = [hse_names[i] + j for j in ['_i-1', '_i', '_i+1'] for i in range(5)]
            col_names += col_hse
        col_dssp = [i + j for j in ['_i-1', '_i', '_i+1'] for i in dssp_names]
        col_names += col_dssp


        ring_column_names = ['C_RING', 'CA_RING', 'CB_RING', 'N_RING', 'H_RING', 'HA_RING', 'HA2_RING', 'HA3_RING','CG_RING','CD_RING', 'CD1_RING', 'CD2_RING', 'CE_RING', 'CE1_RING', 'CE2_RING', 'CG1_RING', 'CG2_RING', 'CZ_RING', 'HB_RING', 'HB2_RING', 'HB3_RING', 'HD1_RING', 'HD2_RING', 'HD21_RING', 'HD22_RING', 'HD3_RING', 'HE_RING', 'HE1_RING', 'HE2_RING', 'HE21_RING', 'HE22_RING', 'HE3_RING', 'HG_RING', 'HG1_RING', 'HG12_RING', 'HG13_RING', 'HG2_RING', 'HG3_RING', 'HZ_RING', 'CE3_RING', 'CZ3_RING', 'HZ3_RING', 'CH2_RING', 'HH2_RING', 'CZ2_RING', 'HZ2_RING', 'ND2_RING','NE1_RING','NE2_RING']
        efield_column_names = ['H_EFIELD', 'HA_EFIELD', 'HB_EFIELD', 'HB2_EFIELD', 'HB3_EFIELD', 'HD1_EFIELD', 'HD2_EFIELD', 'HD21_EFIELD', 'HD22_EFIELD', 'HD3_EFIELD', 'HE_EFIELD', 'HE1_EFIELD', 'HE2_EFIELD', 'HE3_EFIELD', 'HE21_EFIELD', 'HE22_EFIELD', 'HG_EFIELD', 'HG1_EFIELD', 'HG12_EFIELD', 'HG13_EFIELD', 'HG2_EFIELD', 'HG3_EFIELD', 'HH2_EFIELD', 'HZ_EFIELD', 'HZ2_EFIELD', 'HZ3_EFIELD', 'N_EFIELD', 'ND2_EFIELD', 'NE1_EFIELD', 'NE2_EFIELD']
        ionizable_column_names = ['HG_PROTON', 'HD2_PROTON', 'HE2_PROTON', 'HD1_PROTON']
        hbond_schain_column_names = [i+j for i in ['HB', 'HB1', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE21', 'HE22', 'HG', 'HG1', 'HG12', 'HG13', 'HG2', 'HG3', 'HZ', 'HD11', 'HD12', 'HD13', 'HD23', 'HE3','HZ3','HH2','HZ2', 'HZ1', 'HG21', 'HG22', 'HG23']  for j in ['_dHA', '_COS_H', '_COS_A', '_EXISTS', '_ENERGY']]

        col_names += ring_column_names
        col_names += efield_column_names
        col_names += ionizable_column_names
        col_names += hbond_schain_column_names
        bfact_names = ['AVG_B' + i for i in ['_i-1', '_i', '_i+1']]
        col_names += bfact_names
        if sequence_columns > 0:
            seq_match_cols = ['RESNAME_i-' + str(i) for i in range(sequence_columns, 0, -1)] + ['RESNAME_i+' + str(i) for i in range(1, sequence_columns+1)]
            col_names += seq_match_cols



        full_df = pd.DataFrame(columns=col_names)

        # Get SPARTA+ features with single-residue function
        df_1res_all_chains = self.df_from_file_1res(fpath, hbrad, ha_bond=ha_bond, efilt=efilt, efilter_O2=efilter_O2, s2rad=s2rad, hse=hse, first_chain_only=first_chain_only, bfact_mode=bfact_mode, sequence_columns=sequence_columns)

        for chain in list(set(df_1res_all_chains['CHAIN'].tolist())):
            df = pd.DataFrame(columns=col_names)
            df_1res = df_1res_all_chains[df_1res_all_chains['CHAIN']==chain]
            df_1res.index=range(len(df_1res))
            #Get list of residues from 1res df to account for chain breaks.
            #Defensive check for resnum uniqueness and monotonicity
            res_list = df_1res['RES_NUM'].tolist()
            if len(set(res_list)) != len(res_list):
                raise Exception('residue number list has duplicates: '+fpath)
            if (np.sort(res_list) != res_list).any():
                raise Exception('residue number list is not ordered: '+fpath)

            for i in range(len(df_1res)):

                # Assign ID columns
                df.loc[i, col_id] = df_1res.loc[i, col_id].values

                #Get resnumber for residue we are looking at
                res_i_num = df_1res.loc[i, 'RES_NUM']

                # Assign column variables containing data from previous residue in the PDB file
                if (res_i_num-1) not in res_list:
                    blosum_prev = [0]*20
                    psi_prev = [0, 0]
                    phi_prev = [0, 0]
                    chi_prev = [0]*21
                    hb_prev = [0]*5
                    s2_prev = 0
                    hse_prev = 5*[0]
                    resname_prev = ''
                    dssp_prev = 16*[0]
                    bfact_prev = [0]
                else:
                    blosum_prev = list(df_1res.loc[i-1, blosum_names].values)
                    psi_prev = [df_1res.loc[i-1, 'PSI_COS'], df_1res.loc[i-1, 'PSI_SIN']]
                    phi_prev = [df_1res.loc[i-1, 'PHI_COS'], df_1res.loc[i-1, 'PHI_SIN']]
                    chi_prev = list(df_1res.loc[i-1, chi_names].values)
                    hb_prev = list(df_1res.loc[i-1, hbprev_names].values)
                    s2_prev = df_1res.loc[i-1, 'S2']
                    if hse:
                        hse_prev = list(df_1res.loc[i-1, hse_names].values)
                    resname_prev = df_1res.loc[i-1, 'RESNAME']
                    dssp_prev = list(df_1res.loc[i-1, dssp_names])
                    bfact_prev = [df_1res.loc[i-1, 'AVG_B']]

                # Assign column variables containing data from next residue in the PDB file
                if (res_i_num+1) not in res_list:
                    blosum_next = [0]*20
                    psi_next = [0, 0]
                    phi_next = [0, 0]
                    chi_next = [0]*21
                    hb_next = [0]*5
                    s2_next = 0
                    hse_next = 5*[0]
                    if rcshifts:
                        res_next = 'ALA'
                    resname_next = ''
                    dssp_next = 16*[0]
                    bfact_next = [0]
                else:
                    blosum_next = list(df_1res.loc[i+1, blosum_names].values)
                    psi_next = [df_1res.loc[i+1, 'PSI_COS'], df_1res.loc[i+1, 'PSI_SIN']]
                    phi_next = [df_1res.loc[i+1, 'PHI_COS'], df_1res.loc[i+1, 'PHI_SIN']]
                    chi_next = list(df_1res.loc[i+1, chi_names].values)
                    hb_next = list(df_1res.loc[i+1, hbnext_names].values)
                    s2_next = df_1res.loc[i+1, 'S2']
                    if hse:
                        hse_next = list(df_1res.loc[i+1, hse_names].values)
                    if rcshifts:
                        res_next = df_1res.loc[i+1, 'RESNAME']
                    resname_next = df_1res.loc[i+1, 'RESNAME']
                    dssp_next = list(df_1res.loc[i+1, dssp_names])
                    bfact_next = [df_1res.loc[i+1, 'AVG_B']]

                # Insert row into DataFrame
                df.loc[i, col_extra_resnames] = [resname_prev, resname_next]
                df.loc[i, col_phipsi] = phi_prev + psi_prev + list(df_1res.loc[i, phipsi_names].values) + phi_next + psi_next
                df.loc[i, col_chi] = chi_prev + list(df_1res.loc[i, chi_names].values) + chi_next
                df.loc[i, col_hbprev] = hb_prev
                df.loc[i, col_hbond] = list(df_1res.loc[i, hb_names].values)
                df.loc[i, col_hbnext] = hb_next
                df.loc[i, col_s2] = [s2_prev, df_1res.loc[i, 'S2'], s2_next]
                df.loc[i, col_blosum] = blosum_prev + list(df_1res.loc[i, blosum_names].values) + blosum_next
                if rcshifts:
                    resname = df.loc[i, 'RESNAME']
                    if res_next == 'PRO':
                        rccs = [randcoil_pro[j][resname] for j in atom_names]
                    else:
                        rccs = [randcoil_ala[j][resname] for j in atom_names]
                    if df_1res.loc[i, 'CYS_OX']:
                        if res_next == 'PRO':
                            print("Warning! Oxidized cys found preceding PRO!")
                            # Assume that next residue cannot be proline
                        for idx,j in enumerate(atom_names):
                            rccs[idx] += oxidized_cys_correction[j]
                    df.loc[i, col_rccs] = rccs
                if hse:
                    df.loc[i, col_hse] = hse_prev + list(df_1res.loc[i, hse_names].values) + hse_next
                df.loc[i, ring_column_names] = list(df_1res.loc[i, ring_column_names].values)
                df.loc[i, efield_column_names] = list(df_1res.loc[i, efield_column_names].values)
                df.loc[i, ionizable_column_names] = list(df_1res.loc[i, ionizable_column_names].values)
                df.loc[i, hbond_schain_column_names] = list(df_1res.loc[i, hbond_schain_column_names].values)
                df.loc[i, col_dssp] = dssp_prev + list(df_1res.loc[i, dssp_names]) + dssp_next
                df.loc[i, bfact_names] = bfact_prev + [df_1res.loc[i, 'AVG_B']] + bfact_next
                if sequence_columns > 0:
                    df.loc[i, seq_match_cols] = list(df_1res.loc[i, seq_match_cols])

            #full_df = full_df.append(df)
            full_df = pd.concat([full_df,df])
        return full_df


    def df_from_file_tripeptide(self, fpath, hbrad=5.0, s2rad=10.0, rcshifts=True):

        '''Function to create a pandas DataFrame of full tripeptide SPARTA+ features from a given PDB file.

        Args:
            fpath - Path to PDB file (Str)
            hbrad - Max length of hydrogen bonds (Float)
            s2rad - Distance within which to include heavy atoms for modeling S2 parameter (Float)
            rcshifts - Include a column for random coil chemical shifts (Bool)

        Returns:
            df - DataFrame  with number of rows equal to number of standard amino acids in the PDB and columns given by the SPARTA+ features as well as some preliminary file and residue ID info and additional features beyond the SPARTA+ set (Pandas DataFrame)
        '''

        # Get SPARTA+ features with single-residue function and then define column names
        df_1res = self.df_from_file_1res(fpath, hbrad, s2rad)
        col_id = ['FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM']
        col_phipsi = [i+j+k for k in ['i-1', 'i', 'i+1'] for i in ['PHI_', 'PSI_'] for j in ['COS_', 'SIN_']]
        col_chi = [i+j+k for k in ['_i-1', '_i', '_i+1'] for i in ['CHI1_','ALTCHI1_', 'CHI2_','ALTCHI2_', 'CHI3_', 'CHI4_', 'CHI5_'] for j in ['COS', 'SIN', 'EXISTS']]
        col_hbond = [i+j+k for k in ['_i-1', '_i', '_i+1'] for i in ['Ha_', 'HN_', 'O_']
                     for j in ['d_HA', '_COS_H', '_COS_A', '_EXISTS']]
        col_s2 = ['S2'+i for i in ['_i-1', '_i', '_i+1']]
        blosum_names = ['BLOSUM62_NUM_'+sorted(list(IUPACData.protein_letters_3to1.keys()))[i].upper() for i in range(20)]
        col_blosum = [blosum_names[i]+j for j in ['_i-1', '_i', '_i+1'] for i in range(20)]
        col_names = col_id + col_phipsi + col_chi + col_hbond + col_s2 + col_blosum
        if rcshifts:
            col_rccs = ['RC_' + i for i in atom_names]
            col_names += col_rccs


        df = pd.DataFrame(columns=col_names)

        #Names of columns from single-residue DataFrame for easier access
        phipsi_names = [i+j for i in ['PHI_', 'PSI_'] for j in ['COS', 'SIN']]
        chi_names = [i+j for i in ['CHI1_','ALTCHI1_', 'CHI2_','ALTCHI2_', 'CHI3_', 'CHI4_', 'CHI5_'] for j in ['COS', 'SIN', 'EXISTS']]
#        hbprev_names = ['O_'+j for j in ['d_HA', 'COS_H', 'COS_A', 'EXISTS']]
        hb_names = [i+j for i in ['Ha_', 'HN_', 'O_'] for j in ['d_HA', 'COS_H', 'COS_A', 'EXISTS']]
#        hbnext_names = ['HN_'+j for j in ['d_HA', 'COS_H', 'COS_A', 'EXISTS']]

        for i in range(len(df_1res)):
            # Assign ID columns
            df.loc[i, col_id] = df_1res.loc[i, col_id].values

            # Assign column variables containing data from previous residue in the PDB file
            if i == 0:
                blosum_prev = [0]*20
                phipsi_prev= [0]*4
                chi_prev = [0]*21
                hb_prev = [0]*12
                s2_prev = 0
            else:
                blosum_prev = list(df_1res.loc[i-1, blosum_names].values)
                phipsi_prev = [df_1res.loc[i-1, 'PHI_COS'], df_1res.loc[i-1, 'PHI_SIN'], df_1res.loc[i-1, 'PSI_COS'], df_1res.loc[i-1, 'PSI_SIN']]
                chi_prev = list(df_1res.loc[i-1, chi_names].values)
                hb_prev = list(df_1res.loc[i-1, hb_names].values)
                s2_prev = df_1res.loc[i-1, 'S2']

            # Assign column variables containing data from next residue in the PDB file
            if i == len(df_1res)-1:
                blosum_next = [0]*20
                phipsi_next= [0]*4
                chi_next = [0]*21
                hb_next = [0]*12
                s2_next = 0
                if rcshifts:
                    res_next = 'ALA'
            else:
                blosum_next = list(df_1res.loc[i+1, blosum_names].values)
                phipsi_next = [df_1res.loc[i+1, 'PHI_COS'], df_1res.loc[i+1, 'PHI_SIN'], df_1res.loc[i+1, 'PSI_COS'], df_1res.loc[i+1, 'PSI_SIN']]
                chi_next = list(df_1res.loc[i+1, chi_names].values)
                hb_next = list(df_1res.loc[i+1, hb_names].values)
                s2_next = df_1res.loc[i+1, 'S2']
                if rcshifts:
                    res_next = df_1res.loc[i+1, 'RESNAME']

            # Insert row into DataFrame
            df.loc[i, col_phipsi] = phipsi_prev + list(df_1res.loc[i, phipsi_names].values) + phipsi_next
            df.loc[i, col_chi] = chi_prev + list(df_1res.loc[i, chi_names].values) + chi_next
#            df.loc[i, col_hbprev] = hb_prev
            df.loc[i, col_hbond] = hb_prev + list(df_1res.loc[i, hb_names].values) + hb_next
#            df.loc[i, col_hbnext] = hb_next
            df.loc[i, col_s2] = [s2_prev, df_1res.loc[i, 'S2'], s2_next]
            df.loc[i, col_blosum] = blosum_prev + list(df_1res.loc[i, blosum_names].values) + blosum_next
            if rcshifts:
                resname = df.loc[i, 'RESNAME']
                if res_next == 'PRO':
                    rccs = [randcoil_pro[i][resname] for i in atom_names]
                else:
                    rccs = [randcoil_ala[i][resname] for i in atom_names]
                df.loc[i, col_rccs] = rccs


        return df


