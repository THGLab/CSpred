#!/usr/bin/env python
# This program executes both sequence-based alignment (using BLAST) and structure-based alignment (usint mTM-align) to find the best alignment for a specific pdb file with entities in the refDB database, and use the average chemical shifts from refDB to predict the chemical shifts for backbone H/C/N atom chemical shifts for the query protein

# Author: Jie Li
# Date created: Aug 21, 2019

import Bio
from Bio import PDB
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio import Align
from Bio.SubsMat.MatrixInfo import blosum62
from save_pdb import PDBSaver
import subprocess
import os
import shutil
import sys
import toolbox
import pandas as pd
import numpy as np
import argparse

DEBUG=False
GLOBAL_TEST_CUTOFF=0.99
SCRIPT_PATH=os.path.dirname(os.path.realpath(__file__))
BLAST_DEFAULT_EXE=SCRIPT_PATH+"/bins/ncbi-blast-2.9.0+/bin/blastp"
MTM_DEFAULT_EXE=SCRIPT_PATH+"/bins/mTM-align/mTM-align"
os.environ["BLASTDB"]=SCRIPT_PATH+"/refDB/"  # Set the refDB position

# Do checks beforehand to make sure the two alignment programs: BLAST and mTM-align are installed and configured correctly
# try:
#     check_result=subprocess.check_output(["which","blastp"])
# except:
#     check_result=""
# assert len(check_result)!=0,"Cannot find BLAST program! Please make sure BLAST is correctly configured."
# try:
#     check_result=subprocess.check_output(["which","mTM-align"])
# except:
#     check_result=""
# assert len(check_result)!=0,"Cannot find mTM-align program! Please make sure mTM-align is correctly configured."

# Decompress refDB pdb files if they don't exist
if not os.path.exists(SCRIPT_PATH+"/refDB/pdbs/"):
    os.makedirs(SCRIPT_PATH+"/refDB/pdbs")
    print("Decompressing mTM-align database...")
    os.system("tar -xzf %s/refDB/pdbs.tgz -C %s/refDB/"%(SCRIPT_PATH,SCRIPT_PATH))

# ================ Define Random Coils ======================
# Wishart et al. in J-Bio NMR, 5 (1995) 67-81.
paper_order = ['Ala', 'Cys','Asp','Glu','Phe','Gly','His','Ile','Lys','Leu','Met','Asn','Pro','Gln','Arg','Ser','Thr','Val','Trp','Tyr']

paper_order = [i.upper() for i in paper_order]

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




randcoil_ala = {i: dict(zip(paper_order, rc_ala[i])) for i in toolbox.ATOMS}
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







randcoil_pro = {i: dict(zip(paper_order, rc_pro[i])) for i in toolbox.ATOMS}

EXTERNAL_MAPPINGS = {"HIE":"HIS","HID":"HIS","HIP":"HIS","CAS":"CYS","CSD":"CYS","MSE":"MET","CSO":"CYS"}

SS_CAPS = {"H": 3.7, "HA": 3, "C": 10, "CA": 11.25, "CB": 20, "N": 22, "CG": np.nan, 'CD': np.nan, 'CD1': np.nan, 'CD2': np.nan, 'CE': np.nan, 'CE1': np.nan, 'CE2': np.nan, 'CG1': np.nan, 'CG2': np.nan, 'CZ': np.nan, 'HB': np.nan, 'HB2': np.nan, 'HB3': np.nan, 'HD1': np.nan, 'HD2': np.nan, 'HD21': np.nan, 'HD22': np.nan, 'HD3': np.nan, 'HE': np.nan, 'HE1': np.nan, 'HE2': np.nan, 'HE21': np.nan, 'HE22': np.nan, 'HE3': np.nan, 'HG': np.nan, 'HG1': np.nan, 'HG12': np.nan, 'HG13': np.nan, 'HG2': np.nan, 'HG3': np.nan, 'HZ': np.nan, 'CE3': np.nan, 'CZ3': np.nan, 'HZ3': np.nan, 'CH2': np.nan, 'HH2': np.nan, 'CZ2': np.nan, 'HZ2': np.nan, 'ND2': np.nan,'NE1': np.nan,'NE2': np.nan}

def read_sing_chain_PDB(path,fix_unknown_res=True,remove_alternate_res=True):
    '''
    Reads a pdb file from path and check whether it is single-chained. If true, return the chain object
    fix_unknown_res = whether or not change the residues with non-standard names into standard names using EXTERNAL_MAPPINGS
    remove_duplicate_res = whether or not remove alternate residues (two residues at the same resnum position) in the chain
    '''
    parser=PDB.PDBParser()
    struc=parser.get_structure("query",path)
    #print(path)
    if len(struc)>1:
        print("Multiple models exist in this pdb file! Only the first model is taken.")
    struc=struc[0]
    try:
        assert len(struc) == 1
    except AssertionError:
        print("Multiple chains exist in this pdb file! Skipping...")
    
    chain=struc.child_list[0]
    if fix_unknown_res:
        deletion=[]
        existing_resnum=[]
        for i in range(len(chain.child_list)):
            if chain.child_list[i].resname in EXTERNAL_MAPPINGS:
                print("Warning: residue %s[%d] is recognized as %s"%(chain.child_list[i].resname,chain.child_list[i].id[1],EXTERNAL_MAPPINGS[chain.child_list[i].resname]))
                chain.child_list[i].resname=EXTERNAL_MAPPINGS[chain.child_list[i].resname]
            elif chain.child_list[i].resname not in toolbox.protein_dict:
                # Removing the unrecognized residues
                print("Warning: Unknown residue encountered: %s[%d]"%(chain.child_list[i].resname,chain.child_list[i].id[1]))
                deletion.append(chain.child_list[i].id)
            if remove_alternate_res:
                if chain.child_list[i].id[1] in existing_resnum:
                    print("Warning: residue %s[%d%s] ignored because it is an alternate residue"%(chain.child_list[i].resname,chain.child_list[i].id[1],chain.child_list[i].id[2]))
                    deletion.append(chain.child_list[i].id)
                else:
                    existing_resnum.append(chain.child_list[i].id[1])
        if len(deletion)>0:
            for item in deletion:
                chain.detach_child(item)
        saver=PDBSaver()
        saver.set_structure(chain)
        basename=os.path.basename(path)
        saver.save(basename.replace(".pdb","_fix.pdb"))
    return chain

def chain_to_seq(chain,fasta_output=None,res_num=True):
    '''
    Accepts a biopython chain object and returns the sequence of that chain
    fasta_output = if not None, output the sequence to a fasta file
    res_num = if true, return the residue numbers in the original PDB file along with the sequence
    '''
    residues=[]
    resnum=[]
    for residue in chain.child_list:
        if residue.resname in toolbox.protein_dict:
            residues.append(residue.resname)
            resnum.append(residue.id[1])
    seq=Seq(toolbox.form_seq(residues))
    if fasta_output:
        record=SeqRecord(seq,id="query",description="")
        with open(fasta_output,"w") as f:
            f.write(record.format("fasta"))
    if res_num:
        return seq,resnum
    else:
        return seq

class blast_result:
    def __init__(self):
        self.target_name=""
        self.score=0
        self.Evalue=0
        self.Lmatch=0 # Matched length
        self.Tmatch=0 # Total length of matched region in matched sequence
        self.source_seq=""
        self.target_seq=""
        self.coverage=0
        self.__last_source_num=0
        self.__last_target_num=0

    def parse(self,line):
        '''
        Function to parse pdb name, blast score and Evalue from a line in blast output file
        '''
        self.target_name=line.split()[0]
        self.score=float(line.split()[-2])
        self.Evalue=float(line.split()[-1])

    def parse_match(self,line):
        '''
        Function to parse matching length from a line in blast output file
        '''
        identity_entry=line.split(",")[0]
        number_entry=[entry for entry in identity_entry.split() if "/" in entry][0]
        self.Lmatch,self.Tmatch=[int(n) for n in number_entry.split("/")]

    def parse_seq(self,line,obj):
        '''
        Functionb to parse blast source/target sequence in a line of the blast output file
        obj = "source" / "target"
        '''
        start_num=int(line.split()[1])
        end_num=int(line.split()[3])
        if obj=="source":
            if self.source_seq=="" or start_num==self.__last_source_num+1:
                self.source_seq+=line.split()[2]
                self.__last_source_num=end_num
        elif obj=="target":
            if self.target_seq=="" or start_num==self.__last_target_num+1:
                self.target_seq+=line.split()[2]
                self.__last_target_num=end_num

    def calc_coverage(self,len_total):
        '''
        Calculate coverage of a blast match. Because the source sequence from blast may be shorter than the real sequence, the total length of the real sequence must be provided.
        '''
        assert self.source_seq!="" and self.target_seq!=""
        # self.coverage=len([i for i in range(len(self.source_seq)) if self.source_seq[i]==self.target_seq[i]])/len_total
        self.coverage=self.Lmatch/max(self.Tmatch,len_total)


def blast(seq,db_name="refDB.blastdb",cleaning=True,return_aligned_seq=False,working_dir=None):
    '''
    Execute a blast query on the sequence and return the blast results
    seq = the query sequence (type: Bio.Seq.Seq) or the path to the fasta file (type: str)
    db_name = the name of blast database
    cleaning = if true, clean the files and folders generated by executing the blast program
    return_aligned_seq = whether include the aligned sequences in the blast_result
    working_dir = a folder to store the intermediate results
    '''
    if working_dir is None:
        working_dir="blast/"
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.mkdir(working_dir)
    if type(seq) is str:
        fasta_name=working_dir+os.path.split(seq)[-1]
        shutil.copy(seq,fasta_name)
    elif type(seq) is Bio.Seq.Seq:
        fasta_name=working_dir+"query.fasta"
        record=SeqRecord(seq,id="query",description="")
        with open(fasta_name,"w") as f:
            f.write(record.format("fasta"))
    cmd=BLAST_DEFAULT_EXE+" -db %s -query %s -out %s  > /dev/null 2>&1"%(db_name,fasta_name,working_dir+"blast.out")
    os.system(cmd)
    results={}
    mode="ignore"
    for line in open(working_dir+"blast.out"):
        if "Sequences producing significant alignments:" in line:
            mode="add_match"
            continue
        elif line[0]==">":
                mode=line.split()[0].replace(">","")
        if mode=="add_match" and line.strip()!="":
            result=blast_result()
            result.parse(line)
            results[result.target_name]=result
        else:
            if mode!="ignore":
                if "Identities =" in line:
                    if results[mode].source_seq=="":
                        results[mode].parse_match(line)
                    else:
                        mode="ignore"
                elif "Query" in line and return_aligned_seq:
                    results[mode].parse_seq(line,"source")
                elif "Sbjct" in line and return_aligned_seq:
                    results[mode].parse_seq(line,"target")
    for identifier in results:
        results[identifier].calc_coverage(len(seq))
    if cleaning:
        shutil.rmtree(working_dir)
    return results

class mTM_align_result:
    def __init__(self,pdbid):
        self.target_name=pdbid
        self.rmsd=0
        self.TMscore=0
        self.source_seq=""
        self.target_seq=""
        self.coverage=0
    
    def parse_alignment(self,source_seq, target_seq):
        '''
        Function to parse the alignment generated by mTM-align (multiple sequences alignment) to the alignments between two
        '''
        assert len(source_seq)==len(target_seq)
        for i in range(len(source_seq)):
            if source_seq[i]=="-" and target_seq[i]=="-":
                continue
            else:
                self.source_seq+=source_seq[i]
                self.target_seq+=target_seq[i]
        self.coverage=len([i for i in range(len(self.source_seq)) if self.source_seq[i]==self.target_seq[i]])/len(self.source_seq)

def mTM_align(source_file,alignment_candidates,db_path=SCRIPT_PATH+"/refDB/pdbs/",cleaning=True,working_dir=None):
    '''
    Execute a multiple structure alignment for the specified source file with the candidate alignment structures using the mTM-alignment algorithm
    source_file = file name of the PDB to be aligned with (type: str)
    alignment_candidates = all candidate alignment PDBIDs (with chain ID) that need to be aligned with (type: List of str)
    db_path = path that all refDB single chain PDB files are stored
    cleaning = if true, clean the files and folders generated by executing mTM-align
    working_dir = a folder to store the intermediate results
    '''
    if working_dir is None:
        working_dir="mTM_align/"
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.mkdir(working_dir)
    shutil.copy(source_file,working_dir+"query.pdb")
    for candidate in alignment_candidates:
        shutil.copy(db_path+candidate.split(".")[1]+".pdb",working_dir+"%s.pdb"%candidate)
    with open(working_dir+"inputs","w") as f:
        f.write("query.pdb\n")
        for candidate in alignment_candidates:
            f.write("%s.pdb\n"%candidate)
    os.chdir(working_dir)
    cmd=MTM_DEFAULT_EXE+" -i inputs"+" > /dev/null 2>&1"
    os.system(cmd)
    results={candidate:mTM_align_result(candidate) for candidate in alignment_candidates}
    with open("pairwise_rmsd.txt") as f:
        title=f.readline() # Read the first line that is the title
        title=[item.replace(".pdb","") for item in title.split()]
        for line in f:
            if "query.pdb" in line:
                wanted_line=line # The line starts with "query.pdb" contains the alignment information that we want
    for rmsd,candidate_pdb in zip(wanted_line.split()[1:],title):
        if candidate_pdb!="query":
            results[candidate_pdb].rmsd=float(rmsd)
    with open("pairwise_TMscore.txt") as f:
        title=f.readline() # Read the first line that is the title
        title=[item.replace(".pdb","") for item in title.split()]
        for line in f:
            if "query.pdb" in line:
                wanted_line=line # The line starts with "query.pdb" contains the alignment information that we want
    for score,candidate_pdb in zip(wanted_line.split()[1:],title):
        if candidate_pdb!="query":
            results[candidate_pdb].TMscore=float(score)
    # Read alignments
    all_alignments=SeqIO.parse("result.fasta","fasta")
    alignment_seqs={}
    for alignment in all_alignments:
        if alignment.id=="query.pdb":
            query_seq=alignment.seq
        else:
            alignment_seqs[alignment.id.replace(".pdb","")]=alignment.seq
    for seq in alignment_seqs:
        results[seq].parse_alignment(query_seq,alignment_seqs[seq])
    os.chdir("../")
    if cleaning:
        shutil.rmtree(working_dir)
    return results

def get_blosum_value(resname1,resname2):
    '''
    Function for acquiring the BLOSUM62 substitution score from res1 to res2
    '''
    code1,code2=toolbox.form_seq([resname1,resname2])
    if (code1,code2) in blosum62:
        return blosum62[(code1,code2)]
    else:
        return blosum62[(code2,code1)]

def Needleman_Wunsch_alignment(seq1,seq2):
    '''
    Function for doing global alignment between seq1 and seq2 using Needleman-Wunsch algorithm implemented in Biopython
    '''
    missing=None
    if "-" in seq1:
        # Need to handle "-" beforehand, otherwise the alignment may fail
        missing=[s=="-" for s in seq1]
        seq1=seq1.replace("-","")
    aligner=Align.PairwiseAligner()
    aligner.open_gap_score=-10
    aligner.extend_gap_score=-0.5
    aligner.substitution_matrix=blosum62
    alignment=aligner.align(seq1,seq2)[0]
    alignment_info=alignment.__str__().split("\n")
    aligned1,aligned2=alignment_info[0],alignment_info[2]
    if missing is None:
        final1=aligned1
        final2=aligned2
    else:
        # Assign alignment with "-"
        final1_temp=""
        final2_temp=""
        j=0
        for s in missing:
            if s:
                final1_temp+="-"
                final2_temp+="-"
            else:
                while aligned1[j]=="-" and j<len(aligned1):
                    final1_temp+=aligned1[j]
                    final2_temp+=aligned2[j]
                    j+=1
                if j<len(aligned1):
                    final1_temp+=aligned1[j]
                    final2_temp+=aligned2[j]
                    j+=1
        if j<len(aligned1):
            final1_temp+=aligned1[j:]
            final2_temp+=aligned2[j:]
        # Cleaning up
        final1=""
        final2=""
        for i in range(len(final1_temp)):
            if not (final1_temp[i]=="-" and final2_temp[i]=="-"):
                final1+=final1_temp[i]
                final2+=final2_temp[i]
    return final1,final2



def assign_aligned_shifts(source_seq,target_seq,target_id,refDB,strict):
    '''
    Function to transfer matched sequence chemical shifts to the source sequence
    source_seq = the input sequence with chemical shifts to be predicted (type: str)
    target_seq = one of the matched sequence with the input sequence (type: str)
    target_id = the PDB identifier (with chain ID) for the target sequence
    refDB = the refDB database containing the shifts for target protein
    strict = strictness level of shift transfer (0 - Strict, 1 - Normal, 2 - Permissive)

    Returns a list, with each element being a dictionary about the chemical shift difference at that residue from random coil value for the different atom types, if a faithful match is found 
    '''
    if (refDB.RES_NUM==np.arange(len(refDB))+1).all():
        # All the residue numbers are consecutive and there is no error
        refDB_seq=toolbox.form_seq(refDB.RESNAME)
    else:
        refDB_seq=""
        start=1
        for i in range(len(refDB)):
            refDB_seq+="-"*(refDB.loc[i,"RES_NUM"]-start)
            start=refDB.loc[i,"RES_NUM"]+1
            refDB_seq+=toolbox.form_seq([refDB.loc[i,"RESNAME"]])
    if len(source_seq)!=len(target_seq):
        # Make sure source seq and target seq are aligned at first place
        source_seq,target_seq=Needleman_Wunsch_alignment(source_seq,target_seq)
    shift_seq,pdb_seq=Needleman_Wunsch_alignment(refDB_seq,target_seq.replace("-",""))
    # source_seq is refDB sequence of matched pdb, target_seq is PDB sequence
    refDB_seq_shifts=[]
    n=0
    for i in range(len(shift_seq)):
        if shift_seq[i]=="-":
            refDB_seq_shifts.append({})
        else:
            residue=toolbox.decode_seq(shift_seq[i])
            assert residue==refDB.iloc[n]["RESNAME"]
            record={atom:refDB.iloc[n][atom] for atom in toolbox.ATOMS}
            record["TARGET_RESNAME"]=residue
            record["TARGET_RESNAME_i+1"]=refDB.iloc[n]["RESNAME_i+1"]
            refDB_seq_shifts.append(record)
            n+=1
    refDB_pdb_shifts=[]
    for i in range(len(shift_seq)):
        if pdb_seq[i]!="-":
            refDB_pdb_shifts.append(refDB_seq_shifts[i])
    query_ref_pdb_shifts=[]
    n=0
    for i in range(len(target_seq)):
        if target_seq[i]=="-":
            query_ref_pdb_shifts.append({})
        else:
            query_ref_pdb_shifts.append(refDB_pdb_shifts[n])
            n+=1
    results=[]
    n=0
    for i in range(len(source_seq)):
        if source_seq[i]!="-":
            shifts=query_ref_pdb_shifts[i]
            try:
                shifts["SOURCE_RESNAME"]=toolbox.decode_seq(source_seq[i])
            except KeyError:
                shifts["SOURCE_RESNAME"]="UNK"
            results.append(shifts)
            n+=1
    # Finally, transfer shifts to the query sequene based on these rules:
    #   1) If the residues are the same, shifts are directly transferred
    #   2) If the target residue is not the same as the source residue, but the BLOSUM substitution score is larger than zero (or using CRAZY GUESS mode), the difference between target shift and random coil values for the target residue is transferred to the source residue
    #   3) If the target residue is not the same as the source residue and the BLOSUM substitution score is smaller than zero (and not in CRAZY GUESS mode), or no matching residues, the shifts for the source residue is not given
    for i in range(len(results)):
        if "TARGET_RESNAME" in results[i]:
            if strict==2 or (strict==1 and get_blosum_value(results[i]["SOURCE_RESNAME"],results[i]["TARGET_RESNAME"])>0) or (strict==0 and results[i]["SOURCE_RESNAME"]==results[i]["TARGET_RESNAME"]):
                if results[i]["TARGET_RESNAME_i+1"]=="PRO":
                    for atom in toolbox.ATOMS:
                        results[i][atom]-=randcoil_pro[atom][results[i]["TARGET_RESNAME"]]
                else:
                    for atom in toolbox.ATOMS:
                        results[i][atom]-=randcoil_ala[atom][results[i]["TARGET_RESNAME"]]
            else:
                results[i]={}
        else:
            results[i]={}
    return results

def main(path,strict,secondary=False,test=False,exclude=False,shifty=False,blast_score_threshold=0,e_value_threshold=1e-10,long_Tmatch_threshold=40,short_Tmatch_threshold=20,long_match_percent_threshold=0.15,short_match_percent_threshold=0.4,TMscore_threshold=0.8,rmsd_threshold=1.75,coverage_threshold=0.3,refDB_shifts_path=SCRIPT_PATH+"/refDB/shifts_df/",custom_working_dir=None):
    '''
    The main function for calculating chemical shifts using shifty++
    path = The path to the pdb file that need to be calculated (type: str)
    strict = Strictness level of shift transfer (0 - Strict, 1 - Normal, 2 - Permissive)
    secondary = Whether or not output secondary shift
    test = Whether or not use the test BLAST database
    exclude = Whether or not use the Exclude mode
    shifty = Whether or not use the SHIFTY mode
    blast_score_threshold = minimum score reported by BLAST program required to be considered as a candidate alignment
    e_value_threshold = maximum expectation value reported by BLAST program required to be considered as a candidate alignment
    long_Tmatch_threshold = the minimum total length of the matched sequence for applying long_match_percent_threshold
    long_match_percent_threshold = minimum percentage coverage of the exact matching residues with the matched sequence for a long match
    short_Tmatch_threshold = the minimum total length of the matched sequence for applying short_match_percent_threshold
    short_match_percent_threshold = minimum percentage coverage of the exact matching residues with the matched sequence for a short match
    TMscore_threshold = minimum mTM-align score required to be considered as a good alignment
    rmsd_threshold = maximum RMSD from mTM-align permitted between two alignment
    coverage_threshold = minimum coverage for mTM-align candidates to be considered in final shift transfer
    refDB_shifts_path = path to all the csv file storing the refDB shifts
    custom_working_dir = when specified, use this folder to process blast and mTM-align

    returns a pandas.DataFrame containing all the calculated shifts
    '''
    fixname=os.path.basename(path).replace(".pdb","_fix.pdb")
    seq,resnum=chain_to_seq(read_sing_chain_PDB(path))
    blast_result=blast(seq,db_name="train.blastdb" if test else "refDB.blastdb",return_aligned_seq=True,cleaning=not DEBUG,working_dir=custom_working_dir) # Only the SHIFTY mode needs the aligned sequence
    candidates=[]
    for result in blast_result.values():
        if result.score>=blast_score_threshold and result.Evalue<=e_value_threshold:
            # Those pass the selection criterion
            if result.Tmatch>=long_Tmatch_threshold and result.Lmatch/result.Tmatch>=long_match_percent_threshold:
                # Long matches
                if not exclude:
                    candidates.append(result)
                else:
                    if not result.coverage > GLOBAL_TEST_CUTOFF:
                        candidates.append(result)
            elif result.Tmatch>=short_Tmatch_threshold and result.Lmatch/result.Tmatch>=short_match_percent_threshold:
                # Short matches
                if not exclude:
                    candidates.append(result)
                else:
                    if not result.coverage > GLOBAL_TEST_CUTOFF:
                        candidates.append(result)
                
    if len(candidates)==0:
        residues=toolbox.decode_seq(seq)
        result_dict={"RESNAME":residues,"RESNUM":resnum}
        df=pd.DataFrame(result_dict)
        for atom in toolbox.ATOMS:
            df[atom]=np.nan           
        print("No sequence in database generates possible alignments")
        if os.path.exists(fixname):
            os.remove(fixname)
        for atom in toolbox.ATOMS:
            df[atom]=np.nan     
            df[atom+"_BEST_REF_SCORE"]=0
            df[atom+"_BEST_REF_COV"]=0
            df[atom+"_BEST_REF_MATCH"]=0
        return df
    final=[]
    identities=[]
    if shifty:
        # In SHIFTY mode, do not do structural alignment
        best_match=np.argmax([item.score for item in candidates])
        final.append(candidates[best_match])
        identities.append(candidates[best_match].coverage)
        if os.path.exists(fixname):
            os.remove(fixname)
    else:
        # Do mTM alignment
        candidates=[item.target_name for item in candidates]
        if os.path.exists(fixname):
            mtm_results=mTM_align(fixname,candidates,cleaning=not DEBUG,working_dir=custom_working_dir)  
            os.remove(fixname)
        else:
            mtm_results=mTM_align(path,candidates,cleaning=not DEBUG,working_dir=custom_working_dir) 
        blast_scores=[]
        for result in mtm_results.values():
            if result.TMscore>TMscore_threshold and result.rmsd<rmsd_threshold and result.coverage>coverage_threshold:
                identity=blast_result[result.target_name].coverage
                final.append(result)
                identities.append(identity)
                blast_scores.append(blast_result[result.target_name].score)
        # Calculate normalized blast scores so that it can be considered together with TM scores
        if len(blast_scores)>0:
            normalized_blast_scores=np.array(blast_scores)/np.max(blast_scores)
    if len(final)==0:
        residues=toolbox.decode_seq(seq)
        result_dict={"RESNAME":residues,"RESNUM":resnum}
        df=pd.DataFrame(result_dict)
        print("No significant structure alignment is possible!")
        for atom in toolbox.ATOMS:
            df[atom]=np.nan           
            df[atom+"_BEST_REF_SCORE"]=0
            df[atom+"_BEST_REF_COV"]=0
            df[atom+"_BEST_REF_MATCH"]=0
        return df
    print("Calculating using %d references with maximal identity %.2f"%(len(final),np.max(identities)))
    refDB={}
    for item in final:
        refDB[item.target_name]=pd.read_csv(refDB_shifts_path+item.target_name+".csv")
    # changed candidate.source_seq->seq
    candidate_shifts=[assign_aligned_shifts(seq,candidate.target_seq,candidate.target_name,refDB[candidate.target_name],strict) for candidate in final]
    if shifty:
        scores=[1]
    else:
        scores=[candidate.TMscore*normalized_blast_scores[idx] for idx,candidate in enumerate(final)]
    seq_shifts=[]
    # Fix problems when the sequence read by Biopython is not the same as the sequence generated by mTM-align
    mtm_recognized_seq=final[0].source_seq.replace("-","")
    if not shifty and mtm_recognized_seq!=str(seq):
        biopython_seq,mtm_seq=Needleman_Wunsch_alignment(str(seq),mtm_recognized_seq)
        biopython_seq=list(biopython_seq)
        for i in range(len(mtm_seq)):
           if mtm_seq[i]=="-" and biopython_seq[i]!="-":
               biopython_seq[i]="x" # Marked for deletion
        biopython_seq="".join(biopython_seq)
        biopython_seq.replace("-","") 
        seq=""
        old_resnum=resnum
        resnum=[]
        for i in range(len(biopython_seq)):
            if biopython_seq[i]!="x":
                seq+=biopython_seq[i]
                resnum.append(old_resnum[i])
    for i in range(len(seq)):
        resname=toolbox.decode_seq(seq[i])
        next_pro=False
        if i+1<len(resnum) and resnum[i+1]==resnum[i]+1:
            if seq[i+1]=="P": # the next residue in the query protein is proline
                next_pro=True
        residue_shifts={}
        residue_shifts["RESNAME"]=resname
        residue_shifts["RESNUM"]=resnum[i]
        for atom in toolbox.ATOMS:
            shifts=[]
            reference_scores=[]
            res_scores=[] # probably need to do some non-linear conversion of the scores
            for candidate_shift,score in zip(candidate_shifts,scores):
                if atom in candidate_shift[i] and not np.isnan(candidate_shift[i][atom]):
                    shifts.append(candidate_shift[i][atom])
                    target_score=np.exp(score*5)*np.exp(get_blosum_value(candidate_shift[i]["SOURCE_RESNAME"],candidate_shift[i]["TARGET_RESNAME"]))
                    res_scores.append(target_score)
                    reference_scores.append(target_score)
                else:
                    reference_scores.append(0)
            if len(shifts)>0:
                # calculate weighted average for the specific residue based on mTM alignment scores and BLOSUM numbers
                rc_diff=np.sum(np.array(shifts)*np.array(res_scores))/np.sum(res_scores)
                # capping rc_diff to prevent making crazy errors due to bad database records
                rc_diff = min(rc_diff, SS_CAPS[atom])
                rc_diff = max(rc_diff, -SS_CAPS[atom])
                if next_pro:
                    if secondary:
                        residue_shifts[atom]=rc_diff
                        residue_shifts[atom+"_RC"]=randcoil_pro[atom][resname]
                    else:
                        residue_shifts[atom]=rc_diff+randcoil_pro[atom][resname]
                else:
                    if secondary:
                        residue_shifts[atom]=rc_diff
                        residue_shifts[atom+"_RC"]=randcoil_ala[atom][resname]
                    else:
                        residue_shifts[atom]=rc_diff+randcoil_ala[atom][resname]
            else:
                residue_shifts[atom]=np.nan
            max_ref = np.argmax(reference_scores)
            if reference_scores[max_ref]>0:
                residue_shifts[atom+"_BEST_REF_SCORE"]=final[max_ref].TMscore * np.exp(-final[max_ref].rmsd) 
                residue_shifts[atom+"_BEST_REF_COV"]=min(final[max_ref].coverage,identities[max_ref])
                residue_shifts[atom+"_BEST_REF_MATCH"]=int(candidate_shifts[max_ref][i]["SOURCE_RESNAME"] == candidate_shifts[max_ref][i]["TARGET_RESNAME"])
            else:
                residue_shifts[atom+"_BEST_REF_SCORE"]=0
                residue_shifts[atom+"_BEST_REF_COV"]=0
                residue_shifts[atom+"_BEST_REF_MATCH"]=0
        seq_shifts.append(residue_shifts)
    result=pd.DataFrame(seq_shifts)
    # result=result[["RESNUM", "RESNAME"] + toolbox.ATOMS + ["MAX_IDENTITY", "AVG_IDENTITY"]]
    return result

if __name__=="__main__":
    args=argparse.ArgumentParser(description="This program executes both sequence-based alignment (using BLAST) and structure-based alignment (using mTM-align) to find the best alignment for a specific pdb file with entities in the refDB database, and use the average chemical shifts from refDB to predict the chemical shifts for backbone H/C/N atom chemical shifts for the query protein")
    args.add_argument("input",help="The query PDB file for which the shifts are calculated")
    args.add_argument("--output", "-o",help="Filename of generated output file. A file [shifts.csv] is generated by default",default="shifts.csv")
    args.add_argument("--strict","-s",help="Strict level of shift transfer:\n\t0 - Strict, only the exact matching residue shifts are transferred\n\t1 - Normal, transfer the shifts for residues that are the same or have positive substitution scores (from BLOSUM62)\n\t2 - Permissive, transfer all shifts regardless of the likeliness of substitution.",type=int,default=1)
    args.add_argument("--secondary","-2",help="If this flag is set, the output will be secondary shifts (observed shifts-random coil shifts) instead of observed shifts",action="store_true",default=False)
    args.add_argument("--test","-t",help="If this flag is set, the test BLAST database is used, which means all sequences in the validation set and test set will not be included in the BLAST search database",action="store_true",default=False)
    args.add_argument("--exclude","-e",help="Exclude mode, another way of analyzing the performance of SHIFTY++. When selecting sequences going to the structure alignment, those completely identical examples are excluded.",action="store_true",default=False)
    args.add_argument("--shifty","-y",help="SHIFTY mode, only the top hit from sequence alignment is considered for shift transfer",action="store_true",default=False)
    args=args.parse_args()
    result=main(args.input,strict=args.strict,secondary=args.secondary,test=args.test,exclude=args.exclude,shifty=args.shifty)  
    result.to_csv(args.output,index=None)




