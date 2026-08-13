#/usr/bin/env python
# Script for generating .csv format datasets for training and testing 

# Author: Jie Li, Michael Martin
# Date created: Feb 20, 2019

import os
import os.path
import shutil
import sys
sys.path.append('../')
import toolbox
import pynmrstar
import pandas as pd
import numpy as np
import glob
from Bio.SeqUtils import IUPACData
from Bio.SubsMat.MatrixInfo import blosum62
from spartap_features import *
import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)
import pickle
import multiprocessing
WORKER_COUNT=10
import random

pynmrstar.ALLOW_V2_ENTRIES = True
PDB_FOLDER="pdbs/"
DATASET_FOLDER="../datasets/"

atom_names = ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE3', 'CE2', 'CG', 'CG1', 'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3', 'H', 'HA', 'HB', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE3', 'HE21', 'HE22', 'HG', 'HG1', 'HG12', 'HG13', 'HG2', 'HG3', 'HH2', 'HZ', 'HZ2', 'HZ3', 'N', 'ND2', 'NE1', 'NE2']

amino_acids = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}


# Extract shifts files if they are not extracted
if not os.path.exists("shifts"):
    print("Extracting shift files...")
    os.system("tar -xzf shifts.tgz")

# Make sure the folder for storing training datasets exist
if not os.path.exists(DATASET_FOLDER):
    os.mkdir(DATASET_FOLDER)


def get_shifts(file,format="talos"):
    '''
        Reads a file in talos (.tab) format or in nmrstar (.nmrstar) format and returns a 2-d dictionary {RESID,ATOM:shift}
    '''
    shift_dict = {}
    if format=="talos":
        with open(file) as f:
            yang_file = f.read()
            for line_num, line in enumerate(yang_file.split('\n')):
                if 'FORMAT' in line:
                    start_line = line_num + 2
                if 'S2' in line:
                    end_line = line_num
                if 'pH' in line:
                    pH_line=line_num
            # Extract shifts
            for line in yang_file.split('\n')[start_line:end_line]:
                if len(line)==0:
                    continue
                resid,resname,atomname,shift=line.split()
                resid=int(resid)
                shift=float(shift)
                if resid not in shift_dict.keys():
                    shift_dict[resid] = {"RES":resname}   
                shift_dict[resid][atomname] = shift     
            

            # Extract RCI-S2 values
            for line in yang_file.split("\n")[end_line+1:pH_line]:
                if len(line)==0:
                    continue
                resid,s2,resname=line.split()
                resid=int(resid)
                s2=float(s2)
                if shift_dict[resid]["RES"]==resname:
                    shift_dict[resid]["RCI_S2"]=s2
                else:
                    print("RESID %d not match for S2!"%resid)
                    shift_dict[resid]["RCI_S2"]=np.nan
                #shift_dict[resid].pop("RES") 
            # Extract pH
            pH=eval(yang_file.split('\n')[pH_line+1])
    # Do some renaming and averaging stuff to keep consistency of atom definitions
    for res in shift_dict.keys():
        if 'HA2' in shift_dict[res].keys() and 'HA3' in shift_dict[res].keys():
            average = 0.5*(shift_dict[res]['HA2']+shift_dict[res]['HA3'])
            shift_dict[res]['HA'] = average
            shift_dict[res].pop("HA2")
            shift_dict[res].pop("HA3")
        elif 'HA2' in shift_dict[res].keys():
            shift_dict[res]['HA'] = shift_dict[res]['HA2']
            shift_dict[res].pop("HA2")
        elif 'HA3' in shift_dict[res].keys():
            shift_dict[res]['HA'] = shift_dict[res]['HA3']
            shift_dict[res].pop("HA3")
        if 'HN' in shift_dict[res].keys():
            shift_dict[res]['H'] = shift_dict[res]['HN']
            shift_dict[res].pop("HN")

    return shift_dict,pH




def build_single_chain_df(pdb_file_name,shift_dict,alignment=None,pH=5,rcfeats=True, hse=True, first_chain_only=False, sequence_columns=0,hbrad=[5.0]*3):
    '''
        Function for building dataframe for pdb-shift combination.
        Returns a pandas dataframe for a specific single-chain pdb file.
    '''
    # Build feature reader and read all features
    feature_reader = PDB_SPARTAp_DataReader()
    pdb_df = feature_reader.df_from_file_3res(pdb_file_name,rcshifts=rcfeats, hse=hse, first_chain_only=first_chain_only, sequence_columns=sequence_columns,hbrad=hbrad)
    if alignment is not None: 
    # Working in sparta+ way
        for atom in atom_names:
            pdb_df[atom] = ''
        pdb_df['MATCHED_BMRB'] = 0
        for index, row in pdb_df.iterrows():
            chain = row['CHAIN']
            bmrb_seq = alignment[chain]['bmrb_alignment']
            bmrb_res_list = alignment[chain]['bmrb_res_list']
            pdb_seq = alignment[chain]['pdb_alignment']
            pdb_res_list = alignment[chain]['pdb_res_list']
            pdb_res_num_i = row['RES_NUM']
            pdb_res_list_index_i = pdb_res_list.index(pdb_res_num_i)

            seq_index_i = [j for j in range(len(pdb_seq)) if len(pdb_seq[0:j+1].replace('-', ''))==pdb_res_list_index_i+1][0]

            if seq_index_i==0:
                if (pdb_seq[seq_index_i] == bmrb_seq[seq_index_i]) and (pdb_seq[seq_index_i+1] == bmrb_seq[seq_index_i+1]):
                    pdb_df.loc[index, 'MATCHED_BMRB'] = 1
                    bmrb_seq_segment = bmrb_seq[0:seq_index_i+1]
                    bmrb_resnum = int(len([i for i in bmrb_seq_segment if i!='-']))
                    pdb_df.loc[index, 'BMRB_RES_NUM'] = bmrb_resnum
                    if bmrb_resnum in bmrb_res_list:
                        for atom in shift_dict[bmrb_resnum].keys():
                            pdb_df.loc[index, atom] = shift_dict[bmrb_resnum][atom]

            elif seq_index_i==len(bmrb_seq)-1:
                if (pdb_seq[seq_index_i] == bmrb_seq[seq_index_i]) and (pdb_seq[seq_index_i-1] == bmrb_seq[seq_index_i-1]):
                    pdb_df.loc[index, 'MATCHED_BMRB'] = 1
                    bmrb_seq_segment = bmrb_seq[0:seq_index_i+1]
                    bmrb_resnum = int(len([i for i in bmrb_seq_segment if i!='-']))
                    pdb_df.loc[index, 'BMRB_RES_NUM'] = bmrb_resnum
                    if bmrb_resnum in bmrb_res_list:
                        for atom in shift_dict[bmrb_resnum].keys():
                            pdb_df.loc[index, atom] = shift_dict[bmrb_resnum][atom]
                            
                            
            else:
                if (pdb_seq[seq_index_i-1] == bmrb_seq[seq_index_i-1]) and (pdb_seq[seq_index_i] == bmrb_seq[seq_index_i]) and (pdb_seq[seq_index_i+1] == bmrb_seq[seq_index_i+1]):
                    pdb_df.loc[index, 'MATCHED_BMRB'] = 1
                    bmrb_seq_segment = bmrb_seq[0:seq_index_i+1]
                    bmrb_resnum = int(len([i for i in bmrb_seq_segment if i!='-']))
                    pdb_df.loc[index, 'BMRB_RES_NUM'] = bmrb_resnum
                    if bmrb_resnum in bmrb_res_list:
                        for atom in shift_dict[bmrb_resnum].keys():
                            pdb_df.loc[index, atom] = shift_dict[bmrb_resnum][atom]

                        
    else:
        # working in shiftx2 way
        shift_df=pd.DataFrame.from_dict(shift_dict,orient='index')
        kept_cols=atom_names+["RCI_S2"]+['RES']
        shift_df['RES'] = shift_df['RES'].map(amino_acids)
        kept_cols=list(set(kept_cols)&set(shift_df.columns))
        shift_df=shift_df[kept_cols]
        for atom_name in atom_names+["RCI_S2"]:
            if atom_name not in kept_cols:
                shift_df[atom_name]=np.nan
        shift_df["RES_NUM"]=shift_df.index
        pdb_df["RES_NUM"]=pdb_df["RES_NUM"].astype(int)
        pdb_df = pdb_df.merge(shift_df, on='RES_NUM',how='left')
    
    pdb_df["pH"]=pH
    return pdb_df

def build_test_input(pdb_file_name,pH=5,rcfeats=True, hse=True, first_chain_only=False, sequence_columns=20,hbrad=[5.0]*3):
    '''
        Function for building dataframe for testing set (so that no shifts are given)
        Returns a pandas dataframe for a specific single-chain pdb file.
    '''
    # Build feature reader and read all features
    feature_reader = PDB_SPARTAp_DataReader()

    
    pdb_df = feature_reader.df_from_file_3res(pdb_file_name,rcshifts=rcfeats, hse=hse, first_chain_only=first_chain_only, sequence_columns=sequence_columns,hbrad=hbrad)
    pdb_df["pH"]=pH
    return pdb_df

def build_spartap(seq_alignment_dict):
    '''
    Generate all dataframe files for SPARTA+ pdbs
    '''
    for pdb_bmrb_id in seq_alignment_dict.keys():
        print('Processing SPARTA+ structure: '+pdb_bmrb_id)
        bmrbid,pdbid=pdb_bmrb_id.split(".")
        pdb_single_chain_files=[item for item in os.listdir(PDB_FOLDER+"train/") if pdbid in item]
        # Make sure there is only one match
        if len(pdb_single_chain_files)!=1:
            print("Unexpected file numbers for %s"%pdb_bmrb_id)
            continue
        else:
            pdb_single_chain_file=pdb_single_chain_files[0]
        if os.path.exists(DATASET_FOLDER+"train/"+os.path.basename(pdb_single_chain_file).replace(".pdb",".csv")):
            continue
        shifts,pH=get_shifts("shifts/"+bmrbid+".tab")
        
        df=build_single_chain_df(PDB_FOLDER+"train/"+pdb_single_chain_file,shifts,seq_alignment_dict[pdb_bmrb_id],pH=pH)
        df.to_csv(DATASET_FOLDER+os.path.basename(pdb_single_chain_file).replace(".pdb",".csv"))

def build_shiftx2(pdb_to_shift_dict):
    '''
    Generate all dataframe files for SHIFTX2 pdbs
    '''
    for pdbid in pdb_to_shift_dict:
        pid=pdbid[:5]
        print("Processing SHIFTX2 structure: "+pid)
        pdb_single_chain_files=[item for item in os.listdir(PDB_FOLDER+"train/") if pid in item]

        if len(pdb_single_chain_files)!=1:
            print("Ignored %s"%pid)
            continue
        else:
            pdb_single_chain_file=pdb_single_chain_files[0]
        if os.path.exists(DATASET_FOLDER+"train/"+os.path.basename(pdb_single_chain_file).replace(".pdb",".csv")):
            continue
        shift_file=pdb_to_shift_dict[pdbid].split(".")[0]+".tab"
        shifts,pH=get_shifts("shifts/"+shift_file)
        
        df=build_single_chain_df(PDB_FOLDER+"train/"+pdb_single_chain_file,shifts,pH=pH)
        
        # remove shifts values if residues in crystal structure and nmr file are different
        df['Values_Same'] = np.where((df['RESNAME'] == df['RES']) | (df['RES'].isna()) | (df['RESNAME'].isna()), True, False)
        if False in df['Values_Same'].values:
            df.loc[df['Values_Same'] == False, atom_names] = None

        df = df.drop(columns=['Values_Same','RES'])
        df.to_csv(DATASET_FOLDER+os.path.basename(pdb_single_chain_file).replace(".pdb",".csv"))

            

def build_refdb_test(pdb_bmr_dict):
    '''
    Generate feature-only dataframe files for test pdbs
    '''
    for pdbid in pdb_bmr_dict:
        pid=pdbid[:4]
        print("Processing refDB testing structure: "+pid)
        file_pos=PDB_FOLDER+"test/%s.pdb"%pdbid
        if os.path.exists(file_pos):
            shift_pos="shifts/bmr%d.str.corr"%pdb_bmr_dict[pdbid]
            pH=toolbox.get_pH(shift_pos)
            
            try:
                df=build_test_input(file_pos,pH)
            except Exception as e:
                #print(pid,e)
                with open("error.log","a") as f:
                    f.write("%s\t%s\n"%(pid,e))
            df.to_csv("../test/%s.csv"%pdbid)
        else:
            print("Cannot find",pid)
            continue



def build_new(pdb_bmr_dict):
    '''
    Generate all dataframe files for new pdbs
    '''
    best_entity='1'
    
    for pdbid in pdb_bmr_dict:
        pid=pdbid[:5]
        print("Processing refDB training structure: "+pid)
        file_pos=PDB_FOLDER+"train/" + pdbid
        if os.path.exists(file_pos):
            shift_pos="shifts/" + pdb_bmr_dict[pdbid]
            pH=toolbox.get_pH(shift_pos)
            df_features=build_test_input(file_pos,pH=pH)           
            bmrid = pdb_bmr_dict[pdbid][3:-9]
            try:
                entry = pynmrstar.Entry.from_database(bmrid)
            except OSError:
                print(bmrid, " is not in the database")
                pass
            cs = entry.get_loops_by_category('Atom_chem_shift')[0].get_tag(['Entity_id', 'Seq_ID', 'Comp_ID', 'Atom_ID', 'Val'])    
            cs = [ [i[1], i[2], i[3], i[4]]  for i in cs if i[0] == best_entity]    

            cs = [[pid] + sublist for sublist in cs]
                
            df = pd.DataFrame(cs, columns=['FILE_ID', 'RES_NUM', 'RESNAME', 'Atom_ID', 'Val']) 
            df['Val'] = df['Val'].astype(float) 
            df['RES_NUM'] = df['RES_NUM'].astype(int) 
            ptdf = pd.pivot_table(df, index=['FILE_ID','RES_NUM','RESNAME'], columns='Atom_ID', values='Val') 
            ptdf = ptdf.reindex(columns=atom_names) 
            ptdf.fillna(value=np.nan, inplace=True)  
            ptdf.reset_index(inplace=True)  
            

            try:    
                ptdf.loc[ptdf['RESNAME'] == 'LYS', 'HZ'] = ptdf.loc[ptdf['RESNAME'] == 'LYS', 'HZ1']
                ptdf.loc[ptdf['RESNAME'] == 'LYS', 'HZ2'] = np.nan
                ptdf.loc[ptdf['RESNAME'] == 'LYS', 'HZ3'] = np.nan
            except KeyError:
                pass    
            
            try:
                ptdf.loc[ptdf['RESNAME'] == 'ILE', 'HG2'] = ptdf.loc[ptdf['RESNAME'] == 'ILE', 'HG21']
                ptdf.loc[ptdf['RESNAME'] == 'ILE', 'HD1'] = ptdf.loc[ptdf['RESNAME'] == 'ILE', 'HD11']
            except KeyError:
                pass
            
            try:
                ptdf.loc[ptdf['RESNAME'] == 'LEU', 'HD1'] = ptdf.loc[ptdf['RESNAME'] == 'LEU', 'HD11']
                ptdf.loc[ptdf['RESNAME'] == 'LEU', 'HD2'] = ptdf.loc[ptdf['RESNAME'] == 'LEU', 'HD21']
                ptdf.loc[ptdf['RESNAME'] == 'LEU', 'HD21'] = np.nan
                ptdf.loc[ptdf['RESNAME'] == 'LEU', 'HD22'] = np.nan
            except KeyError:
                pass     
            
            try:
                ptdf.loc[ptdf['RESNAME'] == 'VAL', 'HG1'] = ptdf.loc[ptdf['RESNAME'] == 'VAL', 'HG11']
                ptdf.loc[ptdf['RESNAME'] == 'VAL', 'HG2'] = ptdf.loc[ptdf['RESNAME'] == 'VAL', 'HG21']
                ptdf.loc[ptdf['RESNAME'] == 'VAL', 'HG12'] = np.nan
                ptdf.loc[ptdf['RESNAME'] == 'VAL', 'HG13'] = np.nan
            except KeyError:
                pass 
            
            try:
                ptdf.loc[ptdf['RESNAME'] == 'THR', 'HG2'] = ptdf.loc[ptdf['RESNAME'] == 'THR', 'HG21']   
            except KeyError:
                pass       
               
            try:
                ptdf.loc[ptdf['RESNAME'] == 'MET', 'HE'] = ptdf.loc[ptdf['RESNAME'] == 'MET', 'HE1']
                ptdf.loc[ptdf['RESNAME'] == 'MET', 'HE1'] = np.nan
                ptdf.loc[ptdf['RESNAME'] == 'MET', 'HE2'] = np.nan
                ptdf.loc[ptdf['RESNAME'] == 'MET', 'HE3'] = np.nan
            except KeyError:
                pass    
            
            
            try:
                ptdf.loc[ptdf['RESNAME'] == 'ALA', 'HB'] = ptdf.loc[ptdf['RESNAME'] == 'ALA', 'HB1']
                ptdf.loc[ptdf['RESNAME'] == 'ALA', 'HB2'] = np.nan
                ptdf.loc[ptdf['RESNAME'] == 'ALA', 'HB3'] = np.nan
            except KeyError:
                pass  
            
            try:
                ptdf.drop(columns=['HZ1', 'HD11', 'HD12', 'HD13', 'HG21', 'HG22', 'HG23', 'HD23', 'HG11', 'HB1'], inplace=True)        
            except KeyError:
                pass 

            # Calculate average for 'HA2' and 'HA3' and replace with 'HA'
            if ptdf['HA'].empty:
                if 'HA2' in ptdf.columns and 'HA3' in ptdf.columns:
                    ha2_values = ptdf['HA2'].values
                    ha3_values = ptdf['HA3'].values
                    ha_average = 0.5*(ha2_values + ha3_values)
                    ptdf['HA'] == ha_average.round(2)

            combined_df = df_features.merge(ptdf, on=['RES_NUM', 'RESNAME'], how='left')
            combined_df.to_csv(DATASET_FOLDER +"%s.csv"%pid)

        else:
            print("Cannot find",pid)
            continue  





if __name__=="__main__":
    # Load pickle files to obtain matching between pdbs and shifts
    with open('seq_alignment_dict.pkl', 'rb') as f:
        seq_alignment_dict = pickle.load(f)
    

    with open("test200.pkl","rb") as f:        
        pdb_bmr_dict=pickle.load(f)


    with open("pdb_to_shift_dict-old.pkl","rb") as f:
        pdb_to_shift_dict_old=pickle.load(f)


    with open("pdb_to_shift_dict-new.pkl","rb") as f:
        pdb_to_shift_dict_new=pickle.load(f)



    #############Parallel execute whole building
    pool=multiprocessing.Pool(processes=WORKER_COUNT)
    
    seq_alignment_dict_list = [dict() for i in range(WORKER_COUNT)]
    keys_to_pop = ['11013.3ERR','15388.1WRK', 'malate.1D8C', '4083.1JBE']
    for key in keys_to_pop:
        if key in seq_alignment_dict:
            seq_alignment_dict.pop(key)
    worker_idx = 0
    for sa_key in seq_alignment_dict.keys():
        seq_alignment_dict_list[worker_idx][sa_key] = seq_alignment_dict[sa_key]
        if worker_idx==WORKER_COUNT-1:
            worker_idx = 0
        else:
            worker_idx+=1
    pool.map(build_spartap, seq_alignment_dict_list)
    print("Finishes sparta+ data")
    
    '''
    pdb_bmr_dict_list = [dict() for i in range(WORKER_COUNT)]
    keys_to_pop = ['2HWNB']
    for key in keys_to_pop:
        if key in pdb_bmr_dict:
            pdb_bmr_dict.pop(key)    
    worker_idx = 0
    for sa_key in pdb_bmr_dict.keys():
        pdb_bmr_dict_list[worker_idx][sa_key] = pdb_bmr_dict[sa_key]
        if worker_idx==WORKER_COUNT-1:
            worker_idx = 0
        else:
            worker_idx+=1
    pool.map(build_refdb_test, pdb_bmr_dict_list)
    print("Finishes refDB testing data")
    '''

    pdb_to_shift_dict_new_list = [dict() for i in range(WORKER_COUNT)]
    keys_to_pop = ['108M_.pdb','1KMVA.pdb','1EXP_.pdb','4CROB.pdb','2LZH_.pdb','1QD7C.pdb']
    for key in keys_to_pop:
        if key in pdb_to_shift_dict_new:
            pdb_to_shift_dict_new.pop(key)
    worker_idx = 0
    for sa_key in pdb_to_shift_dict_new.keys():
        pdb_to_shift_dict_new_list[worker_idx][sa_key] = pdb_to_shift_dict_new[sa_key]
        if worker_idx==WORKER_COUNT-1:
            worker_idx = 0
        else:
            worker_idx+=1
    pool.map(build_new, pdb_to_shift_dict_new_list)
    print("Finishes new data")
    

    pdb_to_shift_dict_old_list = [dict() for i in range(WORKER_COUNT)]
    #keys_to_pop = ['108M_.pdb','1KMVA.pdb','1EXP_.pdb']
    #for key in keys_to_pop:
    #    if key in pdb_to_shift_dict_old:
    #        pdb_to_shift_dict_old.pop(key)
    worker_idx = 0
    for sa_key in pdb_to_shift_dict_old.keys():
        pdb_to_shift_dict_old_list[worker_idx][sa_key] = pdb_to_shift_dict_old[sa_key]
        if worker_idx==WORKER_COUNT-1:
            worker_idx = 0
        else:
            worker_idx+=1
    pool.map(build_shiftx2, pdb_to_shift_dict_old_list)
    print("Finishes shiftx2 data")



    
    #############Sequential execute whole building
    # build_refdb_test(pdb_bmr_dict)
    # build_shiftx2(pdb_to_shift_dict)
    # build_spartap(seq_alignment_dict)

    

