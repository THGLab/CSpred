#/usr/bin/env python
# Script for PDB protonation using PDB2PQR

# Author: Aleksandra Ptaszek
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
WORKER_COUNT=1
import random

pynmrstar.ALLOW_V2_ENTRIES = True
PDB_FOLDER="pdbs/"



# Extract shifts files if they are not extracted
if not os.path.exists("shifts"):
    print("Extracting shift files...")
    os.system("tar -xzf shifts.tgz")


def filter_file(filename, string_to_remove):
    # Read the original file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Filter out lines that contain the specified string
    filtered_lines = [line for line in lines if string_to_remove not in line]

    # Write the filtered lines back to the same file
    with open(filename, 'w') as f:
        f.writelines(filtered_lines)
        
        

def get_ph(file,format="talos"):
    if format=="talos":
        with open(file) as f:
            yang_file = f.read()
            for line_num, line in enumerate(yang_file.split('\n')):
                if 'pH' in line:
                    pH_line=line_num
            # Extract pH
            pH=eval(yang_file.split('\n')[pH_line+1])
    return pH



def protonate_spartap(seq_alignment_dict):
    for pdb_bmrb_id in seq_alignment_dict.keys():
        print('Processing SPARTA+ structure: '+pdb_bmrb_id)
        bmrbid,pdbid=pdb_bmrb_id.split(".")
        pdb_single_chain_files=[item for item in os.listdir(PDB_FOLDER+"train/") if pdbid in item]
        
        # PDB cleaning
        if pdb_single_chain_files:
            file_pos=PDB_FOLDER+"train/%s.pdb"%pdb_single_chain_files[0][:5]
            print(file_pos)
            lines_to_remove = {'2VYIB':" 963  N   ALA B 209", '1BNJA':"1  C   GLN A   2", '1O8XA':"1144  N   PRO A 146",
                               '2BF2A':"800  N   MET A 102", '2VQES':"648  N   GLY S  82", '2VN1B':"969  N   GLU B 129",
                               '1TOLA':"1172  N   ALA A 216"}
            for key, value in lines_to_remove.items():
                if pdb_single_chain_files[0][:5]==key:
                    filter_file(file_pos,value)
        
        # Make sure there is only one match
        if len(pdb_single_chain_files)!=1:
            print(pdb_single_chain_files)
            print("Unexpected file numbers for %s"%pdb_bmrb_id)
            continue
        else:
            pdb_single_chain_file=pdb_single_chain_files[0]
        pH=get_ph("shifts/"+bmrbid+".tab")
        
        toolbox.propka_prot(PDB_FOLDER+"train/"+pdb_single_chain_file,pH)



def protonate_shiftx2(pdb_to_shift_dict):

    for pdbid in pdb_to_shift_dict:
        pid=pdbid[:5]
        print("Processing SHIFTX2 structure: "+pid)
        
        # PDB cleaning
        file_pos=PDB_FOLDER+"train/%s.pdb"%pdbid[:5]
        lines_to_remove = {'1W41A':"749  N   LYS A  99", '1SNMA':"1082  N   GLU A 142", 
                           '2BF5B':"726  N   MET B 102", '5PTIA':" D", 
                           '1H7MA':"729  N   GLY A  97", '1F4PA':"USER"}
        for key, value in lines_to_remove.items():
            if pdbid[:5]==key:
                filter_file(file_pos,value)
                
        pdb_single_chain_files=[item for item in os.listdir(PDB_FOLDER+"train/") if pid in item]
        if len(pdb_single_chain_files)!=1:
            print("Ignored %s"%pid)
            continue
        else:
            pdb_single_chain_file=pdb_single_chain_files[0]

        shift_file=pdb_to_shift_dict[pdbid].split(".")[0]+".tab"
        pH=get_ph("shifts/"+shift_file)

        toolbox.propka_prot(PDB_FOLDER+"train/"+pdb_single_chain_file,pH)

            





def protonate_refdb_test(pdb_bmr_dict):
    for pdbid in pdb_bmr_dict:
        pid=pdbid[:4]
        print("Processing refDB testing structure: "+pid)
        file_pos=PDB_FOLDER+"test/%s.pdb"%pdbid
        # PDB cleaning
        lines_to_remove = {'1CY5A':"ATOM    743  N   VAL A  93", '1UWXA':"N   LYS A  62", '1OKSA':'416  N   HIS A  55'}
        for key, value in lines_to_remove.items():
            if pdbid[:5]==key:
                filter_file(file_pos,value)

        if os.path.exists(file_pos):
            shift_pos="shifts/bmr%d.str.corr"%pdb_bmr_dict[pdbid]
            pH=toolbox.get_pH(shift_pos)
            
            toolbox.propka_prot(file_pos,pH)

        else:
            print("Cannot find",pid)
            continue




def protonate_new(pdb_bmr_dict):
    for pdbid in pdb_bmr_dict:
        pid=pdbid[:5]
        print("Processing refDB training structure: "+pid)
        file_pos=PDB_FOLDER+"train/" + pdbid
        # PDB cleaning
        lines_to_remove = {'1TYMB':"235  N   THR B  30", '1VCXA':" D", '1OB1F':"744  N   HIS F  97", 
                           '2J03S':"771  N   GLY S 109", '2UUBP':"701  N   ALA P  84", '2BP3A':"657  N   ASP A1955", 
                           '1VDRB':"1183  N   ALA B 159", '2CHDA':"1020  N   ILE A 510", '2UWDA':"1625  N   LYS A 224", 
                           '1KMVA':" 1471  N   ASP A 186", '2MB5_':" D", 
                           '2BUOA':"647  N   LEU A 231"}
        for key, value in lines_to_remove.items():
            if pdbid[:5]==key:
                filter_file(file_pos,value)
        #print(file_pos)
        if os.path.exists(file_pos):
            shift_pos="shifts/" + pdb_bmr_dict[pdbid]
            pH=toolbox.get_pH(shift_pos)
            #protonation
            toolbox.propka_prot(file_pos,pH)

        else:
            print("Cannot find",pid)
            continue






if __name__=="__main__":
    # Load pickle files to obtain matching between pdbs and shifts
    with open('seq_alignment_dict.pkl', 'rb') as f:
        seq_alignment_dict = pickle.load(f)


    #with open("pdb_bmr_dict.pkl","rb") as f:
    with open("test200.pkl","rb") as f:
        pdb_bmr_dict=pickle.load(f)


    with open("pdb_to_shift_dict-old.pkl","rb") as f:
        pdb_to_shift_dict_old=pickle.load(f)
        #pdb_to_shift_dict_new=pickle.load(f)


    with open("pdb_to_shift_dict-new.pkl","rb") as f:
        pdb_to_shift_dict_new=pickle.load(f)



    #############Parallel execute whole building
    pool=multiprocessing.Pool(processes=WORKER_COUNT)


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
    pool.map(protonate_refdb_test, pdb_bmr_dict_list)
    print("Finishes refDB testing data")
    

    
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
    pool.map(protonate_spartap, seq_alignment_dict_list)
    print("Finishes sparta+ data")
    
    
    
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
    pool.map(protonate_new, pdb_to_shift_dict_new_list)
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
    pool.map(protonate_shiftx2, pdb_to_shift_dict_old_list)
    print("Finishes old shiftx2 data")
    



folder_train = PDB_FOLDER + '/train'
folder_test = PDB_FOLDER + '/test'
folder_list = [folder_train, folder_test]


for folder in folder_list:
    files = os.listdir(folder)
    for file_name in files:
        if file_name.endswith('.log') or file_name.endswith('.remove'):
            file_path = os.path.join(folder, file_name)
            os.remove(file_path)
            print(f"Removed: {file_path}")






