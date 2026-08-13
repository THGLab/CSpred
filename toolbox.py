from Bio.SeqUtils import IUPACData
from Bio import PDB
from urllib.request import urlopen
import pickle
import os
import re
import numpy as np
from numpy import argmin
import subprocess
import sys
import pandas as pd
from io import StringIO
import ssl
import subprocess

protein_dict={pair[0].upper():pair[1] for pair in IUPACData.protein_letters_3to1.items()}
protein_dict_reverse={pair[1]:pair[0].upper() for pair in IUPACData.protein_letters_3to1.items()}


RULER="0         1         2         3         4         5         6         7         8         9        10        11        12"
ATOMS = ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE3', 'CE2', 'CG', 'CG1', 'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3', 'H', 'HA', 'HB', 'HB2', 'HB3', 'HD1', 'HD2', 'HD21', 'HD22', 'HD3', 'HE', 'HE1', 'HE2', 'HE3', 'HE21', 'HE22', 'HG', 'HG1', 'HG12', 'HG13', 'HG2', 'HG3', 'HH2', 'HZ', 'HZ2', 'HZ3', 'N', 'ND2', 'NE1', 'NE2']

AMINOACIDS = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']


mae = lambda x: np.abs(x).mean()
rmse=lambda x: np.sqrt(np.square(x).mean())
try:
    check_result=subprocess.check_output(["which","reduce"])
except:
    check_result=""
REDUCE_STATUS=len(check_result)!=0


def is_heteroatom(residue):
    return residue.id[0].isspace() and residue.id[2] != 'HOH'


class PDBSaver:
    '''
    Self-defined function for writing a PDB file from Biopython chain object
    '''
    def __init__(self):
        self.structure=None

    def set_structure(self,structure):
        self.structure=structure

    def save(self,address):
        if self.structure==None:
            raise TypeError("Structure not set!")
        else:
            contents=[]
            atom_counter=0
            for residue in self.structure.get_residues():
                for atom in residue.get_atoms():
                    if atom.is_disordered():
                        atom=atom.child_dict[sorted(atom.child_dict)[0]]
                    atom_counter+=1
                    if is_heteroatom(residue):
                        contents.append("ATOM %6d  %-4s%3s %s%4d%1s   %8.3f%8.3f%8.3f  1.00%6.2f         %3s\n"%(atom_counter,atom.name,residue.resname,self.structure.id,residue.get_id()[1],residue.get_id()[2],atom.coord[0],atom.coord[1],atom.coord[2],atom.bfactor,atom.element))
                    else:
                        contents.append("HETATM %4d  %-4s%3s %s%4d%1s   %8.3f%8.3f%8.3f  1.00%6.2f         %3s\n"%(atom_counter,atom.name,residue.resname,self.structure.id,residue.get_id()[1],residue.get_id()[2],atom.coord[0],atom.coord[1],atom.coord[2],atom.bfactor,atom.element))
            with open(address,"w") as f:
                f.writelines(contents)



def download_pdb(pdb_id,chain_id=None,destination=None,add_hydrogen=False,residue_whitelist=None):
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        data=urlopen("https://files.rcsb.org/download/%s.pdb"%pdb_id)
    except:
        print("Cannot download PDB %s"%pdb_id)
        return False
    s=[]
    for line in data:
        s.append(line.decode("utf-8"))
    if destination is None:
        destination=os.getcwd()+"/"
    if chain_id=="_":
        filepath=destination+"%s_.pdb"%pdb_id
    else:
        filepath=destination+"%s.pdb"%pdb_id
    with open(filepath,"w") as f:
        f.writelines(s)
    if chain_id not in {None,"_"} or residue_whitelist != None:
        # Need post process after download
        parser=PDB.PDBParser()
        struc=parser.get_structure(pdb_id+chain_id,filepath)
        if chain_id not in {None,"_"}:
            # Get single chain PDB
            if len(struc)>1:
                print("Multiple structures found for %s, only the first structure is taken."%pdb_id)
                struc=struc[0]
            chains=[item for item in struc.get_chains() if item.id==chain_id]
            if len(chains)!=1:
                print("Cannot find chain %s for PDB %s"%(chain_id,pdb_id))
                return False
            else:
                struc=chains[0]
        else:
            struc=struc[0].child_list[0]
        if residue_whitelist:
            # Save only whitelist residues
            deletion=[]
            for residue in struc.child_list:
                if residue.resname not in residue_whitelist:
                    deletion.append(residue.id)
            for delete_id in deletion:
                struc.detach_child(delete_id)

        io=PDBSaver()
        io.set_structure(struc)
        os.remove(filepath)
        filepath=destination+"%s.pdb"%(pdb_id+chain_id)
        io.save(filepath)
    if add_hydrogen:
        assert REDUCE_STATUS, "REDUCE is not correctly configured. Please make sure REDUCE is in your path. \nFor more information, please visit http://kinemage.biochem.duke.edu/software/reduce.php"
        os.system("reduce -build %s > %s -Quiet"%(filepath,filepath+".H"))
        os.rename(filepath+".H",filepath)
    print("PDB %s downloaded to %s"%(pdb_id,filepath))
    return True
    
    
    
    
def propka_prot(filepath,pH=5):

    pH_pattern = r'^[0-9]+(\.[0-9]+)?$'

    #if re.match(pH_pattern, pH):
    #os.system("pdb2pqr30 --nodebump --keep-protons --ff=AMBER --titration-state-method=propka --quiet --with-ph %s --pdb-output %s %s %s"%(pH,filepath+".H",filepath,filepath+".remove"))
    if (pH <= 14) and (pH > 0):
        os.system("pdb2pqr30 --nodebump --ff=AMBER --titration-state-method=propka --quiet --log-level ERROR  --with-ph %s --pdb-output %s %s %s"%(pH,filepath+".H",filepath,filepath+".remove"))
    else:
        print('pH not provided')
        os.system("pdb2pqr30 --nodebump --ff=AMBER --quiet --log-level ERROR --pdb-output %s %s %s"%(filepath+".H",filepath,filepath+".remove"))

    try:
        os.rename(filepath+".H",filepath)
    except:
        print("Protonation failed")

    #else:
    #os.system("pdb2pqr30 --nodebump --keep-protons --noopt --ff=AMBER --titration-state-method=propka --with-ph 7 %s %s"%(filepath,filepath+".H"))
    #os.rename(filepath+".H",filepath)
    print("PDB re-protonated to %s"%(filepath))
    return True
    
    
    
    
    
    
    
    
    
    
    

def fetch_seq(pdb_id,chain_id=None):
    if chain_id=="_":
        chain_id="A"
    try:
        data=urlopen("https://www.rcsb.org/pdb/download/viewFastaFiles.do?structureIdList=%s&compressionType=uncompressed"%pdb_id)
    except:
        print("Cannot find sequence for PDB",pdb_id)
    seq=""
    record_seq=False
    all_chain=chain_id==None
    all_chain_record=dict()
    for line in data:
        line=line.decode("utf-8")
        if len(line.strip())==0:
            # An empty line
            continue
        if "<!DOCTYPE html" in line:
            # Need to find superseded PDB
            main_page=urlopen("https://www.rcsb.org/structure/removed/"+pdb_id)
            content=main_page.read().decode("utf-8")
            supersede=content.split("It has been replaced (superseded) by&nbsp<a href=\"/structure/")[1][:4]
            print("PDB %s has been superseded by %s"%(pdb_id,supersede))
            return fetch_seq(supersede,chain_id)
        if all_chain:
            # No chain specified, return all chains in a dictionary

            if line[0]==">":
                # Start of a new chain
                if len(seq)!=0:
                    all_chain_record[chain_id]=seq
                seq=""
                chain_id=line.split("|")[0].split(":")[1]
                continue
            else:
                seq=seq+line.strip()
        else:
            # Chain specified, only return that chain
            if ">%s:%s"%(pdb_id.upper(),chain_id.upper()) in line:
                # The next line starts the required sequence
                record_seq=True
                continue
            else:
                if line[0]==">":
                    # Start of a chain that we don't need
                    record_seq=False
                    if len(seq)!=0:
                        # Requested chain found and recorded
                        return seq
                    continue
                else:
                    if record_seq:
                        seq=seq+line.strip()
    if all_chain:
        # add the last sequence when no chain is specified
        all_chain_record[chain_id]=seq
        return all_chain_record
    else:
        return seq

def decode_seq(seq,supplementary_dict=None):
    lookup_dict=protein_dict_reverse
    if supplementary_dict is not None:
        for item in supplementary_dict:
            lookup_dict[item]=supplementary_dict[item]
    seq=seq.upper()
    if len(seq)==1:
        return lookup_dict[seq]
    return [lookup_dict.get(r,"UNK") for r in seq]

def form_seq(arr,supplementary_dict=None):
    lookup_dict=protein_dict
    if supplementary_dict is not None:
        for item in supplementary_dict:
            lookup_dict[item]=supplementary_dict[item]
    return "".join([lookup_dict[r.upper()] for r in arr])

def load_pkl(path):
    with open(path,"rb") as f:
        obj=pickle.load(f)
    return obj

def dump_pkl(obj,path):
    with open(path,"wb") as f:
        pickle.dump(obj,f)
    print("Saved",os.getcwd()+"/"+path)

def get_pH(shift_file_path,default=5):
    '''
    Get pH value description from a shift file
    '''
    with open(shift_file_path) as f:
        data=f.read()
    regex=re.compile("pH.*\d+.*")
    pH_line=regex.search(data)
    if pH_line is None:
        return default
    else:
        pH_line=pH_line.group()
        digit_re=re.compile("\d+\.\d+")
        candidate_numbers=digit_re.findall(pH_line)
        int_re=re.compile(" \d{1,2} ")
        candidate_numbers.extend(int_re.findall(pH_line))
        if len(candidate_numbers)==0:
            return default
        # We assume that the pH value is the closest number to the word "pH"
        distances=[abs(pH_line.index(num)-pH_line.index("pH")) for num in candidate_numbers]
        return eval(candidate_numbers[argmin(distances)])

def get_res(file):
    with open(file) as f:
        data=f.read()
    regex=re.compile("RESOLUTION.*\d+\.\d+.*ANGSTROMS.")
    resolution_line=regex.search(data)
    if resolution_line is None:
        return None
    else:
        resolution_line=resolution_line.group()
    digit_re=re.compile("\d+\.\d+")
    resolution=digit_re.search(resolution_line)
    if resolution is None:
        return None
    else:
        resolution=eval(resolution.group())
    return resolution
    
def get_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(StringIO(gpu_stats.decode("utf-8").replace("MiB","")),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    gpu_df["usage"]=gpu_df["memory.free"]/(gpu_df["memory.used"]+gpu_df["memory.free"])
    idx = gpu_df['usage'].idxmax()
    if gpu_df.loc[idx,"usage"]<0.1:
        idx=None
    return idx
