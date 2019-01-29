#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 12:11:25 2017

@author: kcbennett
"""



import pandas as pd
import math
from Bio import PDB
from Bio.PDB.PDBParser import PDBParser
from Bio.SubsMat.MatrixInfo import blosum62
from Bio.SeqUtils import IUPACData

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


class BaseDataReader(object):
    """ Base class for reading data files

    The main function of any child of this class is to read a text file
    in some relevant format and return a pandas DataFrame containing the
    data in that file

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
        """ Fix improperly formatted atom types

        This means moving any number at the start of thpandase string to
        the end and changing Deuteriums to Hydrogens. E.g. if the
        input is '1DG2', this becomes 'HG12'.

        Args:
            start_atom_type (str): inital atom type

        Returns:
            the fixed atom type (str)

        """
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
        """ Fix wrong residue names

        This is just removing extraneous capital letters at
        the front of some residue names
        Args:
            start_res_name (str): intital residue name

        Returns: 
            the fixed residue name (str)
        """
        res_name = start_res_name
        if len(start_res_name) > 3:
            res_name = start_res_name[1:]
        return res_name


    def blosum_nums(self, res_obj):
        '''Get blosum62 block substitution matrix comparison numbers
        
        Args:
            res_obj (Bio.PDB.Residue): the residue that is being compared
        
        Returns:
            The 20 blosum62 comparison numbers (list of floats) or [0]*20 if residue not found.
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
        '''Get a binary vector indicating which of the 20 standard amino acids the residue is.
        
        Args:
            res_obj (Bio.PDB.Residue): the residue that is being classified 
         
        Returns:
            A list of 20 integers all but one of which are zeros and the remaining is a one.  
            The position of the 1 element indicates which of the 20 standard amino acids the
            residue is.  If the residue is not one of the standard 20, returns [0]*20
        '''
        
        res_name = self._fix_res_name(res_obj.resname)
        out = 0*[20]
        for i, aa in enumerate(AAlist):
            if res_name == aa:
                out[i] = 1
        return out
        
    
    @staticmethod
    def calc_phi_psi(chain_obj):
        '''Get list of dihedral phi and psi angles
        
        Args:
            chain_obj (Bio.PDB.Chain): a chain of residues for which to
            calculate the dihedral angles
            
        Returns:
            A list of angle parameters[cos(phi), sin(phi), cos(psi), sin(psi)] 
            with [0]*4 for the absence of an angle
        '''
        
        polys = PDB.PPBuilder().build_peptides(chain_obj)
        phi_psi_list=[]
        for poly in polys:
            phi_psi_list.append(poly.get_phi_psi_list())
        
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
        return out_list
                
    
    def calc_torsion_angles(self, res_obj, chi=[1,2]):
        '''Get list of chi angles
        
        Args:
            res_obj (Bio.PDB.Residue): a residue for which to calculate 
            torsion angles
            
            chi (list of ints): a list of integers in range(1,6) that specify which chi
            angles to calculate.  Default is [1, 2]
            
        Returns:
            The sum of lists of torsion angle parameters [cos(chi), sin(chi), 0/1],
            with the boolean indicating existence of the angle, for each of the chi
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
    #            alt_chi = "altchi%s" % x
    #            if alt_chi in chi_defs.keys():
    #                chi_names.append(alt_chi)
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
    def s2_param(nn_tree, res_obj, prev_Oatom, rad=10.0, b_param=-0.1):
        
        '''Calculate S2 order parameter of N-H bond based on the contact model 
        put forth in Zhang, Bruschweiler (2002) J. Am. Chem. Soc 124
        
        Args:
            nn_tree (Bio.PDB.NeighborSearch): a tree containing the 
            distances between atoms in the set of interest
            
            res_obj (Bio.PDB.Residue): the residue for which we are 
            are calculating S2
            
            prev_Oatom (Bio.PDB.Atom): the Oxygen atom for the previous
            residue in the chain
            
            rad (float): the radius within which to search for heavy atoms
            to include in the contact model (note that, in principle, we 
            could use different radii for the rOsum and rHsum in the model)
            
            b_param (float): a parameter in the model
            
        Returns: 
            S2 (float): The S2 order parameter for the N-H bond of res_obj
        '''
      
        rOsum = 0
        rHsum = 0
        
        if prev_Oatom != None:
            nn_atoms = nn_tree.search(prev_Oatom.get_coord(), rad, level='A')
            prev_res = prev_Oatom.get_parent()
            for atom in nn_atoms:
                if 'H' not in atom.get_name() and (atom - prev_Oatom) > 0.1:
                    at_res = atom.get_parent()
                    if (at_res != res_obj) and (at_res != prev_res):
                        rOsum += math.exp(-(atom - prev_Oatom))                                                                                               
                else:
                    pass
        else:
            pass
        
        for at in ['H', '1H']:
            try:
                Hatom = res_obj[at]
                nn_atoms = nn_tree.search(Hatom.get_coord(), rad, level='A')
                for atom in nn_atoms:
                    if 'H' not in atom.get_name() and (atom - Hatom) > 0.1:
                        at_res = atom.get_parent()
                        if (at_res != res_obj) and (at_res != prev_res):
                            rHsum += math.exp(-(atom - Hatom))
                    else:
                        pass
                break
            except KeyError:
                pass
                    
        S2 = math.tanh(0.8 * rOsum + 0.8 * rHsum) + b_param
        return S2
    
    
    @staticmethod
    def find_nearest_atom(nn_tree, catom, rad, atom_type='Any', excl=[]):
        
        '''Find the nearest atom of type atom_type within 
        radius rad of a central atom catom. Note that the
        function finds the closest atom that contains the atom_type string 
        in its name rather than relying on an exact match.
        
        Args:
            nn_tree (Bio.PDB.NeighborSearch): a tree containing the 
            distances between atoms in the set of interest
            
            catom (Bio.PDB.Atom): Central atom for which we wish to find the 
            nearest neighbor
            
            rad (float): The distance within which to search
            
            atom_type (str): The type of atom for which to search. Default
            is to search for any atom.
            
            excl (List of Bio.PDB.Atom): List of atom objects to exclude from the search
            
        Returns:
            nn_atom (Bio.PDB.Atom): The nearest atom_type atom within rad of
            catom.  Returns None if no such atom found within rad of catom
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
                if dist < d_min and dist > 0.1:
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
        else:
            for idx, dist in enumerate(dists):
                atom_name = atoms[idx].name
                if dist < d_min and dist > 0.1 and (atom_type in atom_name):
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
        
        '''Find the next nearest atom of type atom_type within 
        radius rad of a central atom catom. Note that the
        function finds the closest atom that contains the atom_type string 
        in its name rather than relying on an exact match.
        
        Args:
            nn_tree (Bio.PDB.NeighborSearch): a tree containing the 
            distances between atoms in the set of interest
            
            catom (Bio.PDB.Atom): Central atom for which we wish to find the 
            next nearest neighbor
            
            rad (float): The distance within which to search
            
            atom_type (str): The type of atom for which to search. Default
            is to search for any atom.
            
            excl (List of Bio.PDB.Atom): List of atom objects to exclude from the search
            
        Returns:
            nxtnn_atom (Bio.PDB.Atom): The next nearest atom_type atom within 
            rad of catom.  Returns None if no such atom found within rad of catom
        '''
        
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
    
        
    def hbond_network(self, nn_tree, res_obj, rad=3.0):
        
        '''Constructs parameters that encode the structure of 3 different Hydrogen 
        bonds that may occur on a given residue. Each possible hydrogen bond is 
        described by 4 parameters: The first is the distance from donor hydrogen 
        to acceptor atom dHA. The second and third numbers are the cosines of the 
        bond angle at the acceptor atom and the donor hydrogen respecctively.  
        The final number is a boolean for the existence of the H-bond. The 
        parameters are returned as a single list in the following order:
        alpha Hydrogen, Nitrogen Hydrogen, Oxygen.
        
        Args:
            nn_tree (Bio.PDB.NeighborSearch): a tree containing the 
            distances between atoms in the set of interest
            
            res_obj (Bio.PDB.Residue): Residue for which we wish to find the 
            Hydrogen bond parameters
            
            rad (float): The distance within which to search
            
        Returns:
            A (list of floats) sum of lists [dHA, Cos(phi), Cos(psi), 1] containing the bond 
            parameters or [0, 0, 0, 0] for each of 3 possible Hydrogen bonds.
            The order of parameters is alpha Hydrogen, Nitrogen Hydrogen,
            Oxygen.
        '''

        # First initialize all bonds as nonexistent
        HAbond_Params = [0]*4
        HNbond_Params = [0]*4
        Obond_Params = [0]*4

        # Define atoms
        for at in ['HA', '1HA']:
            try:
                HAatom = res_obj[at]
                break
            except KeyError:
                HAatom = None
        for at in ['H', '1H']:
            try:
                HNatom = res_obj[at]
                break
            except KeyError:
                HNatom = None
        try:
            Oatom = res_obj['O'] 
        except KeyError:
            Oatom = None
        
        res_i_atoms = res_obj.get_atoms()
        # Use the above-defined functions to find the nearest Hydrogen/Oxygen to each atom
        # We only search within rad so if we get None, there is no H-bond.
        if HAatom is None:
            HAOatom = None
        else:
            HAOatom = self.find_nearest_atom(nn_tree, HAatom, rad, atom_type = 'O', excl=res_i_atoms)
            
        if HNatom is None:
            HNOatom = None
        else:
            HNOatom = self.find_nearest_atom(nn_tree, HNatom, rad, atom_type = 'O', excl=res_i_atoms)

        if Oatom is None:
            OHatom = None
        else:
            OHatom = self.find_nearest_atom(nn_tree, Oatom, rad, atom_type = 'H', excl=res_i_atoms)

        # Bond distances can now be obtained directly.  Angles will require the atoms to 
        # which the bonding hydrogens are themselves covalently bonded.  For HA and HN, 
        # we can just find the nearest Oxygens, Nitrogens, and Carbons.  We can restrict 
        # the search to rad still, on the assumption that covalent bonds are shorter than hydrogen bonds.
        if HAOatom is None:
            pass
        else:
            ACatom = res_obj['CA']
            HAbond_dist = HAatom - HAOatom
            HAangle2 = PDB.calc_angle(ACatom.get_vector(), HAatom.get_vector(), HAOatom.get_vector())
            try:
                HAOCatom = HAOatom.get_parent()['C']
                HAangle1 = PDB.calc_angle(HAatom.get_vector(), HAOatom.get_vector(), HAOCatom.get_vector())
            except KeyError:
                HAangle1 = math.pi
            HAbond_Params = [HAbond_dist, math.cos(HAangle1), math.cos(HAangle2), 1]
    
        if HNOatom is None:
            pass
        else:
            HNbond_dist = HNatom - HNOatom
            Natom = res_obj['N']
            HNangle2 = PDB.calc_angle(Natom.get_vector(), HNatom.get_vector(), HNOatom.get_vector())
            try:
                HNOCatom = HNOatom.get_parent()['C']
                HNangle1 = PDB.calc_angle(HNatom.get_vector(), HNOatom.get_vector(), HNOCatom.get_vector())
            except KeyError:
                HNangle1 = math.pi
            HNbond_Params = [HNbond_dist, math.cos(HNangle1), math.cos(HNangle2), 1]

        if OHatom is None:
            pass
        else:
            Obond_dist = Oatom - OHatom
            Catom = res_obj['C']
            try:
                OHOatom = OHatom.get_parent()['O']
                OHangle1 = PDB.calc_angle(Catom.get_vector(), Oatom.get_vector(), OHatom.get_vector())
                OHangle2 = PDB.calc_angle(OHOatom.get_vector(), OHatom.get_vector(), Oatom.get_vector())
                Obond_Params = [Obond_dist, math.cos(OHangle1), math.cos(OHangle2), 1]
            except KeyError:
                pass
        
        output = HAbond_Params + HNbond_Params + Obond_Params
        return output


    def df_from_file_1res(self, fpath, hbrad=3.0, s2rad=10.0, hse=False):
        
        '''Function to create a pandas DataFrame of single-residue SPARTA+ features 
        from a given PDB file.
        
        Args:
            fpath (str): path to PDB file
            hbrad (float): max length of hydrogen bonds
            s2rad (float): distance within which to include heavy atoms for modeling S2 parameter
            
        Returns:
            A dataframe (pd.DataFrame) with number of rows equal to number of standard
            amino acids in the PDB and columns given by the SPARTA+ features as well 
            as some preliminary file and residue ID info
            '''
        
        parser = PDBParser(PERMISSIVE=1)
        structure = parser.get_structure('structure', fpath)
        file_id = fpath.split('/')[-1].split('.')[0].split('_')[0]
        file_name = fpath.split('/')[-1]
        
        col_names = ['FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM']
        col_names +=[i+j for i in ['PHI_', 'PSI_'] for j in ['COS', 'SIN']]
        col_names += [i+j for i in ['CHI1_', 'CHI2_'] for j in ['COS', 'SIN', 'EXISTS']]
        col_names += [i+j for i in ['Ha_', 'HN_', 'O_'] for j in ['d_HA', 'COS_H', 'COS_A', 'EXISTS']]
        col_names += ['S2']
        AAlet3 = [i.upper() for i in sorted(IUPACData.protein_letters_3to1.keys())]
        col_names += ['BLOSUM62_NUM_'+AAlet3[i] for i in range(20)]
        col_names += ['HSE_CA' + i  for i in ['_U', '_D', '_Angle']]
        col_names += ['HSE_CB' + i for i in ['_U', '_D']]

        
        data = []
        for model in structure:
            nn_tree = PDB.NeighborSearch(list(model.get_atoms()))
            if hse:
                hseca_calc = PDB.HSExposureCA(model)
                hsecb_calc = PDB.HSExposureCB(model)
            
            for chain in model:
                dihedrals = self.calc_phi_psi(chain)
                l = 0
                prevOatom = None
                for res in chain:
                    resname = self._fix_res_name(res.resname)
                    
                    if resname not in AAlet3:
                        continue
                    res_id = res.get_id()
                    resnum = res_id[1]
                    row_data = [file_id, file_name, resname, resnum]
                    row_data += dihedrals[l]
                    torsions = self.calc_torsion_angles(res)
                    row_data += torsions
                    HBonds = self.hbond_network(nn_tree, res, hbrad)
                    row_data += HBonds
                    row_data += [self.s2_param(nn_tree, res, prevOatom, s2rad)]
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
                        prevOatom = res['O']
                    except KeyError:
                        pass
                    
                    data.append(row_data)
                    l +=1
        df = pd.DataFrame(data, columns=col_names)
        return df
                   

    def df_from_file_3res(self, fpath, hbrad=3.0, s2rad=10.0, rcshifts=True, hse=False):
        
        '''Function to create a pandas DataFrame of SPARTA+ features 
        from a given PDB file.
        
        Args:
            fpath (str): path to PDB file
            hbrad (float): max length of hydrogen bonds
            s2rad (float): distance within which to include heavy atoms for modeling S2 parameter
            
        Returns:
            A dataframe (pd.DataFrame) with number of rows equal to number of standard 
            amino acids in the PDB and columns given by the SPARTA+ features as well 
            as some preliminary file and residue ID info
        '''
            
        #Names of columns from single-residue DataFrame for easier access
        phipsi_names = [i+j for i in ['PHI_', 'PSI_'] for j in ['COS', 'SIN']]
        chi_names = [i+j for i in ['CHI1_', 'CHI2_'] for j in ['COS', 'SIN', 'EXISTS']]
        hbprev_names = ['O_'+j for j in ['d_HA', 'COS_H', 'COS_A', 'EXISTS']]
        hb_names = [i+j for i in ['Ha_', 'HN_', 'O_'] for j in ['d_HA', 'COS_H', 'COS_A', 'EXISTS']]
        hbnext_names = ['HN_'+j for j in ['d_HA', 'COS_H', 'COS_A', 'EXISTS']]
        hse_names = ['HSE_CA' + i  for i in ['_U', '_D', '_Angle']]
        hse_names += ['HSE_CB' + i for i in ['_U', '_D']]

        # Define column names for new DataFrame
        col_id = ['FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM']
        col_phipsi = [i+j for i in ['PHI_', 'PSI_'] for j in ['COS_i-1', 'SIN_i-1']]
        col_phipsi += [i+j for i in ['PHI_', 'PSI_'] for j in ['COS_i', 'SIN_i']]
        col_phipsi += [i+j for i in ['PHI_', 'PSI_'] for j in ['COS_i+1', 'SIN_i+1']]
        col_chi = [i+j+k for k in ['_i-1', '_i', '_i+1'] for i in ['CHI1_', 'CHI2_'] for j in ['COS', 'SIN', 'EXISTS']]
        col_hbprev = ['O_'+i+'_i-1' for i in ['d_HA', '_COS_H', '_COS_A', '_EXISTS']]
        col_hbond = [i+j+'_i' for i in ['Ha_', 'HN_', 'O_'] for j in ['d_HA', '_COS_H', '_COS_A', '_EXISTS']]
        col_hbnext = ['HN_'+i+'_i+1' for i in ['d_HA', '_COS_H', '_COS_A', '_EXISTS']]
        col_s2 = ['S2'+i for i in ['_i-1', '_i', '_i+1']]
        blosum_names = ['BLOSUM62_NUM_'+list(IUPACData.protein_letters_3to1.keys())[i].upper() for i in range(20)]
        col_blosum = [blosum_names[i]+j
        for j in ['_i-1', '_i', '_i+1'] for i in range(20)]
        col_names = col_id + col_phipsi + col_chi + col_hbprev + col_hbond + col_hbnext + col_s2 + col_blosum
        if rcshifts:
            col_rccs = ['RC_' + i for i in atom_names]
            col_names += col_rccs
        if hse:
            col_hse = [hse_names[i] + j for j in ['_i-1', '_i', '_i+1'] for i in range(5)]
            col_names += col_hse
        
        df = pd.DataFrame(columns=col_names)
        # Get SPARTA+ features with single-residue function 
        df_1res = self.df_from_file_1res(fpath, hbrad, s2rad, hse=hse)
        
        for i in range(len(df_1res)):
            # Assign ID columns
            df.loc[i, col_id] = df_1res.loc[i, col_id].values
            
            # Assign column variables containing data from previous residue in the PDB file
            if i == 0:        
                blosum_prev = [0]*20
                psi_prev = [0, 0]
                phi_prev = [0, 0]
                chi_prev = [0]*6
                hb_prev = [0]*4
                s2_prev = 0
                hse_prev = 5*[0]
            else:
                blosum_prev = list(df_1res.loc[i-1, blosum_names].values)
                psi_prev = [df_1res.loc[i-1, 'PSI_COS'], df_1res.loc[i-1, 'PSI_SIN']]
                phi_prev = [df_1res.loc[i-1, 'PHI_COS'], df_1res.loc[i-1, 'PHI_SIN']]
                chi_prev = list(df_1res.loc[i-1, chi_names].values)
                hb_prev = list(df_1res.loc[i-1, hbprev_names].values)
                s2_prev = df_1res.loc[i-1, 'S2']
                if hse:
                    hse_prev = df_1res.loc[i-1, hse_names]
            
            # Assign column variables containing data from next residue in the PDB file
            if i == len(df_1res)-1:
                blosum_next = [0]*20
                psi_next = [0, 0]
                phi_next = [0, 0]
                chi_next = [0]*6
                hb_next = [0]*4
                s2_next = 0
                hse_next = 5*[0]
                if rcshifts:
                    res_next = 'ALA'
            else:
                blosum_next = list(df_1res.loc[i+1, blosum_names].values)
                psi_next = [df_1res.loc[i+1, 'PSI_COS'], df_1res.loc[i+1, 'PSI_SIN']]
                phi_next = [df_1res.loc[i+1, 'PHI_COS'], df_1res.loc[i+1, 'PHI_SIN']]
                chi_next = list(df_1res.loc[i+1, chi_names].values)
                hb_next = list(df_1res.loc[i+1, hbnext_names].values)
                s2_next = df_1res.loc[i+1, 'S2']
                if hse:
                    hse_next = df_1res.loc[i+1, hse_names]
                if rcshifts:
                    res_next = df_1res.loc[i+1, 'RESNAME']
                    
            
            # Insert row into DataFrame
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
                    rccs = [randcoil_pro[i][resname] for i in atom_names]
                else:
                    rccs = [randcoil_ala[i][resname] for i in atom_names]
                df.loc[i, col_rccs] = rccs
            if hse:
                df.loc[i, col_hse] = hse_prev + list(df_1res.loc[i, hse_names].values) + hse_next
            
        return df
    
    
    def df_from_file_tripeptide(self, fpath, hbrad=2.5, s2rad=10, rcshifts=True):
 
        '''Function to create a pandas DataFrame of full tripeptide SPARTA+ features 
        from a given PDB file.
        
        Args:
            fpath (str): path to PDB file
            hbrad (float): max length of hydrogen bonds
            s2rad (float): distance within which to include heavy atoms for modeling S2 parameter
            
        Returns:
            A dataframe (pd.DataFrame) with number of rows equal to number of standard 
            amino acids in the PDB and columns given by the SPARTA+ features as well 
            as some preliminary file and residue ID info
        '''
            
        # Get SPARTA+ features with single-residue function and then define column names
        df_1res = self.df_from_file_1res(fpath, hbrad, s2rad)
        col_id = ['FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM']
        col_phipsi = [i+j+k for k in ['i-1', 'i', 'i+1'] for i in ['PHI_', 'PSI_'] for j in ['COS_', 'SIN_']]
        col_chi = [i+j+k for k in ['_i-1', '_i', '_i+1'] for i in ['CHI1_', 'CHI2_'] for j in ['COS', 'SIN', 'EXISTS']]
        col_hbond = [i+j+k for k in ['_i-1', '_i', '_i+1'] for i in ['Ha_', 'HN_', 'O_']
                     for j in ['d_HA', '_COS_H', '_COS_A', '_EXISTS']]
        col_s2 = ['S2'+i for i in ['_i-1', '_i', '_i+1']]
        blosum_names = ['BLOSUM62_NUM_'+list(IUPACData.protein_letters_3to1.keys())[i].upper() for i in range(20)]
        col_blosum = [blosum_names[i]+j for j in ['_i-1', '_i', '_i+1'] for i in range(20)]
        col_names = col_id + col_phipsi + col_chi + col_hbond + col_s2 + col_blosum
        if rcshifts:
            col_rccs = ['RC_' + i for i in atom_names]
            col_names += col_rccs

        
        df = pd.DataFrame(columns=col_names)
        
        #Names of columns from single-residue DataFrame for easier access
        phipsi_names = [i+j for i in ['PHI_', 'PSI_'] for j in ['COS', 'SIN']]
        chi_names = [i+j for i in ['CHI1_', 'CHI2_'] for j in ['COS', 'SIN', 'EXISTS']]
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
                chi_prev = [0]*6
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
                chi_next = [0]*6
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