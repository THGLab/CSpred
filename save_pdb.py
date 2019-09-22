# Self-defined function for writing a PDB file from Biopython chain object (that solves problems associated with disordered atoms)

# Author: Jie Li
# Date created: Sep 22, 2019

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
                    contents.append("ATOM %6d%5s %3s %s %3d     %7.3f %7.3f %7.3f  1.00 %5.2f         %3s\n"%(atom_counter,atom.name,residue.resname,self.structure.id,residue.get_id()[1],atom.coord[0],atom.coord[1],atom.coord[2],atom.bfactor,atom.element))
            with open(address,"w") as f:
                f.writelines(contents)