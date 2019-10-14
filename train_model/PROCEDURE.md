# Procedure for downloading data and training model

## 1 Program requirements
* Python
* Biopython
* PyNMRStar
* [REDUCE](http://kinemage.biochem.duke.edu/software/reduce.php)

## 2 Download PDB files with hydrogen added
  Execute [`download_pdbs.py`]().<br>
  You can specify the number of cores to use by changing the "WORKER" parameter in the script.

## 3 Generate data files