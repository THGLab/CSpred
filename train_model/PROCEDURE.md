# Procedure for downloading data and training UCBShift

## 1 Program requirements
* Python
* Biopython
* PyNMRStar
* [REDUCE](http://kinemage.biochem.duke.edu/software/reduce.php)
* [PDB2PQR](https://pdb2pqr.readthedocs.io/en/latest/getting.html)

## 2 Download PDB files and hydrogenate
  Make sure you are in the [`train_model`]() folder.<br>
  Execute [`download_pdbs.py`]() to download all the single chain PDB files and add hydrogens for training and testing.<br>
  You can choose whether or not include crystal waters and ligands in the structure, by setting the "RESIDUE_WHITELIST" keyword in [`download_pdbs.py`](). If you want all crystal waters and the protein, set whitelist as all amino acids and HOH. If you only want the protein, set whitelist as all amino acids. If you want anything contained in the RCSB structure, set whitelist to None.<br>
  You can specify the number of cores to use by changing the "WORKER" parameter in the script.

## 3 PDB2PQR protonation
Execute `protonation.py`

## 4 Generate data files for training UCBShift-X
  After you have downloaded all the PDB files, execute [`build_df.py`]() to generate .csv feature data files for training and testing.

## 5 Create UCBShift-Y predictions for training the network
  At the same time, you can execute [`make_Y_preds.py`]() to generate UCBShift-Y predictions for each training example that is used to train the combining model of UCBShift.

## 6 Train the models
  Train UCBShift-X models for all six atom types by running [`train.py`]().

## 7 Evaluate performance of the model on the test set
  After the training finishes, you can run [`evaluate.py`]() to make predictions on all test proteins and analyze the results. For each model (UCBShift-X, UCBShift-Y and UCBShift), two analysis files will be generated. One errors.csv file recording the RMSE, minimum and maximum error for each protein and all the atom types, the other preds.csv file contains all the prediction outputs and target values.
