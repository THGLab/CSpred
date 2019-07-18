import numpy as np
import os
import sys
sys.path.append("/data/jerry/NMR")
import toolbox
import json
free_gpu=toolbox.get_free_gpu()
if free_gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(free_gpu)
    print("Selecting GPU",free_gpu)
else:
    print("There is no free GPU available!")
    exit()
import csv

DEBUG=False
if DEBUG:
    os.chdir("/home/jerry/data/NMR/chemshift_prediction")
    config_file="model_configs/debug.json"
else:
    config_file=sys.argv[1]
with open(config_file) as f:
    configs=json.load(f)
os.environ["KERAS_BACKEND"]=configs.get("keras_backend","cntk")
from CSpredictor_networks import *
from data_prep_functions import *
from Bio.SeqUtils import IUPACData
from keras.constraints import max_norm

# The dictionary storing the hydrophobicity for different residues
# Literature: Wimley WC & White SH (1996). Nature Struct. Biol. 3:842-848. 
hydrophobic_dict={'LYS': 1.81, 'GLN': 0.19, 'THR': 0.11, 'ASP': 0.5, 'GLU': 0.12, 'ARG': 1.0, 'LEU': -0.69, 'TRP': -0.24, 'VAL': -0.53, 
'ILE': -0.81, 'PRO': -0.31, 'MET': -0.44, 'ASN': 0.43, 'SER': 0.33, 'ALA': 0.33, 'GLY': 1.14, 'TYR': 0.23, 'HIS': -0.06, 'PHE': -0.58, 'CYS': 0.22}

all_atoms=['H','HA','C','CA','CB','N']
protein_letters=[code.upper() for code in IUPACData.protein_letters_3to1.keys()]
sp_feat_cols=['BLOSUM62_NUM_ALA_i-1', 'BLOSUM62_NUM_CYS_i-1', 'BLOSUM62_NUM_ASP_i-1', 'BLOSUM62_NUM_GLU_i-1', 'BLOSUM62_NUM_PHE_i-1', 
'BLOSUM62_NUM_GLY_i-1', 'BLOSUM62_NUM_HIS_i-1', 'BLOSUM62_NUM_ILE_i-1', 'BLOSUM62_NUM_LYS_i-1', 'BLOSUM62_NUM_LEU_i-1', 'BLOSUM62_NUM_MET_i-1', 
'BLOSUM62_NUM_ASN_i-1', 'BLOSUM62_NUM_PRO_i-1', 'BLOSUM62_NUM_GLN_i-1', 'BLOSUM62_NUM_ARG_i-1', 'BLOSUM62_NUM_SER_i-1', 'BLOSUM62_NUM_THR_i-1', 
'BLOSUM62_NUM_VAL_i-1', 'BLOSUM62_NUM_TRP_i-1', 'BLOSUM62_NUM_TYR_i-1', 'PHI_SIN_i-1', 'PHI_COS_i-1', 'PSI_SIN_i-1', 'PSI_COS_i-1', 'CHI1_SIN_i-1',
 'CHI1_COS_i-1', 'CHI1_EXISTS_i-1', 'CHI2_SIN_i-1', 'CHI2_COS_i-1', 'CHI2_EXISTS_i-1', 'BLOSUM62_NUM_ALA_i', 'BLOSUM62_NUM_CYS_i', 'BLOSUM62_NUM_ASP_i',
 'BLOSUM62_NUM_GLU_i', 'BLOSUM62_NUM_PHE_i', 'BLOSUM62_NUM_GLY_i', 'BLOSUM62_NUM_HIS_i', 'BLOSUM62_NUM_ILE_i', 'BLOSUM62_NUM_LYS_i', 'BLOSUM62_NUM_LEU_i',
 'BLOSUM62_NUM_MET_i', 'BLOSUM62_NUM_ASN_i', 'BLOSUM62_NUM_PRO_i', 'BLOSUM62_NUM_GLN_i', 'BLOSUM62_NUM_ARG_i', 'BLOSUM62_NUM_SER_i', 'BLOSUM62_NUM_THR_i',
 'BLOSUM62_NUM_VAL_i', 'BLOSUM62_NUM_TRP_i', 'BLOSUM62_NUM_TYR_i', 'PHI_SIN_i', 'PHI_COS_i', 'PSI_SIN_i', 'PSI_COS_i', 'CHI1_SIN_i', 'CHI1_COS_i', 
 'CHI1_EXISTS_i', 'CHI2_SIN_i', 'CHI2_COS_i', 'CHI2_EXISTS_i', 'BLOSUM62_NUM_ALA_i+1', 'BLOSUM62_NUM_CYS_i+1', 'BLOSUM62_NUM_ASP_i+1', 
 'BLOSUM62_NUM_GLU_i+1', 'BLOSUM62_NUM_PHE_i+1', 'BLOSUM62_NUM_GLY_i+1', 'BLOSUM62_NUM_HIS_i+1', 'BLOSUM62_NUM_ILE_i+1', 'BLOSUM62_NUM_LYS_i+1', 
 'BLOSUM62_NUM_LEU_i+1', 'BLOSUM62_NUM_MET_i+1', 'BLOSUM62_NUM_ASN_i+1', 'BLOSUM62_NUM_PRO_i+1', 'BLOSUM62_NUM_GLN_i+1', 'BLOSUM62_NUM_ARG_i+1',
'BLOSUM62_NUM_SER_i+1', 'BLOSUM62_NUM_THR_i+1', 'BLOSUM62_NUM_VAL_i+1', 'BLOSUM62_NUM_TRP_i+1', 'BLOSUM62_NUM_TYR_i+1', 'PHI_SIN_i+1', 'PHI_COS_i+1',
'PSI_SIN_i+1', 'PSI_COS_i+1', 'CHI1_SIN_i+1', 'CHI1_COS_i+1', 'CHI1_EXISTS_i+1', 'CHI2_SIN_i+1', 'CHI2_COS_i+1', 'CHI2_EXISTS_i+1', 'O__EXISTS_i-1',
'O_d_HA_i-1', 'O__COS_A_i-1', 'O__COS_H_i-1', 'HN__EXISTS_i', 'HN_d_HA_i', 'HN__COS_A_i', 'HN__COS_H_i', 'Ha__EXISTS_i', 'Ha_d_HA_i', 'Ha__COS_A_i', 
'Ha__COS_H_i', 'O__EXISTS_i', 'O_d_HA_i', 'O__COS_A_i', 'O__COS_H_i', 'HN__EXISTS_i+1', 'HN_d_HA_i+1', 'HN__COS_A_i+1', 'HN__COS_H_i+1', 'S2_i-1', 'S2_i',
 'S2_i+1']
spartap_cols=sp_feat_cols+all_atoms+[a+"_RC" for a in all_atoms]
col_square=["%s_%s_%s"%(a,b,c) for a in ['PHI','PSI'] for b in ['COS','SIN'] for c in ['i-1','i','i+1']]  
dropped_cols=["DSSP_%s_%s"%(a,b) for a in ["PHI","PSI"] for b in ['i-1','i','i+1']]+["BMRB_RES_NUM","MATCHED_BMRB","CG","HA2_RING","HA3_RING","RCI_S2"]
col_lift=[col for col in sp_feat_cols if "BLOSUM" not in col and "_i-1" not in col and "_i+1" not in col]
non_numerical_cols=['3_10_HELIX_SS_i',"A_HELIX_SS_i","BEND_SS_i","B_BRIDGE_SS_i","CHI1_EXISTS_i","CHI2_EXISTS_i","HN__EXISTS_i","Ha__EXISTS_i","NONE_SS_i","O__EXISTS_i","PI_HELIX_SS_i","STRAND_SS_i","TURN_SS_i"]+protein_letters

def Add_res_spec_feats(dataset,include_onehot=True):
    '''
    Adding residue specific features into the dataset (only for current residue), including one-hot representation of the residue,
    and the hydrophobicity of the residue
    '''
    if include_onehot:
        for code in protein_letters:
            dataset[code]=[int(res==code) for res in dataset['RESNAME']]
    dataset["HYDROPHOBICITY"]=[hydrophobic_dict[res] for res in dataset['RESNAME']]


# Feature space lifting
def Lift_Space(dataset,participating_cols,increased_dim,w,b):
    if w is None:
        w = np.random.normal(0, 0.1, (len(participating_cols), increased_dim))
        b = np.random.uniform(0, 2*np.pi, increased_dim)
    lifted_dat_mat=np.cos(dataset[participating_cols].values.dot(w)+b)
    for n in range(increased_dim):
        dataset["Lifted_%d"%n]=lifted_dat_mat[:,n]


# read all files
data_path_train= "datasets/train"
data_path_val= "datasets/validation"
data_path_test="datasets/test"
train_files=[]
val_files=[]
test_files=[]
for file in os.listdir(data_path_train):
    if file.split(".")[-1]=="csv":
        train_files.append(pd.read_csv(data_path_train+"/"+file))
for file in os.listdir(data_path_val):
    if file.split(".")[-1]=="csv":
        val_files.append(pd.read_csv(data_path_val+"/"+file))
for file in os.listdir(data_path_test):
    if file.split(".")[-1]=="csv":
        test_files.append(pd.read_csv(data_path_test+"/"+file))
df_train = pd.concat(train_files,ignore_index=True)
df_val=pd.concat(val_files,ignore_index=True)
df_test=pd.concat(test_files,ignore_index=True)
# Renames columns in spartap data to be same as above columns which are taken from our data construction routines
# Fix columns order to prevent training dataset and testing dataset having different feature orders

df_train = df_train.rename(index=str, columns=sparta_rename_map) 
df_train = df_train[sorted(df_train.columns)]
df_val = df_val.rename(index=str, columns=sparta_rename_map)
df_val = df_val[sorted(df_val.columns)]
df_test = df_test.rename(index=str, columns=sparta_rename_map)
df_test = df_test[sorted(df_test.columns)]
# Reorder the dataframes and fix ambiguity on HA2/HA3
df_train.index = pd.RangeIndex(start=0, stop=len(df_train), step=1)
df_val.index = pd.RangeIndex(start=0, stop=len(df_val), step=1)
df_test.index = pd.RangeIndex(start=0, stop=len(df_test), step=1)
df_train = ha23ambigfix(df_train, mode=0)
df_val = ha23ambigfix(df_val, mode=0)
df_test = ha23ambigfix(df_test, mode=0)

try: # If DSSP columns are available, use them to filter a few bad residues based on the criteria in dihedral_purifier and dssp_purifier
    df_train = dihedral_purifier(df_train, drop_cols=True)
    df_train = dssp_purifier(df_train)
    df_val = dihedral_purifier(df_val, drop_cols=True)
    df_val = dssp_purifier(df_val)
except KeyError:
    pass

if configs["add_res_spec_feats"]:
    Add_res_spec_feats(df_train,include_onehot=configs.get("add_onehot_resname",False))
    Add_res_spec_feats(df_val,include_onehot=configs.get("add_onehot_resname",False))
    Add_res_spec_feats(df_test,include_onehot=configs.get("add_onehot_resname",False))
df_train = diff_targets(df_train,rings=False,coils=True)
df_val = diff_targets(df_val,rings=False,coils=True)
df_test = diff_targets(df_test,rings=False,coils=True)

# pay attention to 10x features
if configs["power_cols"]:
    df_train=feat_pwr(df_train,hbondd_cols+cos_cols,[2])
    df_train=feat_pwr(df_train,hbondd_cols,[-1,-2,-3])
    df_val=feat_pwr(df_val,hbondd_cols+cos_cols,[2])
    df_val=feat_pwr(df_val,hbondd_cols,[-1,-2,-3])
    df_test=feat_pwr(df_test,hbondd_cols+cos_cols,[2])
    df_test=feat_pwr(df_test,hbondd_cols,[-1,-2,-3])



# Drop RESNAME cols
if configs["drop_resname_cols"]:
    for col in df_train.columns:
        if "RESNAME" in col:
            dropped_cols.append(col)
# Drop prev/next cols
if configs["i_cols_only"]:
    all_columns=list(df_train.columns)
    prev_res_columns=[column for column in all_columns if 'i-1' in column]
    next_res_columns=[column for column in all_columns if 'i+1' in column]
    blosum_columns=[column for column in all_columns if column[:6]=='BLOSUM']
    dropped_cols+=prev_res_columns+next_res_columns+blosum_columns
# Feature space lifting
if configs["lift_feat_space"]:
    w=None
    b=None
    Lift_Space(df_train,non_numerical_cols,configs["lift_feat_space"],w,b)
    Lift_Space(df_val,non_numerical_cols,configs["lift_feat_space"],w,b)
    Lift_Space(df_test,non_numerical_cols,configs["lift_feat_space"],w,b)

dropped_cols+=dssp_pp_cols+dssp_energy_cols+["BMRB_RES_NUM","MATCHED_BMRB","CG","RCI_S2",'Unnamed: 0']


df_train=df_train.drop(set(dropped_cols)&set(df_train.columns),axis=1)
df_val=df_val.drop(set(dropped_cols)&set(df_val.columns),axis=1)
df_test=df_test.drop(set(dropped_cols)&set(df_test.columns),axis=1)

# If only spartap features
if configs["spartap_cols_only"]:
    df_train = df_train[spartap_cols]
    df_val = df_val[spartap_cols]
    df_test = df_test[spartap_cols]

# epochs_list, train_rmsd_list, test_rmsd_list= \
# kfold_crossval(3,spartap,"H",fc_eval,deep_model,[[128,64,32]],{"activ":"prelu","pretrain":None,"epochs":90,"opt_type":"adam","opt_override":True, "bnorm":True, "do":0.4})

print("Number of training residues:",len(df_train))
print("Number of validation residues:",len(df_val))
print("Number of testing residues:",len(df_test))

if configs.get("fixed_val_test",False):
    val_eps,train_rmsd,val_rmsd,test_rmsd,mod= \
    train_val_test(df_train,df_val,df_test,configs["atom_list"],eval(configs["evaluation"]),eval(configs["model"]),configs["args"],configs["kwargs"],mod_type=configs["mod_type"])
    with open('results.csv','a') as f:
        f.write(config_file.split("/")[-1].replace(".json","")+"\n")
        csv_writer=csv.writer(f,'excel')
        csv_writer.writerow("Epochs: %d\n Train_rmsd: %d\n Val_rmsd: %d\n Test_rmsd: %d"%(val_eps,train_rmsd,val_rmsd,test_rmsd))
    if "mod_saving_add" in configs:
        mod_saving_add=configs["mod_saving_add"]
    else:
        mod_saving_add="../models/%s"%config_file.split("/")[-1].replace(".json",".h5")
    mod.save(mod_saving_add)
    print("Model saved as",mod_saving_add)
else:
    dat=pd.concat(df_train,df_val)
    epochs_list, train_rmsd_list, test_rmsd_list= \
    kfold_crossval(configs.get("k",3),dat,configs["atom_list"],eval(configs["evaluation"]),eval(configs["model"]),configs["args"],configs["kwargs"],
    per=configs["per"],out='full',mod_type=configs["mod_type"],window=configs["window"],save_plot="svg")

    print(epochs_list)
    print(train_rmsd_list)
    print(test_rmsd_list)

    with open('results.csv','w') as f:
        csv_writer=csv.writer(f,'excel')
        for row in zip(list(epochs_list)*len(atom_list),train_rmsd_list,test_rmsd_list):
            csv_writer.writerow(row)

# CNN test
# epochs_list, train_rmsd_list, test_rmsd_list= \
# kfold_crossval(3,spartap,atom_list,rnn_eval,cnn_model,[[]],
# {'early_stop':None,'tol':10,'min_epochs':10,'epochs':200,'opt_type':'adam',"lrate":5e-5,'do':0.4},
# per=2,out='full',mod_type="rnn",window=7)
#'do':0.4, 'lstm_do':0.4, 'rec_do':0.4,
# kfold_crossval(10,spartap,'H',fc_eval,sparta_model,[],{'pretrain':'GL','tol':3,'epochs':100},per=5,out='full')
# 'pretrain':'GL','tol':5,'min_epochs':20,

# 

# epochs_list, train_rmsd_list, test_rmsd_list= \
# kfold_crossval(3,spartap,atom_list,ann_rnn_eval,ann_rnn_model,[[[256,128],100,[100]],[[50],[50]]],
# {'pretrain':"GL",'tol':10,'min_epochs':5,"epochs":[30,200,150,50],'opt_type':'adam',"lrate":5e-5,'do':[0.1, 0.1, 0.1,0.1], 
# "ann_pretrain":False,"rnn_pretrain":False,"rnn_pretrain_data":"../datasets/refDB.csv"},
# per=2,out='full',rnn=False,window=35,save_plot='svg')

#'min_epochs':5,"epochs":[30,200,150,50]



# arch_list=[[[128],64],[[128],128],[[128,128],64],[[128,64],32],[[128],128,128],[[256],256],[[256],128,64],[[256,128],64],[[256,128],128,64],[[8,8,8],8],[[8,8,8],8,4]]

# arch_results=architecture_selection(spartap,'H',rnn_eval,bidir_lstm_model,arch_list,[],{},rnn=True,save_fig_folder='rnn_archs')
# print(arch_results)

# with open('results.csv','w') as f:
#     csv_writer=csv.writer(f,'excel')
#     for key in arch_results:
#         csv_writer.writerow([key,arch_results[key]])
