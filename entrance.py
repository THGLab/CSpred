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

classification_model_path=configs.get("classification_model_path",None)
if classification_model_path is not None:
    from keras.models import load_model
    model=load_model(classification_model_path)
    Implant_classification(df_train,model)
    Implant_classification(df_val,model)
    Implant_classification(df_test,model)

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
    with open('results.txt','a') as f:
        f.write(config_file.split("/")[-1].replace(".json","")+"\n")
        f.write("Epochs: %d\n Train_rmsd: %d\n Val_rmsd: %d\n Test_rmsd: %d\n\n"%(val_eps,train_rmsd,val_rmsd,test_rmsd))
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

    with open('results.txt','w') as f:
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
