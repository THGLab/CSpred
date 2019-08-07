#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

address=sys.argv[1]
train_df=pd.read_csv(address+"/train_analysis.csv")
test_df=pd.read_csv(address+"/test_analysis.csv")

atom_names = ['HA', 'H', 'CA', 'CB', 'C', 'N']
present_atoms=[atom for atom in atom_names if atom+"_pred" in train_df.columns]

for atom in present_atoms:
    train_pred=train_df[atom+"_pred"]
    train_real=train_df[atom+"_real"]
    test_pred=test_df[atom+"_pred"]
    test_real=test_df[atom+"_real"]
    plt.figure()
    plt.scatter(train_real,train_pred,s=0.1,label="Train")
    plt.scatter(test_real,test_pred,s=0.1,label="Test")
    plt.plot([-1,1],[-1,1],"r:")
    plt.legend()
    valid=test_df[atom+"_err"].notnull()
    err=test_df[valid][atom+"_err"].values
    err=np.sqrt(np.average(err**2))
    corr=np.corrcoef(test_df[valid][atom+"_pred"],test_df[valid][atom+"_real"])[0,1]
    plt.title(atom+" (RMSE:%.2f,CORR:%.2f)"%(err,corr))
    plt.show()
