#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:51:30 2018

@author: bennett
"""

import pandas as pd
import numpy as np
import math
import sklearn as skl
import sklearn.model_selection
import keras
import matplotlib as mpl
import matplotlib.pyplot as plt

atom_names = ['HA', 'H', 'CA', 'CB', 'C', 'N']

col_phipsi = ['PHI_'+i for i in ['COS_i-1', 'SIN_i-1']]
col_phipsi += [i+j for i in ['PHI_', 'PSI_'] for j in ['COS_i', 'SIN_i']]
col_phipsi += ['PSI_'+i for i in ['COS_i+1', 'SIN_i+1']]
col_chi = [i+j+k for k in ['_i-1', '_i', '_i+1'] for i in ['CHI1_', 'CHI2_'] for j in ['COS', 'SIN', 'EXISTS']]
col_hbprev = ['O_'+i+'_i-1' for i in ['d_HA', '_COS_H', '_COS_A', '_EXISTS']]
col_hbond = [i+j+'_i' for i in ['Ha_', 'HN_', 'O_'] for j in ['d_HA', '_COS_H', '_COS_A', '_EXISTS']]
col_hbnext = ['HN_'+i+'_i+1' for i in ['d_HA', '_COS_H', '_COS_A', '_EXISTS']]
col_s2 = ['S2'+i for i in ['_i-1', '_i', '_i+1']]
struc_cols = col_phipsi + col_chi + col_hbprev + col_hbond + col_hbnext + col_s2
blosum_names = ['BLOSUM62_NUM_'+list(IUPACData.protein_letters_3to1.keys())[i].upper() for i in range(20)]
col_blosum = [blosum_names[i]+j for j in ['_i-1', '_i', '_i+1'] for i in range(20)]
seq_cols = col_blosum

train_path = '/home/bennett/Documents/Git_Collaborations_THG/office_home/training_data.pkl'
test_path = '/home/bennett/Documents/Git_Collaborations_THG/office_home/test_data.pkl'
train_data_df = pd.read_pickle(train_path)
test_data_df = pd.read_pickle(test_path)
train_data = train_data_df.drop(['FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM'], axis=1)
test_data = test_data_df.drop(['FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM'], axis=1)

train_path2 = '/Users/kcbennett/Documents/Git_Collaborations_THG/shiftpred/shiftx2_train_df.pkl'
test_path2 = '/Users/kcbennett/Documents/Git_Collaborations_THG/shiftpred/shiftx2_test_df.pkl'
train_df = pd.read_pickle(train_path2)
test_df = pd.read_pickle(test_path2)
train_data = train_df.drop(['FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM'], axis=1)
test_data = test_df.drop(['FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM'], axis=1)

lossf = 'mean_squared_error'


def sparta_model(data, atom, epochs, per, tol):
    '''Constructs a model from the given features and shifts for
    the requested atom.  The model is trained for the given number
    of epochs with the loss being checked every per epochs.  Training
    stops when this loss increases by more than tol. The architecture
    matches SPARTA+ (i.e., a single layer of 30 neurons).'''
    dat = data[data[atom].notnull()]
    feats = dat.drop(atom_names, axis=1).values
    shifts = dat[atom]
    shifts_mean = shifts.mean()
    shifts_std = shifts.std()
    shift_norm = (shifts - shifts_mean) / shifts_std
    shift_norm = shift_norm.values
    feat_train, feat_val, shift_train, shift_val = skl.model_selection.train_test_split(feats, shift_norm, test_size=0.2)
    dim_in = feats.shape[1]

    # Build model
    mod = keras.models.Sequential()
    mod.add(keras.layers.Dense(units=30, activation=act, input_dim=dim_in))
    mod.add(keras.layers.Dense(units=1, activation='linear'))
    mod.compile(loss=lossf, optimizer='sgd')

    # Initialize some outputs
    hist_list = []
    val_list = []

    # Train until the validation loss gets too far above the observed min
    val_min = 10 ** 10
    for i in range(int(epochs/per)):
        pt1 = mod.evaluate(feat_val, shift_val, verbose=0)
        val_list.append(pt1)

        if pt1 < val_min:
            val_min = pt1

        hist = mod.fit(feat_train, shift_train, batch_size=64, epochs=per)
        hist_list += hist.history['loss']
        pt2 = mod.evaluate(feat_val, shift_val, verbose=0)
        delt = pt1 - val_min
        print('The validation loss at round ' + str(i) + ' is ' + str(pt2))
        if delt > tol:
            print('Broke loop at round ' + str(i))
            break

    # Retrain model on full dataset for same number of epochs as was
    # needed to obtain min validation loss
    min_val_idx = min((val, idx) for (idx, val) in enumerate(val_list))[1]
    val_epochs = min_val_idx * per
    mod = keras.models.Sequential()
    mod.add(keras.layers.Dense(units=30, activation=act, input_dim=dim_in))
    mod.add(keras.layers.Dense(units=1, activation='linear'))
    mod.compile(loss=lossf, optimizer='sgd')
    mod.fit(feats, shift_norm, batch_size=64, epochs=val_epochs)

    return shifts_mean, shifts_std, val_list, hist_list, mod


def deep_model(activ, arch, lrate, mom, dec, data, atom, min_epochs, max_epochs, per, tol, do, reg, pretrain=True, bnorm=False, dropout=False, nest=False, rcoil=False, rccs_feat=False):
    '''Constructs a model from the given features and shifts for
    the requested atom.  The model is trained for the given number
    of epochs with the loss being checked every per epochs.  Training
    stops when this loss increases by more than tol. The arch argument
    is a list specifying the number of hidden units at each layer.'''
    dat = data[data[atom].notnull()]
    feats = dat.drop(atom_names, axis=1).values
    shifts = dat[atom]
    shifts_mean = shifts.mean()
    shifts_std = shifts.std()
    shift_norm = (shifts - shifts_mean) / shifts_std
    shift_norm = shift_norm.values
    feat_train, feat_val, shift_train, shift_val = skl.model_selection.train_test_split(feats, shift_norm, test_size=0.2)
    dim_in = feats.shape[1]
    
    opt = keras.optimizers.SGD(lr=lrate, momentum=mom, decay=dec, nesterov=nest)
    
    # Build model
    mod = keras.models.Sequential()
    mod.add(keras.layers.Dense(units=arch[0], activation=activ, input_dim=dim_in, kernel_regularizer=keras.regularizers.l1(reg)))
    for i in arch[1:]:
        mod.add(keras.layers.Dense(units=i, activation=activ, kernel_regularizer=keras.regularizers.l1(reg)))
        if bnorm:
            mod.add(keras.layers.BatchNormalization())
        if dropout:
            mod.add(keras.layers.Dropout(do))
    mod.add(keras.layers.Dense(units=1, activation='linear'))
    mod.compile(loss=lossf, optimizer=opt)

    # Initialize some outputs
    hist_list = []
    val_list = []

    # Train until the validation loss gets too far above the observed min
    val_min = 10 ** 10
    if pretrain:
        for i in range(int(max_epochs/per)):
            pt1 = mod.evaluate(feat_val, shift_val, verbose=0)
            val_list.append(pt1)

            if pt1 < val_min:
                val_min = pt1

            hist = mod.fit(feat_train, shift_train, batch_size=64, epochs=per)
            hist_list += hist.history['loss']
            pt2 = mod.evaluate(feat_val, shift_val, verbose=0)
            delt1 = pt1 - val_min
            delt2 = pt2 - val_min
            print('The validation loss at round ' + str(i) + ' is ' + str(pt2))
            if delt1 > tol and delt2 > tol:
                print('Broke loop at round ' + str(i))
                break
            if pt2 is np.nan:
                print('Broke loop because of NaN')
                break
        min_val_idx = min((val, idx) for (idx, val) in enumerate(val_list))[1]
        val_epochs = max(min_val_idx * per, min_epochs)
    else:
        val_epochs = min_epochs
    # Retrain model on full dataset for same number of epochs as was
    # needed to obtain min validation loss
    mod = keras.models.Sequential()
    mod.add(keras.layers.Dense(units=arch[0], activation=activ, input_dim=dim_in, kernel_regularizer=keras.regularizers.l1(reg)))
    for i in arch[1:]:
        mod.add(keras.layers.Dense(units=i, activation=activ, kernel_regularizer=keras.regularizers.l1(reg)))
        if bnorm:
            mod.add(keras.layers.BatchNormalization())
        if dropout:
            mod.add(keras.layers.Dropout(do))
    mod.add(keras.layers.Dense(units=1, activation='linear'))
    mod.compile(loss=lossf, optimizer=opt)
    mod.fit(feats, shift_norm, batch_size=64, epochs=val_epochs)

    return shifts_mean, shifts_std, val_list, hist_list, mod


def branch_model(activ, arch, lrate, mom, dec, data, atom, min_epochs, max_epochs, per, tol, do, reg, pretrain=True, bnorm=False, dropout=False, nest=False, rcoil=False, rccs_feat=False, merge_softmax=False):
    '''Constructs a model from the given features and shifts for
    the requested atom.  The model is trained for the given number
    of epochs with the loss being checked every per epochs.  Training
    stops when this loss increases by more than tol. The arch argument
    is a list specifying the number of hidden units at each layer.
    The architecture for this network is split into two branches that
    handle structural and sequential information respectively rather
    than being fully connected.  The arch parameter is thus given as a
    list, the first two elements of which are themselves lists giving 
    the neurons for each layer of each branch and the remaining elements
    giving the neurons for the remainder of the network, after the two 
    branches meet. Parameters rcoil and rccs_feat determine whether 
    random coil chemical shifts are used and, if so, whether they are
    subtracted off or used as features.'''
    
    # First define the column names for easy access
    col_phipsi = ['PHI_'+i for i in ['COS_i-1', 'SIN_i-1']]
    col_phipsi += [i+j for i in ['PHI_', 'PSI_'] for j in ['COS_i', 'SIN_i']]
    col_phipsi += ['PSI_'+i for i in ['COS_i+1', 'SIN_i+1']]
    col_chi = [i+j+k for k in ['_i-1', '_i', '_i+1'] for i in ['CHI1_', 'CHI2_'] for j in ['COS', 'SIN', 'EXISTS']]
    col_hbprev = ['O_'+i+'_i-1' for i in ['d_HA', '_COS_H', '_COS_A', '_EXISTS']]
    col_hbond = [i+j+'_i' for i in ['Ha_', 'HN_', 'O_'] for j in ['d_HA', '_COS_H', '_COS_A', '_EXISTS']]
    col_hbnext = ['HN_'+i+'_i+1' for i in ['d_HA', '_COS_H', '_COS_A', '_EXISTS']]
    col_s2 = ['S2'+i for i in ['_i-1', '_i', '_i+1']]
    struc_cols = col_phipsi + col_chi + col_hbprev + col_hbond + col_hbnext + col_s2
    blosum_names = ['BLOSUM62_NUM_'+list(IUPACData.protein_letters_3to1.keys())[i].upper() for i in range(20)]
    col_blosum = [blosum_names[i]+j for j in ['_i-1', '_i', '_i+1'] for i in range(20)]
    seq_cols = col_blosum

    # Now seperate out the features and shifts
    dat = data[data[atom].notnull()]
    feats = dat.drop(atom_names, axis=1)
    struc_feats = dat[struc_cols].values
    seq_feats = dat[seq_cols].values
    if rcoil:
        rccs_feats = dat['RC_' + atom]
    if rcoil and not rccs_feat:
        rccs_names = ['RC_' + i for i in atom_names]
        raw_shifts = dat[atom]
        shifts = raw_shifts - rccs_feats
    else:
        shifts = dat[atom]
    
    # Standardize the shifts
    shifts_mean = shifts.mean()
    shifts_std = shifts.std()
    shift_norm = (shifts - shifts_mean) / shifts_std
    
    # For early stopping pretraining, do a train_test_split
    feat_train, feat_val, shift_train, shift_val = skl.model_selection.train_test_split(feats, shift_norm, test_size=0.2)
    struc_feat_train = feat_train[struc_cols].values
    struc_feat_val = feat_val[struc_cols].values
    seq_feat_train = feat_train[seq_cols].values
    seq_feat_val = feat_val[seq_cols].values
    shift_train = shift_train.values
    shift_val = shift_val.values
    
    # If using random coil chemical shifts as input features, standardize them
    if rcoil and rccs_feat:
        rc_feat_train = feat_train['RC_' + atom].values
        rc_feat_val = feat_val['RC_' + atom].values
        rc_feat_train = (rc_feat_train - shifts_mean) / shifts_std
        rc_feat_val = (rc_feat_val - shifts_mean) / shifts_std
    
    # Define input dimensions and optimization procedure
    struc_dim_in = struc_feat_train.shape[1]
    seq_dim_in = seq_feat_train.shape[1]
    opt = keras.optimizers.SGD(lr=lrate, momentum=mom, decay=dec, nesterov=nest)
    
    # Build model
    # Process structure and sequence information seperately
    struc_in = keras.layers.Input(shape=(struc_dim_in,), name='struc_input')
    struc = keras.layers.Lambda(lambda x: x)(struc_in)
    seq_in = keras.layers.Input(shape=(seq_dim_in,), name='seq_input')
    seq = keras.layers.Lambda(lambda x: x)(seq_in)
    for i in arch[0]:
        struc = keras.layers.Dense(i, activation=activ, kernel_regularizer=keras.regularizers.l1(reg[0]))(struc)
        if bnorm:
            struc = keras.layers.BatchNormalization()(struc)
        if dropout:
            struc = keras.layers.Dropout(do)(struc)
    for i in arch[1]:
        seq = keras.layers.Dense(i, activation=activ, kernel_regularizer=keras.regularizers.l1(reg[1]))(seq)
        if bnorm:
            struc = keras.layers.BatchNormalization()(seq)
        if dropout:
            struc = keras.layers.Dropout(do)(seq)

    # Concatenate structure and sequence (and random coil) information
    if rcoil and rccs_feat:
        rccs = keras.layers.Input(shape=(1,), name='rccs_input')
        merge = keras.layers.concatenate([struc, seq, rccs])
    else:
        merge = keras.layers.concatenate([struc, seq])
    
    # Process merged info, either a single layer with softmax or 
    # multiple layers with same network-wide activation function
    if merge_softmax:
        merge = keras.layers.Dense(arch[2], activation='softmax', kernel_regularizer=keras.regularizers.l1(reg[2]))(merge)
    else:
        for i in arch[2:]:
            merge = keras.layers.Dense(i, activation=activ, kernel_regularizer=keras.regularizers.l1(reg[2]))(merge)
    
    predicted_shift = keras.layers.Dense(units=1, activation='linear')(merge)
    if rcoil and rccs_feat:
        model = keras.Model(inputs=[struc_in, seq_in, rccs], outputs=predicted_shift)
    else:
        model = keras.Model(inputs=[struc_in, seq_in], outputs=predicted_shift)

    model.compile(loss='mean_squared_error', optimizer=opt)

    # Initialize some outputs
    hist_list = []
    val_list = []

    # Train until the validation loss gets too far above the observed min
    val_min = 10 ** 10
    if pretrain:
        for i in range(int(max_epochs/per)):
            if rcoil and rccs_feat:
                pt1 = model.evaluate([struc_feat_val, seq_feat_val, rc_feat_val], shift_val, verbose=0)
                hist = model.fit([struc_feat_train, seq_feat_train, rc_feat_train], shift_train, batch_size=64, epochs=per)
                pt2 = model.evaluate([struc_feat_val, seq_feat_val, rc_feat_val], shift_val, verbose=0)
            else:
                pt1 = model.evaluate([struc_feat_val, seq_feat_val], shift_val, verbose=0)
                hist = model.fit([struc_feat_train, seq_feat_train], shift_train, batch_size=64, epochs=per)
                pt2 = model.evaluate([struc_feat_val, seq_feat_val], shift_val, verbose=0)
            val_list.append(pt1)
            if pt1 < val_min:
                val_min = pt1
            hist_list += hist.history['loss']
            delt1 = pt1 - val_min
            delt2 = pt2 - val_min
            print('The validation loss at round ' + str(i) + ' is ' + str(pt2))
            if delt1 > tol and delt2 > tol:
                print('Broke loop at round ' + str(i))
                break
            if pt2 is np.nan:
                print('Broke loop because of NaN')
                break
        min_val_idx = min((val, idx) for (idx, val) in enumerate(val_list))[1]
        val_epochs = max(min_val_idx * per, min_epochs)
    else:
        val_epochs = min_epochs
    # Retrain model on full dataset for same number of epochs as was
    # needed to obtain min validation loss
    struc_in = keras.layers.Input(shape=(struc_dim_in,), name='struc_input')
    struc = keras.layers.Lambda(lambda x: x)(struc_in)
    seq_in = keras.layers.Input(shape=(seq_dim_in,), name='seq_input')
    seq = keras.layers.Lambda(lambda x: x)(seq_in)

    for i in arch[0]:
        struc = keras.layers.Dense(i, activation=activ, kernel_regularizer=keras.regularizers.l1(reg[0]))(struc)
        if bnorm:
            struc = keras.layers.BatchNormalization()(struc)
        if dropout:
            struc = keras.layers.Dropout(do)(struc)
    for i in arch[1]:
        seq = keras.layers.Dense(i, activation=activ, kernel_regularizer=keras.regularizers.l1(reg[1]))(seq)
        if bnorm:
            struc = keras.layers.BatchNormalization()(seq)
        if dropout:
            struc = keras.layers.Dropout(do)(seq)

    # Concatenate structure and sequence (and random coil) information
    if rcoil and rccs_feat:
        rccs = keras.layers.Input(shape=(1,), name='rccs_input')
        merge = keras.layers.concatenate([struc, seq, rccs])
    else:
        merge = keras.layers.concatenate([struc, seq])
    
    # Process merged info, either a single layer with softmax or 
    # multiple layers with same network-wide activation function
    if merge_softmax:
        merge = keras.layers.Dense(arch[2], activation='softmax', kernel_regularizer=keras.regularizers.l1(reg[2]))(merge)
    else:
        for i in arch[2:]:
            merge = keras.layers.Dense(i, activation=activ, kernel_regularizer=keras.regularizers.l1(reg[2]))(merge)
    
    predicted_shift = keras.layers.Dense(units=1, activation='linear')(merge)
    if rcoil and rccs_feat:
        model = keras.Model(inputs=[struc_in, seq_in, rccs], outputs=predicted_shift)
    else:
        model = keras.Model(inputs=[struc_in, seq_in], outputs=predicted_shift)

    model.compile(loss='mean_squared_error', optimizer=opt)
    
    if rcoil and rccs_feat:
        model.fit([struc_feats, seq_feats, rccs_feats], shift_norm, batch_size=64, epochs=val_epochs)
    else:
        model.fit([struc_feats, seq_feats], shift_norm, batch_size=64, epochs=val_epochs)

    return shifts_mean, shifts_std, val_list, hist_list, model



def sparta_eval(mean, std, data, model, atom):
    dat = data[data[atom].notnull()]
    feats = dat.drop(atom_names, axis=1).values
    shifts = dat[atom].values
    shifts_norm = (shifts - mean) / std
    return model.evaluate(feats, shifts_norm, verbose=0)

def branch_eval(mean, std, data, model, atom):
    dat = data[data[atom].notnull()]
    feats = dat.drop(atom_names, axis=1).values
    struc_feats = dat[struc_cols].values
    seq_feats = dat[seq_cols].values
    shifts = dat[atom].values
    shifts_norm = (shifts - mean) / std
    return model.evaluate([struc_feats, seq_feats], shifts_norm, verbose=0)
    

def sparta_eval2(mean, std, data, model, atom):
    dat = data[data[atom].notnull()]
    feats = dat.drop(atom_names, axis=1).values
    shifts = dat[atom].values
    preds = model.predict(feats)
    preds = preds * std + mean
    sq = (preds - shifts) ** 2
    return sq.mean()


def sparta_eval3(mean, std, data, model, atom):
    dat = data[data[atom].notnull()]
    feats = dat.drop(atom_names, axis=1).values
    shifts = dat[atom].values
    shifts_norm = (shifts - mean) / std
    preds = model.predict(feats)
    sq = (preds - shifts_norm) ** 2
    return sq.mean()

    
cmean, cstd, cval_list, chist_list, sparta_cmod = sparta_model(train_data, 'C', 25000, 100, 5*10**-3)
hmean, hstd, hval_list, hhist_list, sparta_hmod = sparta_model(train_data, 'H', 25000, 100, 5*10**-3)
nmean, nstd, nval_list, nhist_list, sparta_nmod = sparta_model(train_data, 'N', 25000, 100, 5*10**-3)
camean, castd, caval_list, cahist_list, sparta_camod = sparta_model(train_data, 'CA', 25000, 100, 5*10**-3)
cbmean, cbstd, cbval_list, cbhist_list, sparta_cbmod = sparta_model(train_data, 'CB', 25000, 100, 10**-2)
hamean, hastd, haval_list, hahist_list, sparta_hamod = sparta_model(train_data, 'HA', 25000, 100, 5*10**-3)

cmean, cstd, redo_cval_list3, redo_deep_chist_list3, deep_mod1= deep_model('relu', [100, 60, 30], 0.01, 0.70, 5*10**-6, train_data, 'C', 50, 2000, 25, 10**-2, do=0.2, reg=0, pretrain=True, bnorm=True, dropout=True, nest=True)

branch_arch = [[60, 50, 30], [60, 50, 30], 30]
cmean, cstd, redo_cval_list3, redo_deep_chist_list3, branch_mod1 = branch_model('relu', branch_arch, 0.01, 0.70, 5*10**-6, train_data, 'C', 50, 2000, 25, 10**-2, do=0.2, reg=0, pretrain=True, bnorm=True, dropout=True, nest=True)

c_error = np.sqrt(sparta_eval(cmean, cstd, test_data, sparta_cmod, 'C')) * cstd
h_error = np.sqrt(sparta_eval(hmean, hstd, test_data, sparta_hmod, 'H')) * hstd
n_error = np.sqrt(sparta_eval(nmean, nstd, test_data, sparta_nmod, 'N')) * nstd
ca_error = np.sqrt(sparta_eval(camean, castd, test_data, sparta_camod, 'CA')) * castd
cb_error = np.sqrt(sparta_eval(cbmean, cbstd, test_data, sparta_cbmod, 'CB')) * cbstd
ha_error = np.sqrt(sparta_eval(hamean, hastd, test_data, sparta_hamod, 'HA')) * hastd

training_error = np.sqrt(branch_eval(cmean, cstd, train_data, branch_mod1, 'C')) * cstd
testdat_error = np.sqrt(branch_eval(cmean, cstd, test_data, branch_mod1, 'C')) * cstd

np.sqrt(sparta_eval(nmean, nstd, train_data, sparta_nmod, 'N')) * nstd
np.sqrt(sparta_eval(cmean, cstd, test_data, sparta_cmod, 'C')) * cstd
np.sqrt(sparta_eval(hmean, hstd, train_data, sparta_hmod, 'H')) * hstd
np.sqrt(sparta_eval(camean, castd, train_data, sparta_camod, 'CA')) * castd
np.sqrt(sparta_eval(cbmean, cbstd, train_data, sparta_cbmod, 'CB')) * cbstd
np.sqrt(sparta_eval(hamean, hastd, train_data, sparta_hamod, 'HA')) * hastd

archs = [[100, 60, 30]]
drops = [0.4]
lrates = [0.005, 0.01, 0.05, 0.1]
decays = [10**-6, 10**-5, 10**-4]
momenta = [.7, .9, .95]
activs = ['tanh', 'relu']
nester = [True]

results = []
for ip in drops:
    for i in lrates:
        for j in decays:
            for k in activs:
                for l in momenta:
                    for m in nester:
                        cmean, cstd, cval_list3, deep_chist_list3, deep100603015_cmod_bndo = deep_model(k, [100, 60, 30], i, l, j, train_data, 'C', 
                                                                                            min_epochs=50, max_epochs=2000, per=25, tol=10**-2, do=ip, reg=0.01, bnorm=True, dropout=True, nest=m)
                        min_val_idx = min((val, idx) for (idx, val) in enumerate(cval_list3))[1]
                        val_epochs = max(min_val_idx * 25, 50)
                        training_error = np.sqrt(sparta_eval(cmean, cstd, train_data, deep100603015_cmod_bndo, 'C')) * cstd
                        testdat_error = np.sqrt(sparta_eval(cmean, cstd, test_data, deep100603015_cmod_bndo, 'C')) * cstd
                        res = [[100, 60, 30], i, j, k, l, m, ip, val_epochs, training_error, testdat_error]
                        print('Finished round ' + str(res))
                        results.append(res)
        resdf = pd.DataFrame(results, columns=['Architecture', 'Learning_Rate', 'Decay', 'Activation', 'Momentum', 'Nesterov', 'Dropout', 'Epochs', 'Training_Error', 'Test_Error'])
        resdf.to_pickle('/Users/kcbennett/Documents/Git_Collaborations_THG/shiftpred/hyper_search2_reg.pkl')
            
            
results_df = pd.DataFrame(results, columns=['Architecture', 'Learning_Rate', 'Decay', 'Activation', 'Momentum', 'Nesterov', 'Dropout', 'Epochs', 'Training_Error', 'Test_Error'])
results_df.to_pickle('/Users/kcbennett/Documents/Git_Collaborations_THG/shiftpred/hyper_search.pkl')         
            
            
  




















