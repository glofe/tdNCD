# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 17:36:17 2021

@author: king
"""

from scipy.stats.stats import scoreatpercentile
from createNet import NET
from parameters import param
from sklearn import svm, metrics
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score, precision_score, recall_score
from sklearn.model_selection import ShuffleSplit, GridSearchCV

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import scipy.io as sio
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# load
FC = pickle.load(open(param['dirCache'] + r'/corr_z.txt', 'rb'))
groupInfo = sio.loadmat(param['dirCache'] + r'/groupInfo_delNan.mat')['groupInfo_']
tdNCD = sio.loadmat(param['dirCache'] + r'/tdNCD_50tp5s.mat')['tdNCD_50tp5s']
    
tdNCDTotal = []
labelTotal = []
labelCenter_oneHot = []
for c in range(param['nCenter']):
    for g in range(param['nGroup']):
        for s in range(tdNCD[c,g].shape[0]):
            tmp = []
            for td in range(24):
                for r in range(param['nROI']):
                    tmp.append(tdNCD[c,g][s,td,r])
            tdNCDTotal.append(tmp)
            labelTotal.append(g)
            tmpCenter_oneHot = [0] * (param['nCenter'] + 1)
            tmpCenter_oneHot[0] = 1
            tmpCenter_oneHot[c + 1] = 1
            labelCenter_oneHot.append(tmpCenter_oneHot)
tdNCDTotal = np.array(tdNCDTotal)
labelTotal = np.array(labelTotal)
labelCenter_oneHot = np.array(labelCenter_oneHot)

if param['calcHP']:
    AccList_FCN = []
    if param['useFCNet']:
        F1Test = np.zeros((param['nCenter'], len(param['HP_iter']), len(param['HP_lr'])))
        for c in range(param['nCenter']):
            testCenterList = np.arange(7)[np.arange(7) != c].tolist()
            F1Test_ = np.zeros((param['nCenter'] - 1, len(param['HP_iter']), len(param['HP_lr'])))
            for ct in testCenterList:
                ###########################
                #   data
                ###########################
                
                if not param['use_tdNCD']:
                    dataTrain = FC[(labelCenter_oneHot[:,c + 1] == 0) * (labelCenter_oneHot[:,ct + 1] == 0),:]
                    dataTest = FC[labelCenter_oneHot[:,ct + 1] == 1,:]
                    dataValid = FC[labelCenter_oneHot[:,c + 1] == 1,:]
                else:
                    dataTrain= np.c_[FC[(labelCenter_oneHot[:,c + 1] == 0) * (labelCenter_oneHot[:,ct + 1] == 0),:],tdNCDTotal[(labelCenter_oneHot[:,c + 1] == 0) * (labelCenter_oneHot[:,ct + 1] == 0),:]]
                    dataTest = np.c_[FC[(labelCenter_oneHot[:,c + 1] == 0) * (labelCenter_oneHot[:,ct + 1] == 1),:],tdNCDTotal[(labelCenter_oneHot[:,c + 1] == 0) * (labelCenter_oneHot[:,ct + 1] == 1),:]]
                    dataValid = np.c_[FC[labelCenter_oneHot[:,c + 1] == 1,:],tdNCDTotal[labelCenter_oneHot[:,c + 1] == 1,:]]

                labelTrain = labelTotal[(labelCenter_oneHot[:,c + 1] == 0) * (labelCenter_oneHot[:,ct + 1] == 0)]
                labelTest = labelTotal[labelCenter_oneHot[:,ct + 1] == 1]
                labelValid = labelTotal[labelCenter_oneHot[:,c + 1] == 1]

                if param['nClassify'] == 2:
                    dataTrain = dataTrain[labelTrain != 1,:]    # 3 -> 2(NC + AD)
                    dataTest = dataTest[labelTest != 1,:]
                    dataValid = dataValid[labelValid != 1]
                    labelTrain = labelTrain[labelTrain != 1]
                    labelTest = labelTest[labelTest != 1]
                    labelValid = labelValid[labelValid != 1]
                    labelTrain[labelTrain == 2] = 1
                    labelTest[labelTest == 2] = 1
                    labelValid[labelValid == 2] = 1
                
                
                dataTrain = torch.from_numpy(dataTrain).float()
                labelTrain = torch.from_numpy(labelTrain).long()
                dataTest = torch.from_numpy(dataTest).float()
                labelTest = torch.from_numpy(labelTest).long()
                dataValid = torch.from_numpy(dataValid).float()
                labelValid = torch.from_numpy(np.array(labelValid)).long()
                
                ###########################
                #   FCNet
                ###########################
                # train
                for li in range(len(param['HP_lr'])):
                    # model
                    if not param['use_tdNCD']:
                        net = NET(param['nROI'] * (param['nROI'] - 1) // 2, 4096, 1024, 128, param['nClassify'])     # 5 layer FCnet
                    else:
                        net = NET(param['nROI'] * (param['nROI'] - 1) // 2 + 24 * param['nROI'], 4096, 1024, 128, param['nClassify'], dropout_p=0)     # 5 layer FCnet
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(net.parameters(), lr=param['HP_lr'][li])
                    if torch.cuda.is_available() and param['2cuda']:
                        net = net.cuda()
                        criterion = criterion.cuda()
                        dataTrain = dataTrain.cuda()
                        labelTrain = labelTrain.cuda()
                        dataValid = dataValid.cuda()
                        labelValid = labelValid.cuda()
                    iInd = 0
                    stime = time.time()
                    for i in range(param['HP_iter'][-1]):
                        optimizer.zero_grad()
                        if torch.cuda.is_available() and param['2cuda']:
                            dataTrain = dataTrain.cuda()
                            labelTrain = labelTrain.cuda()
                        outputs = net.forward(dataTrain)
                        loss = criterion(outputs, labelTrain)
                        loss.backward()
                        optimizer.step()
                        if i == param['HP_iter'][iInd] - 1:
                            print('valid center: %d, test center: %d, iterTrain: %d, lr: %f, time: %f, loss: %f' % (c + 1, ct + 1, i+1, param['HP_lr'][li], time.time() - stime, loss.cpu().data))
                            labelPred = torch.argmax(F.softmax(net.forward(dataTest),dim=1),dim=1).data
                            print('labelTest 0 num: %d, 1 num: %d, 2 num: %d' % (torch.sum(labelTest == 0), torch.sum(labelTest == 1), torch.sum(labelTest == 2)))
                            print('pred 0 num: %d, 1 num: %d, 2 num: %d' % (torch.sum(labelPred == 0), torch.sum(labelPred == 1), torch.sum(labelPred == 2)))
                            print('acc: %f, F1: %f' % (torch.sum(labelPred == labelTest).numpy() / len(labelTest), f1_score(labelTest, labelPred, average='binary' if param['nClassify'] == 2 else 'macro')))
                            F1Test_[testCenterList.index(ct), iInd, li] = f1_score(labelTest, labelPred, average='binary' if param['nClassify'] == 2 else 'macro')
                            iInd += 1

                            stime = time.time()
            print(F1Test_)
            F1Test[c,:,:] = np.mean(F1Test_, 0)

        if not param['use_tdNCD']:
            np.save(param['dirCache'] + '/FCnet5_HP_classify_' + str(param['nClassify']) + '_use_corr_byF1', F1Test)
        else:
            np.save(param['dirCache'] + '/FCnet5_HP_classify_' + str(param['nClassify']) + '_use_corr_tdNCD_byF1', F1Test)
        print(F1Test)

if param['validation']:
    validRes = {'fprList': [], 'ptrList': [], 'fprList_add_tdNCD': [], 'ptrList_add_tdNCD': []}
    valPred = []
    TP_total = np.zeros((param['nCenter'], 2))
    TN_total = np.zeros((param['nCenter'], 2))
    FP_total = np.zeros((param['nCenter'], 2))
    FN_total = np.zeros((param['nCenter'], 2))
    accCenter = np.zeros((param['nCenter'], 2))
    F1Center = np.zeros((param['nCenter'], 2))
    SENCenter = np.zeros((param['nCenter'], 2))
    SPECenter = np.zeros((param['nCenter'], 2))
    PPVCenter = np.zeros((param['nCenter'], 2))
    NPVCenter = np.zeros((param['nCenter'], 2))
    roiWieght = np.zeros((param['nCenter'], 24, param['nROI']))
    for c in range(param['nCenter']):

        dataTrain = FC[labelCenter_oneHot[:,c + 1] == 0,:]
        dataValid = FC[labelCenter_oneHot[:,c + 1] == 1,:]
        dataTrain_add_tdNCD= np.c_[FC[labelCenter_oneHot[:,c + 1] == 0,:],tdNCDTotal[labelCenter_oneHot[:,c + 1] == 0,:]]
        dataValid_add_tdNCD = np.c_[FC[labelCenter_oneHot[:,c + 1] == 1,:],tdNCDTotal[labelCenter_oneHot[:,c + 1] == 1,:]]
        labelTrain = labelTotal[labelCenter_oneHot[:,c + 1] == 0]
        labelValid = labelTotal[labelCenter_oneHot[:,c + 1] == 1]
        if param['useMCIAD']:
            dataTrain = dataTrain[labelTrain != 0,:]   
            dataValid = dataValid[labelValid != 0,:]
            dataTrain_add_tdNCD  = dataTrain_add_tdNCD[labelTrain != 0,:]   
            dataValid_add_tdNCD = dataValid_add_tdNCD[labelValid != 0,:]
            labelTrain = labelTrain[labelTrain != 0]
            labelValid = labelValid[labelValid != 0]
            labelTrain[labelTrain == 2] = 1
            labelValid[labelValid == 2] = 1
        elif param['nClassify'] == 2:
            dataTrain = dataTrain[labelTrain != 1,:]   
            dataValid = dataValid[labelValid != 1,:]
            dataTrain_add_tdNCD  = dataTrain_add_tdNCD[labelTrain != 1,:]   
            dataValid_add_tdNCD = dataValid_add_tdNCD[labelValid != 1,:]
            labelTrain = labelTrain[labelTrain != 1]
            labelValid = labelValid[labelValid != 1]
            labelTrain[labelTrain == 2] = 1
            labelValid[labelValid == 2] = 1
        dataTrain = torch.from_numpy(dataTrain).float()
        dataValid = torch.from_numpy(dataValid).float()
        dataTrain_add_tdNCD = torch.from_numpy(dataTrain_add_tdNCD).float()
        dataValid_add_tdNCD = torch.from_numpy(dataValid_add_tdNCD).float()
        labelTrain = torch.from_numpy(np.array(labelTrain)).long()
        labelValid = torch.from_numpy(np.array(labelValid)).long()

        F1Test = np.load(param['dirCache'] + '/FCnet5_HP_classify_' + str(param['nClassify']) + '_use_corr_byF1.npy')
        F1Test_add_tdNCD = np.load(param['dirCache'] + '/FCnet5_HP_classify_' + str(param['nClassify']) + '_use_corr_tdNCD_byF1.npy')

        '''validation'''
        print('*' * 60 )
        print('center: ' + str(c + 1))
        bestHP_ind_byF1 = [0,0,0]
        bestHP_ind_byF1_add_tdNCD = [0,0,0]
        for m in range(F1Test[c,:,:].shape[0]):
            for n in range(F1Test[c,:,:].shape[1]):
                if F1Test[c,m,n] > bestHP_ind_byF1[0]:
                    bestHP_ind_byF1[0] = F1Test[c,m,n]
                    bestHP_ind_byF1[1] = m
                    bestHP_ind_byF1[2] = n
                if F1Test_add_tdNCD[c,m,n] > bestHP_ind_byF1_add_tdNCD[0]:
                    bestHP_ind_byF1_add_tdNCD[0] = F1Test_add_tdNCD[c,m,n]
                    bestHP_ind_byF1_add_tdNCD[1] = m
                    bestHP_ind_byF1_add_tdNCD[2] = n
        print('best HP by F1 (corr): ')
        print(param['HP_iter'][bestHP_ind_byF1[1]])
        print(param['HP_lr'][bestHP_ind_byF1[2]])
        net = NET(param['nROI'] * (param['nROI'] - 1) // 2, 4096, 1024, 128, param['nClassify'])     # 5 layer FCnet

        tmpValPred = []
        if param['load_param']:
            net.load_state_dict(torch.load(param['dirCache'] + '/param_FCNet_classify' + str(param['nClassify']) + '_use_corr_center' + str(c + 1) + '_iter' + str(param['HP_iter'][bestHP_ind_byF1[1]]) + '_lr' + str(param['HP_lr'][bestHP_ind_byF1[2]]) + '.pth'))
        else:
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=param['HP_lr'][bestHP_ind_byF1[2]])
            for i in range(param['HP_iter'][bestHP_ind_byF1[1]]):
                optimizer.zero_grad()
                outputs = net.forward(dataTrain)
                loss = criterion(outputs, labelTrain)
                loss.backward()
                optimizer.step()
            torch.save(obj=net.state_dict(), f=param['dirCache'] + '/param_FCNet_classify' + str(param['nClassify']) + '_use_corr_center' + str(c + 1) + '_iter' + str(param['HP_iter'][bestHP_ind_byF1[1]]) + '_lr' + str(param['HP_lr'][bestHP_ind_byF1[2]]) + '.pth')

        if param['nClassify'] == 2:
            scorePred = F.softmax(net.forward(dataValid),dim=1).detach().numpy()
            tmpValPred.append(scorePred.copy())
            fpr,ptr,_ = roc_curve(labelValid, scorePred[:,1], pos_label=1)
            validRes['fprList'].append(fpr)
            validRes['ptrList'].append(ptr)
            labelPred = torch.argmax(F.softmax(net.forward(dataValid),dim=1),dim=1).data
            print('data validation acc[F1]: %f, F1[F1]: %f' % (torch.sum(labelPred == labelValid).numpy() / len(labelValid), f1_score(labelValid, labelPred, average='binary' if param['nClassify'] == 2 else 'macro')))
            TP = np.sum((labelPred.numpy() == 1) * (labelPred.numpy() == labelValid.numpy()))
            TN = np.sum((labelPred.numpy() == 0) * (labelPred.numpy() == labelValid.numpy()))
            FP = np.sum((labelPred.numpy() == 0) * (labelPred.numpy() != labelValid.numpy()))
            FN = np.sum((labelPred.numpy() == 1) * (labelPred.numpy() != labelValid.numpy()))
            TP_total[c,0] = TP
            TN_total[c,0] = TN
            FP_total[c,0] = FP
            FN_total[c,0] = FN
            accCenter[c,0] = torch.sum(labelPred == labelValid).numpy() / len(labelValid)
            F1Center[c,0] = f1_score(labelValid, labelPred, average='binary' if param['nClassify'] == 2 else 'macro')
            SENCenter[c,0] = recall_score(labelValid, labelPred)
            SPECenter[c,0] = TN / (FP + TN)
            PPVCenter[c,0] = precision_score(labelValid, labelPred)
            NPVCenter[c,0] = TN / (TN + FN)
        else:
            scorePred = F.softmax(net.forward(dataValid),dim=1).detach().numpy()
            tmpValPred.append(scorePred.copy())
            labelPred = torch.argmax(F.softmax(net.forward(dataValid),dim=1),dim=1).data
            accCenter[c,0] = torch.sum(labelPred == labelValid).numpy() / len(labelValid)
            F1Center[c,0] = f1_score(labelValid, labelPred, average='binary' if param['nClassify'] == 2 else 'macro')

        
        print('best HP by F1 (corr + tdNCD): ')
        print(param['HP_iter'][bestHP_ind_byF1_add_tdNCD[1]])
        print(param['HP_lr'][bestHP_ind_byF1_add_tdNCD[2]])
        net = NET(param['nROI'] * (param['nROI'] - 1) // 2 + 24 * param['nROI'], 4096, 1024, 128, param['nClassify'])     # 5 layer FCnet
        if param['load_param']:
            net.load_state_dict(torch.load(param['dirCache'] + '/param_FCNet_classify' + str(param['nClassify']) + '_use_corr_tdNCD_center' + str(c + 1) + '_iter' + str(param['HP_iter'][bestHP_ind_byF1_add_tdNCD[1]]) + '_lr' + str(param['HP_lr'][bestHP_ind_byF1_add_tdNCD[2]]) + '.pth'))
        else:
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=param['HP_lr'][bestHP_ind_byF1_add_tdNCD[2]])
            for i in range(param['HP_iter'][bestHP_ind_byF1_add_tdNCD[1]]):
                optimizer.zero_grad()
                outputs = net.forward(dataTrain_add_tdNCD)
                loss = criterion(outputs, labelTrain)
                loss.backward()
                optimizer.step()
            torch.save(obj=net.state_dict(), f=param['dirCache'] + '/param_FCNet_classify' + str(param['nClassify']) + '_use_corr_tdNCD_center' + str(c + 1) + '_iter' + str(param['HP_iter'][bestHP_ind_byF1_add_tdNCD[1]]) + '_lr' + str(param['HP_lr'][bestHP_ind_byF1_add_tdNCD[2]]) + '.pth')

        if param['nClassify'] == 2:
            scorePred = F.softmax(net.forward(dataValid_add_tdNCD),dim=1).detach().numpy()
            tmpValPred.append(scorePred.copy())
            fpr,ptr,_ = roc_curve(labelValid, scorePred[:,1], pos_label=1)
            validRes['fprList_add_tdNCD'].append(fpr)
            validRes['ptrList_add_tdNCD'].append(ptr)
            labelPred = torch.argmax(F.softmax(net.forward(dataValid_add_tdNCD),dim=1),dim=1).data
            print('data validation  acc[F1]: %f, F1[F1]: %f' % (torch.sum(labelPred == labelValid).numpy() / len(labelValid), f1_score(labelValid, labelPred, average='binary' if param['nClassify'] == 2 else 'macro')))
            TP = np.sum((labelPred.numpy() == 1) * (labelPred.numpy() == labelValid.numpy()))
            TN = np.sum((labelPred.numpy() == 0) * (labelPred.numpy() == labelValid.numpy()))
            FP = np.sum((labelPred.numpy() == 0) * (labelPred.numpy() != labelValid.numpy()))
            FN = np.sum((labelPred.numpy() == 1) * (labelPred.numpy() != labelValid.numpy()))
            TP_total[c,1] = TP
            TN_total[c,1] = TN
            FP_total[c,1] = FP
            FN_total[c,1] = FN
            accCenter[c,1] = torch.sum(labelPred == labelValid).numpy() / len(labelValid)
            F1Center[c,1] = f1_score(labelValid, labelPred, average='binary' if param['nClassify'] == 2 else 'macro')
            SENCenter[c,1] = recall_score(labelValid, labelPred)
            SPECenter[c,1] = TN / (FP + TN)
            PPVCenter[c,1] = precision_score(labelValid, labelPred)
            NPVCenter[c,1] = TN / (TN + FN)
        else:
            scorePred = F.softmax(net.forward(dataValid_add_tdNCD),dim=1).detach().numpy()
            tmpValPred.append(scorePred.copy())
            labelPred = torch.argmax(F.softmax(net.forward(dataValid_add_tdNCD),dim=1),dim=1).data
            accCenter[c,1] = torch.sum(labelPred == labelValid).numpy() / len(labelValid)
            F1Center[c,1] = f1_score(labelValid, labelPred, average='binary' if param['nClassify'] == 2 else 'macro')

        valPred.append(tmpValPred)

    if param['nClassify'] == 2:
        validRes['valPred'] = valPred
        validRes['acc'] = accCenter
        validRes['F1'] = F1Center
        validRes['SEN'] = SENCenter
        validRes['SPE'] = SPECenter
        validRes['PPV'] = PPVCenter
        validRes['NPV'] = NPVCenter
        validRes['TP'] = TP_total
        validRes['TN'] = TN_total
        validRes['FP'] = FP_total
        validRes['FN'] = FN_total
    else:
        validRes['valPred'] = valPred
        validRes['acc'] = accCenter
        validRes['F1'] = F1Center
    np.save(param['dirCache'] + '/validation_result_FCnet5_classify' + str(param['nClassify']) + '.npy', validRes)
    print('acc total: ')
    print(accCenter)
    print('acc mean: (corr): %f, (+tdNCD): %f' % (np.mean(accCenter[:,0]), np.mean(accCenter[:,1])))
    print('F1 total: ')
    print(F1Center)
    print('F1 mean: (corr): %f, (+tdNCD): %f' % (np.mean(F1Center[:,0]), np.mean(F1Center[:,1])))
    