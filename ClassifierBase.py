import os
import os.path
import sys
from sys import platform
import numpy as np
import timeit
import TDef
import csv
from csv import writer
import random
from sklearn.utils import check_random_state
import pandas as pd
from sklearn.model_selection import train_test_split
from TDef import *
from sklearn.metrics import *
def GetDatasetFolder():
    db_path = "data\\"
    if platform == "linux" or platform == "linux2":
        db_path = 'data/'
    return db_path

class ClassifierBase:
    def __init__(self,dbname='movielens_rating.csv',random_state=None):
        self.dbname = dbname
        filename = GetDatasetFolder() + dbname
        self.data_pd = pd.read_csv(filename);
        self.N = len(self.data_pd)
        self.D = len(self.data_pd.columns)
        self.dicts = [];self.dicts2 = []
        x = self.data_pd.to_numpy()
       
        self.X = x[:,0:self.D-1]
        self.y = x[:,self.D-1]
        
        self.class_labels =np.unique(self.y) 
        self.n_class = len(self.class_labels)
        asd=123
    
    def TrainTestSplit(self,random_state=TDef.split_seed):
        self.X_train, self.X_test, self.y_train, self.y_test =train_test_split(self.X, self.y, test_size=TDef.test_ratio, random_state=random_state)
        self.y_train =(self.y_train.astype(int))
        self.y_test =(self.y_test.astype(int))
        self.N_train = len(self.X_train)
        self.N_test = len(self.X_test)
        print('Spliting data: Training size:',self.N_train,'Testing size:' , self.N_test, "test_ratio:",TDef.test_ratio, "Split_seed",TDef.split_seed )
    
    def CalcScores(self):
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        probs = self.y_prob
        if self.n_class==2: 
            probs = probs[:, 1]
            self.roc_auc = roc_auc_score(self.y_test,probs )
            precision, recall, thresholds = precision_recall_curve(self.y_test, probs)
            self.pr_auc = auc(recall, precision)
        else: 
            self.pr_auc = -2
            self.roc_auc = roc_auc_score(self.y_test,probs,multi_class='ovr' )
        self.confusion_matrix = confusion_matrix(self.y_test, self.y_pred)
        self.cohen_kappa_score = cohen_kappa_score(self.y_test, self.y_pred)
        self.jaccard_score = jaccard_score(self.y_test, self.y_pred)
        self.matthews_corrcoef = matthews_corrcoef(self.y_test, self.y_pred)
        self.zero_one_loss = zero_one_loss(self.y_test, self.y_pred)
        self.f1_score = f1_score(self.y_test, self.y_pred)
        self.fbeta_score2 = fbeta_score(self.y_test, self.y_pred,beta=2)
        self.fbeta_score05 = fbeta_score(self.y_test, self.y_pred,beta=0.5)

        self.precision_recall_fscore_support = precision_recall_fscore_support(self.y_test, self.y_pred)
        self.precision_score = precision_score(self.y_test, self.y_pred)
        self.recall_score = recall_score(self.y_test, self.y_pred)
        self.balanced_accuracy_score = balanced_accuracy_score(self.y_test, self.y_pred)
        self.classification_report = classification_report(self.y_test, self.y_pred)
        self.hamming_loss = hamming_loss(self.y_test, self.y_pred)
        self.log_loss = log_loss(self.y_test, self.y_pred)
        self.hinge_loss = hinge_loss(self.y_test, probs)
        self.brier_score_loss = brier_score_loss(self.y_test, probs)
        
        print("\nMethod:", self.name, " data:", TDef.data ," accuracy:", "%.3f" % self.accuracy , "roc_auc:","%.3f" % self.roc_auc,"pr_auc:","%.3f" % self.pr_auc)
        self.WriteResultToCSV()
    
    def append_list_as_row(self,file_name, list_of_elem):
        with open(file_name, 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(list_of_elem)
    
    def WriteResultToCSV(self,file=''):
        if not os.path.exists(TDef.folder):
            os.makedirs(TDef.folder)
        if file=='':
            file = TDef.folder+ '/' + self.name + TDef.fname + ".csv" 
        self.dbname = self.dbname.replace("_c","").replace(".csv","").capitalize()
        self.dicts.append(('dbname',self.dbname ))
        self.dicts.append(('N',self.N ))
        self.dicts.append(('D',self.D ))
        self.dicts.append(('test_ratio',TDef.test_ratio ))
        self.dicts.append(('split_seed',TDef.split_seed ))

        self.dicts.append(('accuracy',self.accuracy ))
        self.dicts.append(('roc_auc',self.roc_auc ))
        self.dicts.append(('pr_auc',self.pr_auc ))

        self.dicts.append(('confusion_matrix',self.confusion_matrix ))
        self.dicts.append(('cohen_kappa_score',self.cohen_kappa_score ))
        self.dicts.append(('jaccard_score',self.jaccard_score ))
        self.dicts.append(('matthews_corrcoef',self.matthews_corrcoef ))
        self.dicts.append(('zero_one_loss',self.zero_one_loss ))
        self.dicts.append(('f1_score',self.f1_score ))
        self.dicts.append(('fbeta_score2',self.fbeta_score2 ))
        self.dicts.append(('fbeta_score05',self.fbeta_score05 ))
        self.dicts.append(('precision_recall_fscore_support',self.precision_recall_fscore_support ))
        self.dicts.append(('precision_score',self.precision_score ))
        self.dicts.append(('recall_score',self.recall_score ))
        self.dicts.append(('balanced_accuracy_score',self.balanced_accuracy_score ))
        self.dicts.append(('classification_report',self.classification_report ))
        self.dicts.append(('hamming_loss',self.hamming_loss ))
        self.dicts.append(('log_loss',self.log_loss ))
        self.dicts.append(('hinge_loss',self.hinge_loss ))
        self.dicts.append(('brier_score_loss',self.brier_score_loss ))

        self.dicts.append(('fit_time',self.fit_time ))
        self.dicts.append(('eval_time',self.eval_time ))

        dicts = self.dicts+self.dicts2;
        try:
            if os.path.isfile(file)==False:
                colnames = [i[0] for i in dicts]
                self.append_list_as_row(file,colnames)
            vals = [i[1] for i in dicts]
            self.append_list_as_row(file,vals)
        except Exception  as ex:
            print('Cannot write to file ', file ,'', ex);
            self.WriteResultToCSV(file + str(random.randint(0,1000000)) + '.csv')
    

