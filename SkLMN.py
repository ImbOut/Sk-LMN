import os
import os.path
import sys
from sys import platform
import numpy as np
import pandas as pd
import timeit
from ClassifierBase import ClassifierBase
import TDef
from sklearn.cluster import KMeans
import random
import scipy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn import preprocessing
from tqdm import tqdm
class SkLMN(ClassifierBase):
    #VINH VO
    def map_index(self, index, samples):
        for i in range (0, len(samples)):
            if samples[i] == index:
                return i
        #print("samples = ", samples)
        #print(str(index) + " is at " + str(i))
        return -1
    def calculate_cardinality_of_node(self, x, y, node_indicator):
        x_row = node_indicator[x]
        y_row = node_indicator[y]
        for i in range (1, node_indicator.shape[1]):
            if x_row[i] != y_row[i]:
                return np.sum(node_indicator[:,i-1])
        return 0
    def extract_samples(self, X, indices):
        dataPoints = []
        for i in range (0, len(indices)):
            dataPoints.append(X[indices[i]])
        return dataPoints

    def calculate_m_e(self,x, y, X, tree_list, estimators, estimators_samples):
        cardinality = 0
        for index,i in enumerate(tree_list):
            if self.all_indexces[i][x]==-1:
                self.all_indexces[i][x] = self.map_index(x, estimators_samples[i])
            if self.all_indexces[i][y]==-1:
                self.all_indexces[i][y] = self.map_index(y, estimators_samples[i])
            x2 = self.all_indexces[i][x]
            y2 = self.all_indexces[i][y]

            node_indicator = self.all_indicators[i]
            cardinality += self.calculate_cardinality_of_node( x2, y2, node_indicator)
        if len(tree_list) ==0:
            return float('inf')
        return cardinality/len(tree_list)
    def calculate_m_e_a_tree(self,x, y,i, estimators_sample):
        cardinality = 0
        if self.all_indexces[i][x]==-1:
            self.all_indexces[i][x] = self.map_index(x, estimators_sample)
        if self.all_indexces[i][y]==-1:
            self.all_indexces[i][y] = self.map_index(y, estimators_sample)
        x2 = self.all_indexces[i][x]
        y2 = self.all_indexces[i][y]

        node_indicator = self.all_indicators[i]
        cardinality = self.calculate_cardinality_of_node( x2, y2, node_indicator)
        return cardinality
    def cal_m_e_matrix(self,X, estimators, estimators_samples):
        temp_dict2 = {}
        for i in range(len(estimators)):
            #print("\nProcessing trees",i,":",end=' ')
            estimators_sample = estimators_samples[i]
            step = max( int(len(estimators_sample)/100)-1,1)
            estimator = estimators[i]
            
            for j1 in tqdm(range(len(estimators_sample)-1),"Tree " +str(i)):
                #if (j1 % step == 0):
                    #print(int(j1 / step),end=' ') 

                x = estimators_sample[j1]
                for j2 in range(j1+1,len(estimators_sample)):
                    y = estimators_sample[j2]
                    val = self.calculate_m_e_a_tree(x,y,i, estimators_sample)
                    if x in temp_dict2:
                        temp_dict2[x].append((y,val))
                    else: temp_dict2[x] = [(y,val)]
                    if y in temp_dict2:
                        temp_dict2[y].append((x,val))
                    else: temp_dict2[y] = [(x,val)]
        print("Finish Processing all trees, Calculating distance Matrix")
        #od = collections.OrderedDict(sorted(temp_dict2.items()))
        m_list = [[]  for i in range(len(X))]
        for i,value in temp_dict2.items():
            if i >= self.ax0_max :m_list[i]=[]; continue;
            temp_dict = [0 for i in range(len(X))] 
            temp_dict_count = [0 for i in range(len(X))] 
            for j,val in value:
                if j < self.ax1_min: continue;
                #print(i,j,val)
                temp_dict[j]+= val
                temp_dict_count[j] += 1
            for j in range(len(temp_dict)):
                if j < self.ax1_min: continue;
                if temp_dict_count[j]==0: 
                    temp_dict[j] = 1000000000
                else : 
                    temp_dict[j]/=temp_dict_count[j]
                temp_dict[i]=0
                #m_list.append(temp_dict)
                m_list[i]=temp_dict
            #if (i % 500 == 0):
                #print("Finished Processing Data point ", i)
        for i in range(len(m_list)):
            if(m_list[i]==[]):
                m_list[i] = [1000000000  for i in range(len(X))]
        return m_list

    def cal_m_e_matrix_single(self,X, estimators, estimators_samples):
        temp_dict2 = {}
        for i in range(len(estimators)):
            print("\nProcessing trees",i,":",end=' ')
            estimators_sample = estimators_samples[i]
            step = max( int(len(estimators_sample)/100)-1,1)
            estimator = estimators[i]
            for x in range(len(estimators_sample)-1):
                if (x % step == 0):
                    print(int(x / step),end=' ') 
                for y in range(x+1,len(estimators_sample)):
                    val = self.calculate_cardinality_of_node( x, y, self.all_indicators[i])
                    if x in temp_dict2:
                        temp_dict2[x].append((y,val))
                    else: temp_dict2[x] = [(y,val)]
                    if y in temp_dict2:
                        temp_dict2[y].append((x,val))
                    else: temp_dict2[y] = [(x,val)]
        print("Finish Processing all trees, Calculating distance Matrix")
        m_list = [[]  for i in range(len(X))]
        for i,value in temp_dict2.items():
            if i >= self.ax0_max :m_list[i]=[]; continue;

            temp_dict = [0 for i in range(len(X))] 
            temp_dict_count = [0 for i in range(len(X))] 
            for j,val in value:
                if j < self.ax1_min: continue;
                temp_dict[j]+= val
                temp_dict_count[j] += 1
            for j in range(len(temp_dict)):
                if j < self.ax1_min: continue;
                if temp_dict_count[j]==0: 
                    temp_dict[j] = 1000000000
                else : 
                    temp_dict[j]/=temp_dict_count[j]
                temp_dict[i]=0
                m_list[i]=temp_dict

        for i in range(len(m_list)):
            if(m_list[i]==[]):
                m_list[i] = [1000000000  for i in range(len(X))]
        return m_list


    def cal_m_e_matrix_VINH_VO(self,X, estimators, estimators_samples):
        m_list = []
        for x in range (0, X.shape[0]):
            temp_dict = [] 
            trees_tmp_toan = [i for i in range(len(estimators_samples)) if x in estimators_samples[i] ]
            for y in range (0, X.shape[0]):
                if y != x:
                    #trees = [i for i in range(len(estimators_samples)) if (x in estimators_samples[i] and y in estimators_samples[i])]
                    trees = [i for i in trees_tmp_toan if (y in estimators_samples[i])]
                    m_e = self.calculate_m_e(x, y, X, trees, estimators, estimators_samples)
                    #key_str = "(" + str(x) + "," + str(y) + ")"
                    #temp_dict[key_str] = m_e
                    temp_dict.append(m_e)
                else: temp_dict.append(0)
            m_list.append(temp_dict)
        
            if (x % 10 == 0):
                print("Finished Processing Data point ", x)
        #print(m_list)
        return m_list
    def ComputeAllIndicators(self,clf):
        print("Now! Start MLOS 3")
        self.all_indexces = [[-1 for j in range(len(self.data))] for i in range(len(clf.estimators_))]
        self.all_indicators = []
        for i in range(len(clf.estimators_)):
            samples = self.extract_samples(self.data, clf.estimators_samples_[i])
            node_indicators = clf.estimators_[i].decision_path(samples)
            self.all_indicators.append(node_indicators.toarray())
    def ComputeAllIndicators_Single(self,clf):
        print("Now! Start MLOS 3")
        self.all_indexces = [[-1 for j in range(len(self.data))] for i in range(len(clf.estimators_))]
        self.all_indicators = []
        for i in range(len(clf.estimators_)):
            samples = self.extract_samples(self.data, clf.estimators_samples_[i])
            node_indicators = clf.estimators_[i].decision_path(samples)
            self.all_indicators.append(node_indicators.toarray())
        asd=13
   
    def test(self):
        asd=123
    def Fit(self):
        self.name = 'SkLMN'
        time_start = timeit.default_timer()
        #self.std_scale = preprocessing.StandardScaler().fit(self.X_train)
        #self.X_train = self.std_scale.transform(self.X_train)
        #self.X_test = self.std_scale.transform(self.X_test)
        #self.neigh = KNeighborsClassifier(n_neighbors=TDef.k)
        #self.neigh.fit(self.X_train, self.y_train) 
        GMM =[]
        for i in range(self.n_class):
            l = self.class_labels[i]
            x=self.X_train[np.where( self.y_train == l)[0]]
            gm = GaussianMixture(n_components=1, random_state=0).fit(x)
            GMM.append(gm)
        GM = GMM[0]
        GM.n_components = self.n_class
        for i in range(1,self.n_class):
            GM.means_ = np.append(GM.means_ ,GMM[i].means_, axis=0 )
            GM.covariances_ = np.append(GM.covariances_ ,GMM[i].covariances_, axis=0 )
            GM.precisions_ = np.append(GM.precisions_ ,GMM[i].precisions_, axis=0 )
            GM.precisions_cholesky_ = np.append(GM.precisions_cholesky_ ,GMM[i].precisions_cholesky_, axis=0 )
            GM.lower_bound_ += GMM[i].lower_bound_
            asd=123
        GM.lower_bound_/= self.n_class
        #GM2 = GaussianMixture(n_components=2, random_state=0).fit(self.X_train)
        #a1= GM.predict_proba(self.X_train)
        #a2= GM2.predict_proba(self.X_train)
        #asd=123
        confidents =  GM.predict_proba(self.X_train)
        self.confidents =np.zeros(self.N_train)
        for i in range(self.N_train):
            self.confidents[i] = confidents[i][int(self.y_train[i])]

        asd=123
        


        self.fit_time = timeit.default_timer()-time_start
    def MLOSDistance(self):
        self.ax0_min = 0
        self.ax0_max = self.N_train
        self.ax1_min = self.N_train
        self.ax1_max = self.N
        if TDef.max_samples==-1: TDef.max_samples= self.N
        self.data = np.concatenate((self.X_train,self.X_test), axis=0)
        distances = np.full([self.N, TDef.k], 9e10,dtype=float)
        indexes = np.full([self.N, TDef.k], 9e10,dtype=float)
        clf = IsolationForest(max_samples=TDef.max_samples, random_state=np.random.RandomState(42),n_estimators=TDef.n_estimators)
        print('Building IsolationForest', 'max_samples=',clf.max_samples, 'n_estimators=',clf.n_estimators,'n_neighbors=',TDef.k)
        clf.fit(self.data)
        self.ComputeAllIndicators(clf)
        if TDef.max_samples==len(self.data) and TDef.n_estimators==1:
            m_e_matrix = self.cal_m_e_matrix_single(self.data, clf.estimators_, clf.estimators_samples_)
        else: 
            m_e_matrix = self.cal_m_e_matrix(self.data, clf.estimators_, clf.estimators_samples_)
        
        #m_e_matrix = self.cal_m_e_matrix_VINH_VO(self.data, clf.estimators_, clf.estimators_samples_)
        dist_matrix = np.array(m_e_matrix)[0:self.N_train ,self.N_train:self.N]
        return dist_matrix
        #for i in range(self.N):
        #    dists = m_e_matrix[i]
        #    indexes_ = np.argsort(dists)[1:TDef.k+1]
        #    distances_ = np.array(dists) [indexes_]
        #    for j in range(TDef.k):
        #        distances[i][j] = distances_[j]
        #        indexes[i][j] = indexes_[j]

        #dist_matrixmlos = 
        
    def Evaluate(self):
        time_start = timeit.default_timer()

        
        #############################################################
        dist_matrix = self.MLOSDistance()
        #############################################################3
        #dist_matrix  = scipy.spatial.distance.cdist( self.X_train,self.X_test)
        dist_matrix/= np.max(dist_matrix)
        #D = np.sqrt(np.sum((self.X_test[:, None, :] - self.X_train[None, :, :])**2, axis = -1))
        k_i = np.argpartition(dist_matrix, TDef.k, axis=0)[:TDef.k]
        k_d = np.take_along_axis(dist_matrix, k_i, axis = 0)
        k_i=k_i.transpose()
        k_d=k_d.transpose()
        #self.y_pred = self.neigh.predict(self.X_test)
        #self.y_prob = self.neigh.predict_proba(self.X_test)
        self.y_prob = np.zeros((self.N_test,self.n_class))
        for i in range(self.N_test):
            for j in range(TDef.k):
                jj = int( k_i[i,j] )
                self.y_prob[i][ int( self.y_train[ jj]) ] = (1-k_d[i,j]) *self.confidents[jj]
        sum_of_rows = self.y_prob.sum(axis=1)
        self.y_prob = self.y_prob / sum_of_rows[:, np.newaxis]
        self.y_pred=np.argmax(self.y_prob,1)

        self.eval_time = timeit.default_timer()-time_start

if __name__ == "__main__":
    TDef.InitParameters(sys.argv)
    algo = SkLMN(TDef.data)
    algo.TrainTestSplit(random_state=TDef.split_seed)
    algo.Fit()
    algo.Evaluate()
    algo.CalcScores()
    
