


import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from ..visualization.visualize import visualizeData
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

class preprocessData():
    '''
    Turn raw data into features for modeling
    ----------

    Returns
    -------
    self.final_set:
        Features for modeling purpose
    self.labels:
        Output labels of the features
    enc: 
        Ordinal Encoder definition file
    ohe:
        One hot  Encoder definition file
    '''
    def __init__(self, data = None):
        self.data = data
        
        

        
#         self.final_set,self.labels = self.build_data()
    def printBasicInfo(self):
        print ("**"*30)
        print ("**"*10+'   data shape  '+"**"*10)
        print ("**"*30)
        print(self.data.shape)
        print ("**"*30)
        print ("**"*10+'   data null info   '+"**"*10)
        print ("**"*30)
        print(self.data.isnull().sum())
        print ("**"*30)
        print ("**"*10+'   data info   '+"**"*10)
        print ("**"*30)
        print (self.data.info())
        print ("**"*30)
        print ("**"*10+'   data head   '+"**"*10)
        print ("**"*30)
        print (self.data.head())
        
        
    def plot_distribution(self):
        print ("**"*30)
        print ("**"*10+'   Plot distribution   '+"**"*10)
        print ("**"*30)
        var = self.data.columns.values

        i = 0
        t0 = self.data.loc[self.data['Class'] == 0]
        t1 = self.data.loc[self.data['Class'] == 1]

        sns.set_style('whitegrid')
        plt.figure()
        fig, ax = plt.subplots(8,4,figsize=(16,28))

        for feature in var:
            i += 1
            plt.subplot(8,4,i)
            sns.kdeplot(t0[feature], bw=0.5,label="Class = 0")
            sns.kdeplot(t1[feature], bw=0.5,label="Class = 1")
            plt.xlabel(feature, fontsize=12)
            locs, labels = plt.xticks()
            plt.tick_params(axis='both', which='major', labelsize=12)
        plt.show()
        
    def plotCorrelation(self):
        print ("**"*30)
        print ("**"*10+'   Plot Correlation among data   '+"**"*10)
        print ("**"*30)
        plt.figure(figsize = (16,10))
        plt.title('Credit Card Transactions features correlation plot')
        corr = self.data.corr()
        sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Greens",fmt='.1f',annot=True)
        plt.show()

        
        
        
    def plotFrequency(self):
        print ("**"*30)
        print ("**"*10+' Frequency of the target classes '+"**"*10)
        print ("**"*30)
        self.data["Class"].value_counts().plot(kind="bar",color="red")
        plt.title("Frequency of the target classes", size=20)
        plt.xlabel("Target Labels", size = 18)    
        print ("Below is the exact frequency values for both the target labels.")
        self.target = pd.DataFrame(self.data["Class"].value_counts())
        self.target.style.background_gradient(cmap="Reds")
        
        
    def dataScaling(self):
        
        
        self.X=self.data.drop(columns=["Class"])
        self.y=self.data["Class"]  
        self.names=self.X.columns
        self.scaled_df = preprocessing.scale(self.X)
        self.scaled_df = pd.DataFrame(self.scaled_df,columns=self.names)
        print ("**"*30)
        print ("**"*10+'   Scaled data head   '+"**"*10)
        print ("**"*30)
        print (self.scaled_df.head())
        print ("**"*30)
        print ("**"*10+'   Scaled data describe   '+"**"*10)
        print ("**"*30)
        print (self.scaled_df[["Amount","Time"]].describe())
        
    def splitingData(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.scaled_df, self.y, test_size = 0.20, random_state = 0, shuffle = True, stratify = self.y)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size = 0.25, random_state = 0, shuffle = True, stratify = self.y_train)
        
        
        
        print ("**"*30)
        print ("**"*10+'   Shape of train and test data  '+"**"*10)
        print ("**"*30)
        print (self.X_train.shape, self.X_test.shape)
        print ("**"*30)
        print ("**"*10+'   y_train value counts   '+"**"*10)
        print ("**"*30)
        print (self.y_train.value_counts())
        print ("**"*30)
        print ("**"*10+'   y_test value counts  '+"**"*10)
        print ("**"*30)
        print (self.y_test.value_counts())
                
    def randomSampling(self):
        sm = SMOTE(sampling_strategy='minority',random_state = 33)
        self.X_train_new, self.y_train_new = sm.fit_resample(self.X_train, self.y_train.ravel())
        print ("**"*30)
        print ("**"*10+'   y_train_new value counts bar chart '+"**"*10)
        print ("**"*30)
        plt.figure()
        pd.Series(self.y_train_new).value_counts().plot(kind="bar")
        
        
        
        
        
    def dataPreProcessing(self):
        self.printBasicInfo()
        self.plot_distribution()
        self.plotCorrelation()
        self.plotFrequency()
        self.dataScaling()
        self.splitingData()
        self.randomSampling()
        self.train = (self.X_train, self.y_train)
        self.test = (self.X_test,self.y_test)
        self.val = (self.X_val,self.y_val)
        self.train_new = (self.X_train_new, self.y_train_new)
        return self.train, self.test, self.val, self.train_new
        
        
    
    
    