import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
from json import dumps,loads
class readData():
    '''
    Read data from csv file
    ----------

    Returns
    -------
    self.data:
        data: as a pandas dataframe
    
    '''
    def __init__(self, dataset_path = None ):

        
        self.dataset_path = dataset_path
        
        self.data = []
        self.testData = None
        self.headers = None
    


    ## Read data from csv File
    def readDataFrame(self):
        '''
        Read data from csv File
        ----------
        
        Returns
        -------
        Dataframe 
        '''
        self.data = pd.read_csv(self.dataset_path)
        os.system('rm -rf ' +self.dataset_path)
        
        
        ### Print how many sample we have
        # print(self.data.head())
        return self.data
    def readTestData(self):
        '''
        Read test data to evaluate the app performance
        ----------
        
        Returns
        -------
        Json file
        '''
        self.headers = {"Content-Type" : "application/json"}
        self.testData = {"data":
          {
                "names":
                    ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
               "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
               "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"],
             "ndarray": [[77627,-7.139060068,2.773081604,-6.757845069,4.446455974,-5.464428185,-1.713401451,-6.485365409,3.409394799,-3.053492714,-6.260705515,2.394167666,-6.16353738,0.602850521,-5.606346429,0.206621734,-6.52508104,-11.40836754,-4.693977736,2.431274492,-0.616949301,1.303250309,-0.016118152,-0.876669888,0.382229801,-1.054623888,-0.614606037,-0.766848112,0.409423944,106.9]]

          }
        }
        self.testData = dumps(self.testData)
        return self.testData,self.headers
    
    