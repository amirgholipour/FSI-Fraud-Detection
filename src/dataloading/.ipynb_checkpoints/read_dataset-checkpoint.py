import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd

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
        
        
        ### Print how many sample we have
        # print(self.data.head())
        return self.data