from tensorflow.keras.layers import Bidirectional, Dense, Input, LSTM, Embedding
from tensorflow.keras.models import Sequential
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
import matplotlib.pyplot as plt
import sys
sys.path.append("..")

class buildModel():
    '''
    The model being used here is a modified U-Net. A U-Net consists of an encoder (downsampler) and decoder (upsampler). In-order to learn robust features and reduce the number of trainable parameters, you will use a pretrained model - EfficientNetV2B0 - as the encoder. For the decoder, you will use the upsample block, which is already implemented in the pix2pix example in the TensorFlow Examples repo. (Check out the pix2pix: Image-to-image translation with a conditional GAN tutorial in a notebook.)
    ----------

    Returns
    -------
    self.model:
        Deep learning based Model
    
    '''
    def __init__(self,train_data=None, modelType = 'ml',modelName = 'LogisticRegression'):
        self.X = train_data[0]
        self.y = train_data[1]
        self.model_name = modelName
        self.model_type = modelType
        
        
        
        ##self.base_model, self.layers, self.layer_names
        
    def mlModel(self):
        if self.model_name == 'RandomForestClassifier':
            self.clf = RandomForestClassifier(n_estimators=100)
        elif self.model_name == 'ExtraTreesClassifier':
            self.clf = ExtraTreesClassifier(n_estimators=100)
        elif self.model_name == 'DecisionTreeClassifier':
            self.clf = DecisionTreeClassifier()
        elif self.model_name == 'SupportVectorClassifier':
            self.clf = SVC()
        elif self.model_name == 'KNeighborsClassifier':
            self.clf = KNeighborsClassifier()
        else:
            self.clf = LogisticRegression(solver = 'lbfgs')
            self.model_name == 'LogisticRegression'
    def dlModel(self):
        self.clf = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            384, activation="relu", input_shape=(self.X.shape[1],)
        ),

        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(156, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]

        )

    def defineModel(self):
        if self.model_type =='ml':
            
            self.mlModel()
        elif self.model_type =='dl':
            self.dlModel()
            self.compileModel()
            # self.clf.summary()
                
        
    def compileModel(self):
        '''
        Compile the model
        ----------
        
        Returns
        -------
        
        '''
        
        self.clf.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss='binary_crossentropy',
              metrics=[
                         # tf.keras.metrics.TrueNegatives(name='True_Negatives'),
              # tf.keras.metrics.FalseNegatives(name='False_Negatives'),
              # tf.keras.metrics.TruePositives(name='True_Positives'),
              # tf.keras.metrics.FalsePositives(name='False_Positives'),
              tf.keras.metrics.Precision(name='Precision'),
              tf.keras.metrics.Recall(name='Recall'),
              tfa.metrics.F1Score(num_classes=2, name='F1Score', average="micro")])    
    

    
    def setupModel(self):
        '''
        Build the model
        ----------
        
        Returns
        -------
        
        '''
        self.defineModel()

        return self.clf