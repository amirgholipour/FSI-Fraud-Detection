import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import matplotlib.pyplot as plt
import sys
import shutil
sys.path.append("..")
import os
class trainModel():
    '''
    The model being used here is a modified U-Net. A U-Net consists of an encoder (downsampler) and decoder (upsampler). In-order to learn robust features and reduce the number of trainable parameters, you will use a pretrained model - EfficientNetV2B0 - as the encoder. For the decoder, you will use the upsample block, which is already implemented in the pix2pix example in the TensorFlow Examples repo. (Check out the pix2pix: Image-to-image translation with a conditional GAN tutorial in a notebook.)
    ----------

    Returns
    -------
    self.model:
        Deep learning based Model
    
    '''
    def __init__(self,model = None,train_data= None, val_data = None, modelType = 'ml',epochs = 150, savePath = None):
        
        self.clf = model
        
        self.X = train_data[0]
        self.y = train_data[1]
        self.val_data = val_data
        self.class_weight = {0: .2, 1: .8}
        
        
        
        if self.val_data != None:
            self.val_X = val_data[0]
            self.val_y = val_data[1]
        self.model_type = modelType
        self.save_path = savePath
        self.save_path_workshop = self.save_path.replace('Inference','Workshop').replace('deploy','models')

        if self.model_type =='dl':
            self.early_stop = EarlyStopping(monitor='val_F1Score', mode='max', verbose=1, patience = 20)

            self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.save_path,
                save_weights_only=False,
                monitor='val_F1Score',
                mode='max',
                save_best_only=True)
            self.epochs = epochs
        
        
        
        ##self.base_model, self.layers, self.layer_names
        
    def mlModelTraining(self):
        self.clf.fit(self.X, self.y )
        self.saveMlModel()
        
        
    def dlModelTraining(self):
        
        self.clf.fit(x=self.X, y=self.y, batch_size = 2048, epochs=self.epochs,
          validation_data=(self.val_X , self.val_y), verbose=1,
          callbacks=[self.early_stop,self.model_checkpoint_callback ],class_weight = self.class_weight)#,shuffle=True
        self.saveDlModel()
                

    def saveMlModel(self):
        # save the model to disk
        self.save_path_workshop = self.save_path.replace('Inference','Workshop')
        joblib.dump(self.clf, self.save_path)
        shutil.copy(self.save_path, os.getcwd().replace('notebooks','deploy')) 
    def saveDlModel(self):
        # save the model to disk
        self.save_path_workshop = self.save_path.replace('Inference','Workshop')
        
        shutil.copy(self.save_path, os.getcwd().replace('notebooks','deploy')) 
#         self.clf.save(self.save_path)
#         self.clf.save(self.save_path_workshop)

    
    def modelTraining(self):
        '''
        Train the model
        ----------
        
        Returns
        -------
        
        '''
        if self.model_type=='ml':
            self.mlModelTraining()
        else:
            self.dlModelTraining()

        return self.clf