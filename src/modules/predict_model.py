import sys
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 
from ..visualization.visualize import visualizeData
import matplotlib.pyplot as plt

class predictor(object):
    
    def __init__(self, clf=None,data=None,modelType = 'ml'):
        self.clf = clf
        self.data_x = data[0]
        self.data_y = data[1]
        self.model_type = modelType
    



    def predict(self):
        if self.model_type == 'ml':
                 y_pred = self.clf.predict(self.data_x.values)
                 cnf_matrix = confusion_matrix(self.data_y,y_pred)
                 visualizeData(cm_data=cnf_matrix).confusionMatrixPlot()
#                  # Compute confusion matrix
#                  cnf_matrix = confusion_matrix(y_test,y_pred)
#                  np.set_printoptions(precision=2)

#                  print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

#                     # Plot non-normalized confusion matrix
#                  class_names = [0,1]
#                  plt.figure()
#                  plot_confusion_matrix(cnf_matrix
#                                           , classes=class_names
#                                           , title='Confusion matrix')
#                  plt.show()
                 
        else:
                
                 predict_x=self.clf.predict(self.data_x.values) 
                 y_pred=np.argmax(predict_x,axis=1)
                 cnf_matrix = confusion_matrix(self.data_y,y_pred)
                 visualizeData(cm_data=cnf_matrix).confusionMatrixPlot()