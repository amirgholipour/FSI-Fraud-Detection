import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,PrecisionRecallDisplay


class visualizeData():
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
    def __init__(self, cm_data = None, y_true = None, y_pred = None,modelType = 'ML'):
        self.cm= cm_data
        self.modelType = modelType
        self.y_pred = y_pred
        self.y_true = y_true
    def confusionMatrixPlot(self):
        
        plt.figure(figsize=(8,6))
        sns.set(font_scale=1.2)
        sns.heatmap(self.cm, annot=True, fmt = 'g', cmap="Reds", cbar = False)
        plt.xlabel("Predicted Label", size = 18)
        plt.ylabel("True Label", size = 18)
        plt.title("Confusion Matrix Plotting for "+ self.modelType +"  model", size = 20)
        
        self.current_path= os.getcwd()
        self.inference_path = self.current_path.replace('notebooks','reports')+'/figures/'
        
        
        plt.savefig(self.inference_path+'confusionMatrixPlot'+'_'+self.modelType+'.png')
        plt.show()
        
    def precisionRecallDisplay(self):
        PrecisionRecallDisplay.from_predictions(self.y_true, self.y_pred)
        
        self.current_path= os.getcwd()
        self.inference_path = self.current_path.replace('notebooks','reports')+'/figures/'
        
        plt.savefig(self.inference_path+'PrecisionRecallDisplay'+'_'+self.modelType+'.png')
        plt.show()


        
        
#     def plotConfusionMatrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#             """
#             This function prints and plots the confusion matrix.
#             Normalization can be applied by setting `normalize=True`.
#             """
#             plt.imshow(cm, interpolation='nearest', cmap=cmap)
#             plt.title(title)
#             plt.colorbar()
#             tick_marks = np.arange(len(classes))
#             plt.xticks(tick_marks, classes, rotation=0)
#             plt.yticks(tick_marks, classes)

#             if normalize:
#                 cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#                 #print("Normalized confusion matrix")
#             else:
#                 1#print('Confusion matrix, without normalization')

#             #print(cm)

#             thresh = cm.max() / 2.
#             for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#                 plt.text(j, i, cm[i, j],
#                          horizontalalignment="center",
#                          color="white" if cm[i, j] > thresh else "black")

#             plt.tight_layout()
#             plt.ylabel('True label')
#             plt.xlabel('Predicted label')