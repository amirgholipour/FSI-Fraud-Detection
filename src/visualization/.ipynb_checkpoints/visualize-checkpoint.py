import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output


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
    def __init__(self, cm_data = None, modelName = 'Logistic Regression'):
        self.cm= cm_data
        self.model_name = modelName
    def confusionMatrixPlot(self):
        plt.figure(figsize=(8,6))
        sns.set(font_scale=1.2)
        sns.heatmap(self.cm, annot=True, fmt = 'g', cmap="Reds", cbar = False)
        plt.xlabel("Predicted Label", size = 18)
        plt.ylabel("True Label", size = 18)
        plt.title("Confusion Matrix Plotting for "+ self.model_name +"  model", size = 20)
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