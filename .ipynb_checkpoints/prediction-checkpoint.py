# Imports
import glob
import json
import os
from os.path import splitext,basename
import uuid
import base64

import tensorflow as tf
import joblib
import numpy as np
import json
import traceback
import sys
import os
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
modelType='ml'
def load():
    print("Loading model",os.getpid())
    if modelType == 'ml':
        model = pickle.load(open('./models/finalized_ml_model.pkl', 'rb'))
    else:

        model = tf.keras.models.load_model('./models/finalized_dl_model.h5', compile=False)

    scaler = joblib.load('./models/scaler.pkl')
    
    print("Loaded model")
    return model,scaler



model= load()
print('Models have just loaded!!!!')
def predict(X):
    
    print ('Step1: Loading models')
    print (X['data'])
    print ('Step1 finished!!!!')
    print ('Step2: tokenise the input data.')
    model_ready_input = scaler.transform(X['data'])
    print(model_ready_input)
    print ('Step2 finished!!!!')
    

    print ('Step3:  Do prediction!!!')
    if modelType=='dl':
        result = model.predict(model_ready_input)
        predicted_class=np.round(result)
    else:
        predicted_class = model.predict(model_ready_input)
    print ('Step3 finished!!!!')
                            
    print('Predicted Class name: ', predicted_class)

    
    json_results = {"Predicted Class": str(predicted_class)}
    # json_results = {"Predicted Class": str(predicted_class),"Predicted Class Label": pred_label.tolist(), "Predicted Certainty Score":predicted_class_prob}
    # print(json_results)
    return json_results
    

class JsonSerializer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (
        np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

