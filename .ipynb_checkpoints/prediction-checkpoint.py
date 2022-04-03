import json
import os


import tensorflow as tf
import joblib
import numpy as np

import pickle


modelType='dl'
def load():
    print("Loading model",os.getpid())
    if modelType == 'ml':
        model = pickle.load(open('./models/finalized_ml_model.pkl', 'rb'))
    else:

        model = tf.keras.models.load_model('./models/finalized_dl_model.h5', compile=False)

    scaler = joblib.load('./models/scaler.pkl')
    
    print("Loaded model")
    return model,scaler



model,scaler= load()
class_name = ['None Fraud', 'Fraud']
print('Models have just loaded!!!!')
def predict(X):
    print ('Step1: Loading models')
    print (X['data'])
    print(type(X['data']))
    print ('Step1 finished!!!!')
    print ('Step2: Scaled the input data.')
    # model_ready_input = scaler.transform([X['data']])
    model_ready_input = scaler.transform([X['data']])
    print(model_ready_input)
    print ('Step2 finished!!!!')
    

    print ('Step3:  Perform prediction!!!')
    if modelType=='dl':
        pred_prob = model.predict(model_ready_input)
        predicted_class=int(np.round(pred_prob))
    else:
        predicted_class = int(model.predict(model_ready_input))
        pred_prob = model.predict_proba(model_ready_input)[:, 1]
        # predicted_class=np.round(result)
    print ('Step3 finished!!!!')
    # print(result)
    print(predicted_class)
    pred_label = class_name[predicted_class]
    print('Predicted Class name: ', pred_label)

    
    # json_results = {"Predicted Class": str(predicted_class)}
    json_results = {"Predicted value": str(predicted_class),"Predicted Class Label": pred_label,"Predicted Class Probability": pred_prob.tolist()}
    print(json_results)
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