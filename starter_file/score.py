import json
import numpy as np
import os
import joblib
import logging
from azureml.core.model import Model

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    
    model_path = Model.get_model_path('capstone_best_hyperdrive_model')
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)
    logging.info("Init complete")


def run(raw_data):
    data = json.loads(raw_data)['data']
    data = np.array(data)
    result = model.predict(data)

    return json.dumps({"forecast": result.tolist()})
