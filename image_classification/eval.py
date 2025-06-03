import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
def evaluate_model(model, val_gen):
    results = model.evaluate(val_gen)
    print("\nEvaluation Results:", results)
    return results
