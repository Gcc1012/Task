Image classification using MobileNetV2 using custom callback 
It contains following file:
-->main.py
-->image_classification 
    -->load_data.py 
    -->model.py 
    -->callback.py 
    -->train.py 
    -->eval.py 
    -->utils.py 

 Requirements:

 - Python 3.x
- TensorFlow 2.x
- Keras
- NumPy

Custom Callback:
   Dynamic learning rate reduction after a set number of stagnant epochs

  Early stopping if no improvement continues after several reductions

  Weight restoration  to best-performing state after LR drop

  User interaction after a configurable number of epochs 

Parameters:
patience: Epochs to wait before reducing LR (default: 2)

stop_patience: Max LR reductions before stopping training (default: 3)

threshold: Minimum accuracy before LR change is considered (e.g. 0.85)

factor: LR multiplier 

dwell: Restore best weights after LR change if True

ask_epoch: Prompt user to continue training after this epoch

model_name: Name used in logs
