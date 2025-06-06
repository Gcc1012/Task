from image_classification.load_data import load_data
from image_classification.model import build_model
from image_classification.callback import LRA
from image_classification.train import train_model
from image_classification.eval import evaluate_model
from image_classification.utils import save_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import warnings
warnings.filterwarnings('ignore')

# def main():
#     data_dir = 'data'
#     batch_size = 32
#     epochs = 50
#     ask_epoch = 10
#     freeze = True

#     train_gen, val_gen = load_data(data_dir, batch_size=batch_size)
#     model = build_model(num_classes=train_gen.num_classes, freeze=freeze)

#     lra_callback = LRA(patience=2, stop_patience=3, threshold=0.85, factor=0.5, dwell=True, ask_epoch=ask_epoch, model_name='MobileNetV2')
#     history = train_model(model, train_gen, val_gen, callbacks=[lra_callback], epochs=epochs)
#     print("history---->",history)

#     evaluate_model(model,val_gen)





def main():
    data_dir = r"C:\Users\Gayatri\Documents\Modular Image Classification\task_try\data"
    batch_size = 32
    epochs = 50
    ask_epoch = 10
    freeze = True

    train_gen, val_gen = load_data(data_dir, batch_size=batch_size)
    model = build_model(num_classes=train_gen.num_classes, freeze=freeze)

    lra_callback = LRA(patience=2, stop_patience=3, threshold=0.8, factor=0.5, dwell=True, ask_epoch=ask_epoch, model_name='MobileNetV2')
    history = train_model(model, train_gen, val_gen, callbacks=[lra_callback], epochs=epochs)
    print("history---->", history)

    evaluate_model(model, val_gen)

    save_model(model, 'mobilenetv2_best_model.h5')



if __name__ == '__main__':
    main()
