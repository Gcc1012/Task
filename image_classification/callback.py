import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
import tensorflow as tf
import numpy as np

class LRA(tf.keras.callbacks.Callback):
    def __init__(self, patience, stop_patience, threshold, factor, dwell, ask_epoch, model_name):
        super().__init__()
        self.patience = patience
        self.stop_patience = stop_patience
        self.threshold = threshold
        self.factor = factor
        self.dwell = dwell
        self.ask_epoch = ask_epoch
        self.model_name = model_name
        self.best_weights = None
        self.wait = 0   #epochs where validation loss didn’t improve
        self.stop_wait = 0      #it tracks how many times the learning rate was reduced without improvement.
        self.best_val_loss = np.Inf
        self.epoch_counter = 0

    # def on_epoch_end(self, epoch, logs=None):
    #     val_loss = logs['val_loss']
    #     acc = logs['accuracy']
    #     self.epoch_counter += 1

    #     if val_loss < self.best_val_loss:
    #         self.best_val_loss = val_loss
    #         self.best_weights = self.model.get_weights()
    #         self.wait = 0
    #         self.stop_wait = 0
    #     else:
    #         if acc >= self.threshold:
    #             self.wait += 1
    #             if self.wait >= self.patience:
    #                 # old_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
    #                 old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
    #                 new_lr = old_lr * self.factor
    #                 # tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
    #                 tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
    #                 print(f"\nEpoch {epoch+1}: Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}.")
    #                 self.wait = 0
    #                 self.stop_wait += 1

    #                 if self.dwell:
    #                     self.model.set_weights(self.best_weights)

    #                 if self.stop_wait >= self.stop_patience:
    #                     print("\nEarly stopping due to no improvement after learning rate reductions.")
    #                     self.model.stop_training = True

    #     if self.epoch_counter >= self.ask_epoch:
    #         ans = input("\nContinue training? (y/n): ")
    #         if ans.lower() != 'y':
    #             self.model.stop_training = True



    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs['val_loss']
        acc = logs['accuracy']
        self.epoch_counter += 1
        print()

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_weights = self.model.get_weights()
            self.wait = 0
            self.stop_wait = 0
        else:
            if acc >= self.threshold:
                self.wait += 1
                if self.wait >= self.patience:
                    lr = self.model.optimizer.learning_rate
                    
                    if isinstance(lr, tf.Variable):
                        old_lr = float(tf.keras.backend.get_value(lr))
                        new_lr = old_lr * self.factor
                        tf.keras.backend.set_value(lr, new_lr)
                        print(f"\nEpoch {epoch+1}: Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}.")
                    else:
                        
                        old_lr = lr if isinstance(lr, (float, int)) else 'unknown'
                        new_lr = old_lr * self.factor if isinstance(old_lr, (float, int)) else None
                        if new_lr is not None:
                            self.model.optimizer.learning_rate = new_lr
                            print(f"\nEpoch {epoch+1}: Reducing learning rate from {old_lr} to {new_lr}.")
                        else:
                            print("\nCannot update learning rate automatically (not a variable or float).")

                    self.wait = 0
                    self.stop_wait += 1

                    if self.dwell:
                        self.model.set_weights(self.best_weights)

                    if self.stop_wait >= self.stop_patience:
                        print("\nEarly stopping due to no improvement after learning rate reductions.")
                        self.model.stop_training = True

        if self.epoch_counter >= self.ask_epoch:
            ans = input("\nContinue training? (y/n): ")
            if ans.lower() != 'y':
                self.model.stop_training = True
