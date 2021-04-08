import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import shutil	
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from google.colab import drive


class Trainer():
    """Trainer for model. 
    """
    def __init__(self, save_location, model, model_name):
        self.dir_location = save_location
        self.model = model
        self.model_name = model_name

    def split_data(self, X_dev, Y_dev, train_size):
        print("SPLITING DATA SET")
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(X_dev, Y_dev, train_size=train_size)

    def train_model(self, batch_size, epochs, verbose, callbacks):
        print("TRAINING")
        self.model.fit(self.X_train, self.Y_train, batch_size, epochs, verbose, callbacks, validation_data=(self.X_val, self.Y_val))
        self.graph_results()

    def graph_results(self):
        history = self.model.history
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        fig, ax = plt.subplots(figsize=(18,12))
        train_ax = ax.scatter(x=range(len(train_acc)), y=train_acc, label="Training Accuracy")
        val_ax = ax.scatter(x=range(len(train_acc)), y=val_acc, label="Validation Accuracy")
        legend = ax.legend()
        fig.suptitle("Min/Max Normalized FCNN Accuracy vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        fig.show()

    def save_model(self):
        drive.mount('/content/drive')
        self.model.save(self.model_name) 
        os.listdir()
        for f in os.listdir(): 
            if (len(f.split(".")) == 1): 
                continue
            if f.split(".")[1] =="h5":
                shutil.copyfile(f, self.dir_location + f)