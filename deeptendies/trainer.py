""" Utility Method to train model """
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
  def __init__(self, save_location, model, model_name, train_data, test_data):
    self.dir_location = save_location
    self.model = model
    self.model_name = model_name
    self.train_data = train_data
    self.test_data = test_data

  def train_model(self, epochs, verbose, callbacks ):
    print("TRAINING")
    self.model.fit(self.train_data, steps_per_epoch=len(train_data), epochs=epochs, verbose=verbose, callbacks=callbacks)
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
