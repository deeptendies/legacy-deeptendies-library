import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import shutil	
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers.experimental.preprocessing import Normalization


class Trainer():
    """Trainer for model. 
    """
    def __init__(self, save_location, model, model_name, train_data, test_data):
        self.dir_location = save_location
        self.model = model
        self.model_name = model_name
        self.train_data = train_data
        self.test_data = test_data
        self.hist = None

    def train_model(self, epochs, verbose, callbacks ):
        print("TRAINING")
        self.hist = self.model.fit(self.train_data, steps_per_epoch=len(self.train_data), epochs=epochs, verbose=verbose, callbacks=callbacks)
    def graph_loss(self):
        loss = self.hist.history['loss']
        fig = plt.plot(self.hist.epoch, loss)
        plt.title("Loss vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        return fig 

    def get_predictions(self):
        self.predictions = {'actual': [], "predicted": []}
        idx= 0 
        for i in self.test_data: 
          self.predictions['actual'].append(self.test_data.unnormalize_target(i[1], idx=idx)[0])
          self.predictions['predicted'].append(self.test_data.unnormalize_target(self.model.predict(i[0])[0], idx=idx)[0])
          idx = idx + 1
        return self.predictions
        
    def save_model_collab(self):
        """INTENDED FOR COLLAB ONLY"""
        drive.mount('/content/drive')
        self.model.save(self.model_name) 
        os.listdir()
        for f in os.listdir(): 
            if (len(f.split(".")) == 1): 
                continue
            if f.split(".")[1] =="h5":
                shutil.copyfile(f, self.dir_location + f)

    def save_model_local(self): 
        pass 
