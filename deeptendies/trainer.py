""" Utility Method to train model """
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import shutil	
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers.experimental.preprocessing import Normalization

class Trainer():
	dir_location = "./drive/MyDrive/enel645-team-drive/assignment-2/saved_models/"
	def __init__(self, model_name:str, model, X_data=None, Y_data=None):
		self.model_name = model_name
		self.model = model()
		self.X_data = X_data
		self.Y_data = Y_data

	def split_data(self, train_size:float):
		print("SPLITING DATA SET")
		self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_data, self.Y_data, train_size=train_size)
		print(np.shape(self.X_train[0]))

	def train_model(self, batch_size, epoch, verbose, callbacks):
		print("TRAINING")
		self.model.fit(self.X_train, self.Y_train, batch_size, epochs, verbose, callbacks, validation_data=(self.X_val, self.Y_val))

	def save_model(self):
		self.model.save(self.model_name)

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

		os.listdir()
		for f in os.listdir(): 
			if (len(f.split(".")) == 1): 
				continue
			if f.split(".")[1] =="h5":
				shutil.copyfile(f, dir_location + f)

	def main(self):
		self.split_data(train_size=0.8)
		# self.train_model()
		# self.save_model()



if __name__ == "__main__":

	(X_dev, Y_dev), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()

	def get_scaled_feature_tensor(x, norm_type=0):
	  """Function to get scaled feature matrix. 
	    Use norm_type=0 for min,max scaler and norm_type==1 for standard scaler
	    TODO: Wrap into model for end-to-end solution 
	    TODO: Experiment with normalization over different axis -> eg., Normalization(axis=1) gives normalization by row, axis=2 normalization by column, and axis = 3 normalization by channel. 
	  """ 
	  if norm_type == 0: 
	    scaled_tensor = Rescaling(scale=1.0/255)(x)
	  elif norm_type == 1: 
	    norm_layer = Normalization(axis=None)
	    norm_layer.adapt(x)
	    scaled_tensor = norm_layer(x)
	  return scaled_tensor

	# define model
	def model1(input_shape=(32,32,3), k=10, lr=1e-4, norm_type=0):
		model_input = tf.keras.layers.Input(shape=input_shape)
		norm_input = get_scaled_feature_tensor(model_input, norm_type)
		model_input_flatten = tf.keras.layers.Flatten()(norm_input)
		hidden1 = tf.keras.layers.Dense(64,activation='relu')(model_input_flatten)
		hidden2 = tf.keras.layers.Dense(1024,activation='relu')(hidden1) 
		out = tf.keras.layers.Dense(k, activation='softmax')(hidden2)
		model = tf.keras.models.Model(inputs = model_input, outputs =out)
		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
		return model 

	model_name_fcn_ = "model1.h5"
	early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
	monitor = tf.keras.callbacks.ModelCheckpoint(
	    model_name_fcn_, monitor='val_loss',
	    verbose=0,save_best_only=True,
	    save_weights_only=True,
	    mode='min')

	# Learning rate schedule
	def scheduler(epoch, lr):
	    if epoch%100 == 0:
	        lr = lr/2
	    return lr

	lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose = 0)

	arnold = Trainer("FirstModel", model=model1, X_data=X_dev, Y_data=Y_dev)
	# print(arnold.model)
	# arnold.split_data(train_size=0.8)
	# arnold.train_model(batch_size=32, epochs=100, verbose=1, callbacks=[early_stop, monitor, lr_schedule])



	# arnold.main()