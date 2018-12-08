
import pandas as pd
import numpy as np
import time
import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Dropout
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.models import load_model

class CompleteCode:
	def __init__(self):
		pass
	def preprocess_data(self,csvfile):
		"""Load csv file from the current directory
		    Should provide the fullname of file.
		    For example if test data is in file test.csv then argument should be passed 
		    as test.csv
		    Returns: Pandas dataframe  """
		data = pd.read_csv(csvfile)
		success_fail_data = data.loc[(data['state']=='successful')|(data['state']=='failed')]
		# ID, name might not have huge impact in the success of the project
		filtered_data = success_fail_data.filter(items = ['main_category','currency','state','backers',\
				              'country','usd_pledged_real','usd_goal_real'])
		Main_category = {'Art':1,'Comics':2,'Crafts':3,'Dance':4,'Design':5,'Fashion':6,'Film & Video':7,\
		     'Food':8,'Games':9,'Journalism':10,'Music':11,'Photography':12,'Publishing':13,'Technology':14\
		    ,'Theater':15}
		Currency = {'AUD':1,'CAD':2,'CHF':3,'DKK':4,'EUR':5,'GBP':6,'HKD':7,'JPY':8,'MXN':9,'NOK':10,'NZD':11,'SEK':12\
		,'SGD':13,'USD':14}
		Country = {'AT':1,'AU':2,'BE':3,'CA':4,'CH':5,'DE':6,'DK':7,'ES':8,'FR':9,'GB':10,'HK':11,'IE':12,'IT':13\
		,'JP':14,'LU':15,'MX':16,'N,0"':17,'NL':18,'NO':19,'NZ':20,'SE':21,'SG':22,'US':23}
		State = {'failed':0,'successful':1}
		#Normalization of the data
		filtered_data.state = [State[item] for item in filtered_data.state]
		filtered_data.main_category = [Main_category[item]/15 for item in filtered_data.main_category]
		filtered_data.currency = [Currency[item]/14 for item in filtered_data.currency]
		filtered_data.country = [Country[item]/23 for item in filtered_data.country]
		max_backers = filtered_data.backers.max()
		max_usd_pledged_real = filtered_data.usd_pledged_real.max()
		max_usd_goal_real = filtered_data.usd_goal_real.max()
		filtered_data.backers = [item/max_backers for item in filtered_data.backers]
		filtered_data.usd_pledged_real = [item/max_usd_pledged_real for item in filtered_data.usd_pledged_real]
		filtered_data.usd_goal_real = [item/max_usd_goal_real for item in filtered_data.usd_goal_real]
		return filtered_data
	
	def get_test_data(self,csvfile):
		"""Load csv file from the current directory
		    Should provide the fullname of file.
		    For example if test data is in file test.csv then argument should be passed 
		    as test.csv
		    Returns: Numpy ndarray of Test inputs and Test labels  """
		
		filtered_data = self.preprocess_data(csvfile)
		test_inputs = filtered_data.filter(items = ['main_category','currency','backers',\
				                  'country','usd_pledged_real','usd_goal_real'])
		test_labels = filtered_data.filter(items=['state'])
		# Convert Pandas dataframe to Numpy ndarray 
		numpy_test_inputs = test_inputs.as_matrix()
		numpy_test_labels = test_labels.as_matrix()
		test_label_categ = to_categorical(numpy_test_labels)

		return (numpy_test_inputs,test_label_categ)

	def get_all_data(self,csvfile):
		"""Returns all training data, validation data and test data from the given .csv file
		Should provide the fullname of file.
		For example if test data is in file test.csv then argument should be passed 
		as test.csv
		Returns: Numpy ndarray of train_inputs, train_labels, validatin_input,validation_labels
			test_inputs,test_labels"""

		filtered_data = self.preprocess_data(csvfile)

		#split failed project and successful project to balance the data set
		success_project = filtered_data.loc[filtered_data['state']==1]
		failed_project = filtered_data.loc[filtered_data['state']==0]
		# Split test,validation and training data from individual failed and 
		# successful project
		# To randomly shuffle the successful project data
		success_project = success_project.sample(frac=1)
		# To seperate 10% of data for the test data from successful project
		test_length_success = int(len(success_project)*0.1) 
		#To seperate 15% of data for the validation from succesful project
		validation_length_success = int(len(success_project)*0.25)
		success_test_data = success_project[:test_length_success :]
		success_validation_data = success_project[test_length_success:validation_length_success :]
		success_train_data = success_project[validation_length_success: :]

		# BALANCED THE DATA SET BY USING THE FOLLOWING STEPS
		# To randomly shuffle and get only 67.751% of the failed project data
		# Main purpose of this line is to balance the number of successful and failed project
		failed_project = failed_project.sample(frac=len(success_project)/len(failed_project))
		# To seperate 10% of data for the test data from failed project
		test_length_failed = int(len(failed_project)*0.1) 
		#To seperate 15% of data for the validation from failed project
		validation_length_failed = int(len(failed_project)*0.25)
		failed_test_data = failed_project[:test_length_failed :]
		failed_validation_data = failed_project[test_length_failed:validation_length_failed :]
		failed_train_data = failed_project[validation_length_failed: :]

		#Mergee test data from the failed project to the test data from the successful project
		final_test_data = pd.concat([success_test_data,failed_test_data],axis = 0)
		# Randomly shuffle the final_test_data to fixed the successful test data and failed test data
		final_test_data = final_test_data.sample(frac=1)

		#Mergee validation data from the failed project to the validation data from the successful project
		final_validation_data = pd.concat([success_validation_data,failed_validation_data],axis = 0)
		# Randomly shuffle the final_validaion_data to fixed the successful validaion data and failed validaion data
		final_validation_data = final_validation_data.sample(frac=1)

		#Mergee training data from the failed project to the training data from the successful project
		final_training_data = pd.concat([success_train_data,failed_train_data],axis = 0)
		# Randomly shuffle the final_training_data to fix the successful training data and failed training data
		final_training_data = final_training_data.sample(frac = 1)

		#Seperate inputs from label
		train_inputs = final_training_data.filter(items = ['main_category','currency','backers',\
				                  'country','usd_pledged_real','usd_goal_real'])
		train_labels = final_training_data.filter(items = ['state'])

		test_inputs = final_test_data.filter(items = ['main_category','currency','backers',\
				                  'country','usd_pledged_real','usd_goal_real'])
		test_labels = final_test_data.filter(items = ['state'])

		validation_inputs = final_validation_data.filter(items = ['main_category','currency','backers',\
				                  'country','usd_pledged_real','usd_goal_real'])
		validation_labels = final_validation_data.filter(items = ['state'])

		# Convert Pandas dataframe to numpy ndarray 
		numpy_train_inputs = train_inputs.as_matrix()
		numpy_train_labels = train_labels.as_matrix()
		numpy_test_inputs = test_inputs.as_matrix()
		numpy_test_labels = test_labels.as_matrix()
		numpy_validation_inputs = validation_inputs.as_matrix()
		numpy_validation_labels = validation_labels.as_matrix()

		# Convert all the labels to categorical data
		train_label_categ = to_categorical(numpy_train_labels)
		test_label_categ = to_categorical(numpy_test_labels)
		validation_label_categ = to_categorical(numpy_validation_labels)

		return (numpy_train_inputs,train_label_categ,numpy_validation_inputs,
		       validation_label_categ,numpy_test_inputs,test_label_categ)


	def train(self,numpy_train_inputs,train_label_categ,epoch,batchsize):
		"""Train the classification model
		Arguments:
		    numpy_train_inputs = numpy ndarray to train the model
		    train_label_categ = categorical numpy ndarray 
		    epoch = number of epoch to train the model
		    batchsize = size of the batch
		Save:
		    Save the trained weight and its architecture as trained.h5 in the current 
		    directory"""
		model = Sequential([
		Dense(12,activation='relu',input_shape = (6,)),
		Dense(12,activation='relu'),
		Dense(2,activation='softmax')
		])
		model.compile(optimizer='adam',loss = 'categorical_crossentropy',
		       metrics = ['accuracy'])    
		model.fit(numpy_train_inputs,train_label_categ,epochs = epoch, batch_size=batchsize)
		model.save('trained.h5') 

	def test(self,test_inputs,test_labels):
		"""Load pretrained model weights and  test the accuracy of test data
		Returns:
		    Accuray of the test data"""

		model = load_model('trained.h5')
		test_loss,test_accuracy = model.evaluate(test_inputs,test_labels)
		return test_accuracy


