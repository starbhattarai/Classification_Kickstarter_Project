## To Use a Python module, final_module.py, Please follow the following steps:
        1.Import the final_module:
            a. import final_module as fm
        2.Initialize the class, CompleteCode as:
            a. ob = fm.CompleteCode()
        3. Then access the required function using ob, which is the object of CompleteCode class
   
## Details of the functions within the module are as follows:
        1. preprocess_data(csvfile):
        
            """Load csv file from the current directory
		    Should provide the fullname of file.
		    For example if test data is in file test.csv then argument should be passed 
		    as 'test.csv'
		 Returns: Pandas dataframe  """
            
        2. get_test_data(csvfile):
        
            """Load csv file from the current directory
		    Should provide the fullname of file.
		    For example if test data is in file test.csv then argument should be passed 
		    as 'test.csv'
		 Returns: Numpy ndarray of Test inputs and Test labels  """
        
        3. get_all_data(csvfile):
        
            """Returns all training data, validation data and test data from the given .csv file
            Should provide the fullname of file.
            For example if test data is in file test.csv then argument should be passed 
            as 'test.csv'
		 Returns: Numpy ndarray of train_inputs, train_labels, validatin_input,validation_labels
			test_inputs,test_labels"""
            
        4. train(train_inputs,train_label,epoch,batchsize):
        
            """Train the classification model
		 Arguments:
		    numpy_train_inputs = numpy ndarray to train the model
		    train_label_categ = categorical numpy ndarray 
		    epoch = number of epoch to train the model
		    batchsize = size of the batch
		 Save:
		    Save the trained weight and its architecture as trained.h5 in the current 
		    directory"""
            
        5. test(test_inputs,test_labels):
        
            """Load pretrained model weights and  test the accuracy of test data
		 Returns:
		    Accuray of the test data"""
## To test the test data on the model
        1. Use get_test_data(), it will automatically use preprocess_data()
        2. Pass the test_inputs and test_labels return from get_test_data() to the test() function.
           It will use the pretrained model which is saved within the current directory
## To train the model 
        1. Use get_all_data(), it will automatically use preprocess_data(). 
        2. Pass train_inputs and train_labels return from get_all_data() to the train() function.
           It will save the train model as trained.h5
