# Machine_Learning_Model
Understanding the Basics of Machine Learning through a problem statement using the data from a csv dataset.

The problem statement is as follows:

The goal of this task is to build a neural network model that predicts whether an employee will leave the present company or not depending on various factors. Given a dataset containing historical employee information, the model should predict whether a current employee is likely to leave the company. 
The dataset contains the following Columns:

(a) Education: The highest level of education attained by the individual. 
(b) JoiningYear: The year in which the employee joined the current company. 
(c) City: The city that the individual belongs to.
(d) PaymentTier: Depending on their current salary, the individuals are classified into different tiers.
(e) Age: Age of the employee.
(f) Gender: The gender of the individual.
(g) EverBenched: This is a boolean which tells whether the employee was ever put on the bench in the current company.
(h) ExperienceInCurrentDomain: Years of experience the individual has in the domain they are currently working. 
(i) LeaveOrNot: This is the target variable (boolean) which tells whether the employee will leave the company or not. 

The constraints are as follows:

No Pretrained Models: Participants are not allowed to use pretrained machine learning models or external datasets. The model should be trained from scratch using the provided dataset: task_1a_dataset.csv.
Neural Network architecture for training the dataset has to be developed by the team.
The trained model should be saved as task_1a_trained_model.pth using the torch.jit.save() function as mentioned in the main() function of task_1a.py file.
The code stub provided (task_1a.py) must be followed mandatorily by the teams.
You are allowed to add other functions to the Python file but should not delete the functions which are already provided.
You are allowed to import additional functions from the following libraries: "pandas, numpy, matplotlib, torch and sklearn". Importing any other libraries might cause an issue while running the executable for auto-evaluation.
main() function of task_1a.py should not be modified by the teams. Go through the function thoroughly to understand the flow of the code.

The Boiler Plate Description is as follows:

1. data_preprocessing()

Function name	data_preprocessing()
Purpose	This function will be used to load your csv dataset and preprocess it. Preprocessing involves cleaning the dataset by removing unwanted features, decision about what needs to be done with missing values etc. Note that there are features in the csv file whose values are textual (eg: Industry, Education Level etc). These features might be required for training the model but can not be given directly as strings for training. Hence this function should return encoded dataframe in which all the textual features are numerically labeled
Input Argument	task_1a_dataframe : [ Pandas Dataframe ]
Returns	encoded_dataframe : [ Pandas Dataframe ] 
Processed data as Pandas Dataframe 
Example Call	encoded_dataframe = data_preprocessing(task_1a_dataframe)

2. identify_features_and_targets()

Function name	identify_features_and_targets()
Purpose	The purpose of this function is to define the features and the required target labels. The function returns a python list in which the first item is the selected features and second item is the target label
Input Argument	encoded_dataframe : [ Dataframe ]
Returns	features_and_targets : [ list ] 
python list in which the first item is the selected features and second item is the target label 
Example Call	features_and_targets = ideantify_features_and_targets(encoded_dataframe)

3. load_as_tensors()

Function name	load_as_tensors()
Purpose	This function aims at loading your data (both training and validation) as PyTorch tensors. Here you will have to split the dataset for training and validation, and then load them as as tensors. Training of the model requires iterating over the training tensors. Hence the training sensors need to be converted to iterable dataset object
Input Argument	features_and targets : [ list ]
Returns	tensors_and_iterable_training_data : [ list ] 
Items:
[0]: X_train_tensor: Training features loaded into Pytorch array 
[1]: X_test_tensor: Feature tensors in validation data 
[2]: y_train_tensor: Training labels as Pytorch tensor 
[3]: y_test_tensor: Target labels as tensor in validation data 
[4]: Iterable dataset object and iterating over it in batches, which are then fed into the model for processing 
Example Call	tensors_and_iterable_training_data = load_as_tensors(features_and_targets)

4. class Salary_Predictor(nn.Module)

Function name	class Salary_Predictor(nn.Module)
Purpose	The architecture and behavior of your neural network model will be defined within this class that inherits from nn.Module. Here you also need to specify how the input data is processed through the layers. It defines the sequence of operations that transform the input data into the predicted output. When an instance of this class is created and data is passed through it, the forward method is automatically called, and the output is the prediction of the model based on the input data
Input Argument	None
Returns	predicted_output 
Predicted output for the given input data

5. model_loss_function()

Function name	model_loss_function()
Purpose	To define the loss function for the model. Loss function measures how well the predictions of a model match the actual target values in training data
Input Argument	None
Returns	loss_function 
This can be a pre-defined loss function in PyTorch or can be user-defined 
Example Call	loss_function = model_loss_function()

6. model_optimizer()

Function name	model_optimizer()
Purpose	To define the optimizer for the model. Optimizer is responsible for updating the parameters (weights and biases) in a way that minimizes the loss function
Input Argument	model: An object of the 'Salary_Predictor' class
Returns	optimizer 
Pre-defined optimizer from pytorch 
Example Call	optimizer = model_optimizer(model)

7. model_number_of_epochs()

Function name	model_number_of_epochs()
Purpose	To define the number of epochs for training the model
Input Argument	None
Returns	number_of_epochs: [Integer value] 
Example Call	number_of_epochs = model_number_of_epochs()

8. training_function()

Function name	training_function()
Purpose	All the required parameters for training are passed to this function.
Input Argument	1. model: An object of the 'Salary_Predictor' class 
2. number_of_epochs: For training the model 
3. tensors_and_iterable_training_data: list containing training and validation data tensors and iterable dataset object of training tensors 
4. loss_function: Loss function defined for the model 
5. optimizer: Optimizer defined for the model
Returns	trained_model 
Example Call	trained_model = training_function(model, number_of_epochs, iterable_training_data, loss_function, optimizer)

9. validation_function()

Function name	validation_function()
Purpose	This function will utilise the trained model to do predictions on the validation dataset. This will enable us to understand the accuracy of the model
Input Argument	1. trained_model: Returned from the training function 
2. tensors_and_iterable_training_data: list containing training and validation data tensors and iterable dataset object of training tensors
Returns	model_accuracy: Accuracy on the validation dataset 
Example Call	model_accuracy = validation_function(trained_model, tensors_and_iterable_training_data)
