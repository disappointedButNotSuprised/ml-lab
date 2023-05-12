# Perceptron Algorithm 
from random import seed
from random import randrange
from csv import reader
import random
import matplotlib.pyplot as plt
import copy

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# make a prediction according to provided weights
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	if activation >= 0.0:
		return 1.0
	else: 
		return 0.0

# estimation of perceptron weights by SGD
def train_weights(train, learning_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			weights[0] = weights[0] + learning_rate * error # adjust bias	
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + learning_rate * error * row[i] # adjust weights
	return weights

# the actual perceptron code
def perceptron(train, test, learning_rate, n_epoch):
	predictions = list()
	weights = train_weights(train, learning_rate, n_epoch) # train the perceptron
	# then use it on the test data
	for row in test:
		prediction = predict(row, weights)
		predictions.append(prediction)
	return(predictions)

# run perceptron and report results
def run_algorithm(train_set, test_set, learning_rate, n_epoch, plot_nr, flower_name):
	predicted = perceptron(train_set, test_set, learning_rate, n_epoch)
	data_visualisation(test_set, learning_rate, n_epoch, predicted, plot_nr, flower_name)

# data visulaisation and performance metrics
def data_visualisation(test_dataset, learning_rate, n_epoch, predictions, plot_nr, flower_name):
	
	# confusion matrix
	matrix=list()
	matrix=[[0 for i in range(2)] for j in range(2)]
	i = 0
	for row in test_dataset:
        #constructing the actual matrix
		if row[-1] == 0 and predictions[i] == 0.0:
			matrix[0][0] += 1 #tp
		elif row[-1] != 0 and predictions[i] == 0.0:
			matrix[0][1] += 1 #fp
		elif row[-1] == 0 and predictions[i] != 0.0:
			matrix[1][0] += 1 #fn
		elif row[-1] != 0 and predictions[i] != 0.0:
			matrix[1][1] += 1 #tn
		i = i + 1

	tp = matrix[0][0]
	fp = matrix[0][1]
	fn = matrix[1][0]
	tn = matrix[1][1]

	# confusion matrix table print
	axs[0, plot_nr].set_axis_off() 
	axs[0, plot_nr].set_title("performance metrics for\n" + flower_name + " flowers", fontweight ="bold")
	conf_matrix = axs[0, plot_nr].table( 
    	cellText = [[str(matrix[0][0]),str(matrix[1][1])], [str(matrix[0][1]),str(matrix[1][0])]],  
    	rowLabels = ['true', 'false'],  
    	colLabels = ['positive', 'negative'] , 
		colWidths = [0.4, 0.4],
    	rowColours =["lightblue"] * 2,  
    	colColours =["lightblue"] * 2, 
    	cellLoc ='center',  
    	loc ='center') 
	conf_matrix.scale(1,1.5) 

	# performance metrics
	accuracy = (tp+tn)/(tp+tn+fp+fn) * 100
	precision = tp/(tp+fp) * 100
	recall = tp/(tp+fn) * 100
	specificity = tn/(fp+tn) * 100
	f_score = 2*(precision*recall)/(precision+recall)/100

	metrics = [	[str(round(accuracy,2)) + " %"],
				[str(round(precision,2)) + " %"],
				[str(round(recall,2)) + " %"],
				[str(round(specificity,2)) + " %"],
				[str(round(f_score,1))]]

	# performance metrics table print
	axs[1, plot_nr].set_axis_off() 
	perf_metrics = axs[1, plot_nr].table( 
    	cellText = metrics,  
    	rowLabels = ['accuracy', 'precision', 'recall', 'specificity', 'f-score'],   
		colWidths = [0.4],
    	rowColours =["lightblue"] * 5,   
    	cellLoc ='center',  
    	loc ='center') 
	perf_metrics.scale(1,1.5)

	# center the metrics table to the confusion matrix
	box = axs[1, plot_nr].get_position()
	box.x0 = box.x0 + 0.05
	axs[1, plot_nr].set_position(box)

# for preparing training and validation datasets in a one vs rest fashion
def prepare_subdataset(source_dataset, class_name, p):
	target_dataset = copy.deepcopy(source_dataset)
	for row in target_dataset:
		if row[-1] == class_name:
			row[-1] = 1
		else:
			row[-1] = 0
	output = list()
	for row in range(round(len(target_dataset)*p)):
		output.append(target_dataset[row])
	return output	

###########################
# TESTING
###########################

# load and prepare data
filename = 'iris.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)

random.shuffle(dataset)

dataset_setosa_training = prepare_subdataset(dataset, "Iris-setosa", 0.8)
dataset_versicolor_training = prepare_subdataset(dataset, "Iris-versicolor", 0.8)
dataset_virginica_training = prepare_subdataset(dataset, "Iris-virginica", 0.8)

dataset_setosa_validation = prepare_subdataset(dataset, "Iris-setosa", 0.2)
dataset_versicolor_validation = prepare_subdataset(dataset, "Iris-versicolor", 0.2)
dataset_virginica_validation = prepare_subdataset(dataset, "Iris-virginica", 0.2)

# initial parameters
learning_rate = 0.01
n_epoch = 10

# prepare plot
fig, (axs) = plt.subplots(nrows = 2, ncols = 3,figsize=(12,4))

run_algorithm(dataset_setosa_training, dataset_setosa_validation, learning_rate, n_epoch, 0, "setosa")
run_algorithm(dataset_versicolor_training, dataset_versicolor_validation, learning_rate, n_epoch, 1, "virginica")
run_algorithm(dataset_virginica_training, dataset_virginica_validation, learning_rate, n_epoch, 2, "versicolor")

plt.show()