# one vs all algorithm 
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

class Confusion_Matrix:
	def __init__(self):
		self.tp = 0 # true positive
		self.tn = 0 # true negative
		self.fp = 0 # false positive
		self.fn = 0 # false negative

class Success_Matrix:
	def __init__(self):
		self.setosa_matrix = Confusion_Matrix()
		self.virginica_matrix = Confusion_Matrix()
		self.versicolor_matrix = Confusion_Matrix()

def generate_success_matrix(validation_predictions, validation_data):
	# for storing all the results
	results = Success_Matrix()

	for index in range(len(validation_predictions)):
		# check all the predicitons and compare to validation data
		if validation_data[index][-1] == validation_predictions[index]:
			# report all correct answers in the aswers matrix
			if validation_data[index][-1] == "Iris-setosa":
				results.setosa_matrix.tp += 1
				results.versicolor_matrix.tn += 1	
				results.virginica_matrix.tn += 1
			if validation_data[index][-1] == "Iris-versicolor":
				results.versicolor_matrix.tp += 1
				results.setosa_matrix.tn += 1
				results.virginica_matrix.tn += 1
			if validation_data[index][-1] == "Iris-virginica":
				results.virginica_matrix.tp += 1
				results.setosa_matrix.tn += 1
				results.versicolor_matrix.tn += 1
		else:
			# if the prediction was wrong check what was the prediction and report a false positive on it
			if validation_predictions[index] == "Iris-setosa":
				results.setosa_matrix.fp += 1
				# we need also to report a false negative on the expected answer
				if validation_data[index][-1] == "Iris-versicolor":
					results.versicolor_matrix.fn += 1
					results.virginica_matrix.tn += 1
				else:
					results.virginica_matrix.fn += 1 # if its not a versicolor then it could be only virginica, and so on...
					results.versicolor_matrix.tn += 1

			if validation_predictions[index] == "Iris-versicolor":
				results.versicolor_matrix.fp += 1
				if validation_data[index][-1] == "Iris-setosa":	
					results.setosa_matrix.fn += 1
					results.virginica_matrix.tn += 1
				else:
					results.virginica_matrix.fn += 1
					results.setosa_matrix.tn += 1
					
			if validation_predictions[index] == "Iris-virginica":
				results.virginica_matrix.fp += 1
				if validation_data[index][-1] == "Iris-setosa":
					results.setosa_matrix.fn += 1
					results.versicolor_matrix.tn += 1
				else:
					results.versicolor_matrix.fn += 1
					results.setosa_matrix.tn += 1

	return results

# data visulaisation and performance metrics
def data_visualisation(training_data, validation_data, results):

	# print confusion matrix for each class
	col_labels = ['positive', 'negative'] 
	row_lables = ['true', 'false'] 
	setosa_values = [[str(results.setosa_matrix.tp),str(results.setosa_matrix.tn)], [str(results.setosa_matrix.fp),str(results.setosa_matrix.fn)]] 
	versicolor_values = [[str(results.versicolor_matrix.tp),str(results.versicolor_matrix.tn)], [str(results.versicolor_matrix.fp),str(results.versicolor_matrix.fn)]] 
	virginica_values = [[str(results.virginica_matrix.tp),str(results.virginica_matrix.tn)], [str(results.virginica_matrix.fp),str(results.virginica_matrix.fn)]] 

	fig, axs = plt.subplots(nrows = 3, ncols = 2, figsize=(11,7)) 
	axs[0, 0].set_axis_off() 
	axs[1, 0].set_axis_off()
	axs[2, 0].set_axis_off()
	axs[0 ,1].set_visible(False) # we dont want the extra 6th subplot

	axs[1, 1].set_position([0.6,0.6,0.35,0.35])
	axs[2, 1].set_position([0.6,0.08,0.35,0.35])

	axs[0, 0].set_title('confusion matrix for setosa flowers', fontweight ="bold") 
	axs[1, 0].set_title('confusion matrix for versicolor flowers', fontweight ="bold") 
	axs[2, 0].set_title('confusion matrix for virginica flowers', fontweight ="bold") 
	axs[2, 1].set_title('Iris flowers sepal parameters', fontweight ="bold")
	axs[1, 1].set_title('Iris flowers petal parameters', fontweight ="bold")

	table = axs[0, 0].table( 
    	cellText = setosa_values,  
    	rowLabels = row_lables,  
    	colLabels = col_labels, 
		colWidths = [0.3, 0.3],
    	rowColours =["lightblue"] * 2,  
    	colColours =["lightblue"] * 2, 
    	cellLoc ='center',  
    	loc ='center')   
	table.scale(1,2)      
   
	table = axs[1, 0].table( 
    	cellText = versicolor_values,  
    	rowLabels = row_lables,  
    	colLabels = col_labels, 
		colWidths = [0.3, 0.3],
    	rowColours =["lightblue"] * 2,  
    	colColours =["lightblue"] * 2, 
    	cellLoc ='center',  
    	loc ='center')
	table.scale(1,2)         
   
	table = axs[2, 0].table( 
    	cellText = virginica_values,  
    	rowLabels = row_lables,  
    	colLabels = col_labels,
		colWidths = [0.3, 0.3], 
    	rowColours =["lightblue"] * 2,  
    	colColours =["lightblue"] * 2, 
    	cellLoc ='center',  
    	loc ='center')
	table.scale(1,2)
   
	# add scatter plots
	plot_petal = axs[1, 1]
	color = ['red', 'green', 'blue', 'orange']
	labels= ['setosa', 'versicolor', 'virginica','test data']
	for row in training_data:
		if (row[-1] == "Iris-setosa"):
			axs[1, 1].scatter(row[0],row[1],c=color[0])
		elif(row[-1] == "Iris-versicolor"):
			axs[1, 1].scatter(row[0],row[1],c=color[1])
		elif(row[-1] == "Iris-virginica"):
			axs[1, 1].scatter(row[0],row[1],c=color[2])
	# addtest data to the plot
	for row in validation_data:
		axs[1, 1].scatter(row[0],row[1],c=color[3])
	for i in [0,1,2,3]:
		axs[1, 1].scatter([], [], c=color[i], label=labels[i])
	# add annotations
	axs[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=5)
	axs[1, 1].set_xlabel('Petal length')
	axs[1, 1].set_ylabel('Petal width')

	plot_sepal = axs[2, 1]
	color = ['red', 'green', 'blue', 'orange']
	labels= ['setosa', 'versicolor', 'virginica','test data']
	for row in training_data:
		if (row[-1] == "Iris-setosa"):
			axs[2, 1].scatter(row[2],row[3],c=color[0])
		elif(row[-1] == "Iris-versicolor"):
			axs[2, 1].scatter(row[2],row[3],c=color[1])
		elif(row[-1] == "Iris-virginica"):
			axs[2, 1].scatter(row[2],row[3],c=color[2])
	# add test data to the plot
	for row in validation_data:
		axs[2, 1].scatter(row[2],row[3],c=color[3])
	for i in [0,1,2,3]:
		axs[2, 1].scatter([], [], c=color[i], label=labels[i])
	# add annotations
	axs[2, 1].set_xlabel('Sepal length')
	axs[2, 1].set_ylabel('Sepal width')

	# calculate accuracy rate 

	TP = results.setosa_matrix.tp + results.versicolor_matrix.tp + results.virginica_matrix.tp
	TN = results.setosa_matrix.tn + results.versicolor_matrix.tn + results.virginica_matrix.tn
	FP = results.setosa_matrix.fp + results.versicolor_matrix.fp + results.virginica_matrix.fp
	FN = results.setosa_matrix.fn + results.versicolor_matrix.fn + results.virginica_matrix.fn

	# performance metrics
	accuracy = (TP+TN)/(TP+TN+FP+FN) * 100
	precision = TP/(TP+FP) * 100
	recall = TP/(TP+FN) * 100
	specificity = TN/(FP+TN) * 100
	f_score = 2*(precision*recall)/(precision+recall)/100

	metrics = [str(round(accuracy,2)) + " %",
			str(round(precision,2)) + " %",
			str(round(recall,2)) + " %",
			str(round(specificity,2)) + " %",
			str(round(f_score,1))]

	# report addidtional info on plot
	plt.gcf().text(0.15, 0.05, 'Accuracy: ' + metrics[0], fontsize=12)
	plt.gcf().text(0.15, 0.01, 'Recall:' + metrics[2], fontsize=12)
	plt.gcf().text(0.3, 0.05, 'f-score: ' + metrics[4],  fontsize=12)
	plt.gcf().text(0.3, 0.01, 'Precision: ' + metrics[1], fontsize=12)

# preparing training and validation datasets in a one-vs-rest fashion
def prepare_subdataset(source_dataset, class_name, p, reversed = False):
	target_dataset = copy.deepcopy(source_dataset)
	for row in target_dataset:
		if row[-1] == class_name:
			row[-1] = 1
		else:
			row[-1] = 0
	output_dataset = list()
	dataset_range = range(round(len(target_dataset)*p))
	if (reversed == True):
		dataset_range = range(round(len(target_dataset)*p),0,-1)
	for row in dataset_range:
		output_dataset.append(target_dataset[row])
	return output_dataset

###########################
# TESTING
###########################

# we test always for the same dataset
random.seed(69)

# load and prepare data
filename = 'iris.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)

random.shuffle(dataset)

dataset_setosa_training = prepare_subdataset(dataset, "Iris-setosa", 0.8)
dataset_versicolor_training = prepare_subdataset(dataset, "Iris-versicolor", 0.8)
dataset_virginica_training = prepare_subdataset(dataset, "Iris-virginica", 0.8)

# create a multi-class validation dataset
dataset_all_validation = list()
dataset_range = range(round(len(dataset)*0.2),0,-1)
for row in dataset_range:
	dataset_all_validation.append(dataset[row])

# just for visualisation to show what data was used as multi-class reference
dataset_all_training = list()
dataset_range = range(round(len(dataset)*0.8))
for row in dataset_range:
	dataset_all_training.append(dataset[row])


# initial parameters
learning_rate = 0.01
n_epoch = 10

# train wights for each class
weights_setosa = train_weights(dataset_setosa_training,learning_rate,n_epoch)
weights_versicolor = train_weights(dataset_versicolor_training,learning_rate,n_epoch)
weights_virginica = train_weights(dataset_virginica_training,learning_rate,n_epoch)

# create lists of activation scores of each class
activations_all = list()

for row in dataset_all_validation:
	activation_setosa = 0
	activation_versicolor = 0
	activation_virginica = 0
	
	for i in range(len(row)-1):
		activation_setosa += weights_setosa[i + 1] * row[i]
		activation_versicolor += weights_versicolor[i + 1] * row[i]
		activation_virginica += weights_virginica[i + 1] * row[i]
	
	activations_all.append([('Iris-setosa', activation_setosa), ('Iris-versicolor', activation_versicolor), ('Iris-virginica', activation_virginica)])

# class of sample is assigned by the highest activation value
row_num = 0
winner_value = 0
predictions = list()
for row in dataset_all_validation:
	winner_value = max(activations_all[row_num][0][1],activations_all[row_num][1][1],activations_all[row_num][2][1])
	for tuples_row in activations_all:
		for tuples_col in tuples_row:
			if tuples_col[1] == winner_value:
				predictions.append(tuples_col[0])
	row_num = row_num + 1

# calculate metrics and print report
results_matrix = generate_success_matrix(predictions, dataset_all_validation)
data_visualisation(dataset_all_training, dataset_all_validation, results_matrix)

plt.show()