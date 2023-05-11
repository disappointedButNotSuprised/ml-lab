import numpy as np
import pandas as pd
from csv import reader
from math import sqrt
import matplotlib.pyplot as plt
import csv
import random
from statistics import mode

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

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Convert string column to float
def str_column_to_float(dataset, column):
 for row in dataset:
    row[column] = float(row[column].strip())

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
	return dataset

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

# Make a prediction with the few nearest neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = mode(output_values)
	return prediction

# divide data into trainig, validation and testing subsets
def divide_data(dataset):
	training_dataset = list()
	validation_dataset = list()
	testing_dataset = list()
	# assign records of the dataset to each subset accoring to randomized order
	random.seed(69) # randomize in the same way every run
	RNG_list = random.sample(range(len(dataset)), len(dataset))
	for record in range(len(dataset)):
		if(record < 0.6*len(dataset)):
			training_dataset.append(dataset[RNG_list[record]])
		if(record >= 0.6*len(dataset) and record < 0.8*len(dataset)):
			validation_dataset.append(dataset[RNG_list[record]])
		if(record >= 0.8*len(dataset)):
			testing_dataset.append(dataset[RNG_list[record]])
	# return dataset as list of sublists
	full_dataset = [training_dataset, validation_dataset, testing_dataset]
	return full_dataset


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

#data visualisation
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
	correct_total = results.setosa_matrix.tp + results.versicolor_matrix.tp + results.virginica_matrix.tp
	accuracy = correct_total/float(len(validation_data))

	# calculate precision
	precision = correct_total / (correct_total + results.setosa_matrix.fp + results.versicolor_matrix.fp + results.virginica_matrix.fp)

	# calculate recall
	recall = correct_total / (correct_total + results.setosa_matrix.fn + results.versicolor_matrix.fn + results.virginica_matrix.fn)
	
	# calculate f-score
	f_score = 2 * (precision*recall)/(precision+recall)
	
	# report addidtional info on plot
	plt.gcf().text(0.15, 0.05, "Assumed k = " + str(num_neighbors), fontsize=12)
	plt.gcf().text(0.15, 0.01, f"Recall: {recall :.2%}", fontsize=12)
	plt.gcf().text(0.3, 0.05, f"Accuracy: {accuracy :.2%}", fontsize=12)
	plt.gcf().text(0.3, 0.01, f"Precision: {precision :.2%}", fontsize=12)
	plt.gcf().text(0.225, 0.09, f"f-score: {f_score :.2%}", fontsize=12)

	return accuracy

#*******************************
# Testing
#*******************************

filename = 'iris.csv'
dataset = load_csv(filename)

# convert all values to folat
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)

# data normalisation
dataset = normalize_dataset(dataset, dataset_minmax(dataset))

# divide the data into 60/20/20 portions for training, validation and testing 
dataset_divided = divide_data(dataset)
training_dataset = dataset_divided[0]
validation_dataset = dataset_divided[1]
testing_dataset = dataset_divided[2]

k_list = list()
accuracy_list = list()

# run severeal times to check performacne in realation to k value
for k in range(1,len(training_dataset)):
	predictions = list()
	num_neighbors = k

	# perform predicitons of every data point and collect into list
	for test_point in validation_dataset:
		predictions.append(predict_classification(training_dataset, test_point, num_neighbors))
	results_matrix = generate_success_matrix(predictions, validation_dataset)
	
	# print verbose statistics just for the optimal k value
	if num_neighbors == round(sqrt(len(training_dataset))):
		data_visualisation(training_dataset, validation_dataset, results_matrix)

	# calculate accuracy for current run
	correct_total = results_matrix.setosa_matrix.tp + results_matrix.versicolor_matrix.tp + results_matrix.virginica_matrix.tp
	accuracy = correct_total/float(len(validation_dataset))
	
	accuracy_list.append(accuracy)
	k_list.append(k)

# results visulaisation
fig, ax = plt.subplots()
ax.plot(k_list, accuracy_list)
ax.set_title('algorithm accuracy in relation to k value', fontweight ="bold")
ax.set_xlabel('k value')
ax.set_ylabel('accuracy')
plt.show()

