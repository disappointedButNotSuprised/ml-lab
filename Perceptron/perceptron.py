# Perceptron Algorithm 
from random import seed
from random import randrange
from csv import reader
import random
import matplotlib.pyplot as plt

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

# divide data into trainig and testing subsets
def divide_data(dataset):
	training_dataset = list()
	testing_dataset = list()
	# assign records of the dataset to each subset accoring to randomized order
	random.seed(69) # randomize in the same way every run
	RNG_list = random.sample(range(len(dataset)), len(dataset))
	for record in range(len(dataset)):
		if(record < 0.6*len(dataset)):
			training_dataset.append(dataset[RNG_list[record]])
		else:
			testing_dataset.append(dataset[RNG_list[record]])
	# return dataset as list of sublists
	full_dataset = [training_dataset, testing_dataset]
	return full_dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# make a prediction according to provided weights
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	if activation >= 0.0:
		return 1.0
	else: 
		return-1.0

# estimation of perceptron weights by SGD
def train_weights(train, learning_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			prediction = predict(row, weights)
			if prediction == -1.0:
				prediction = 0.0
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
		if prediction == -1.0:
			prediction = 0.0
		predictions.append(prediction)
	return(predictions)

# run perceptron and report results
def run_algorithm(train_set, test_set, learning_rate, n_epoch):
	predicted = perceptron(train_set, test_set, learning_rate, n_epoch)
	target_outputs = [row[-1] for row in test_set]
	accuracy = accuracy_metric(target_outputs, predicted)
	generate_confusion_matrix(test_set, learning_rate, n_epoch, predicted)
	return accuracy

def generate_confusion_matrix(test_dataset, learning_rate, n_epoch, predictions):
	matrix=list()
	matrix=[[0 for i in range(2)] for j in range(2)]
	i = 0
	for row in test_dataset:
        #constructing the actual matrix
		if row[-1]==0 and predictions[i]==0.0:
			matrix[0][0]+=1 #tp
		elif row[-1]!=0 and predictions[i]==0.0:
			matrix[0][1]+=1 #fp
		elif row[-1]==0 and predictions[i]!=0.0:
			matrix[1][0]+=1 #fn
		elif row[-1]!=0 and predictions[i]!=0.0:
			matrix[1][1]+=1 #tn
		i = i + 1
	
	# data vislualisation
	axs[0].set_axis_off() 
	axs[0].set_title('confusion matrix for \n banknote authentication algorithm effectiveness', fontweight ="bold")
	table = axs[0].table( 
    	cellText = [[str(matrix[0][0]),str(matrix[1][1])], [str(matrix[0][1]),str(matrix[1][0])]],  
    	rowLabels = ['true', 'false'],  
    	colLabels = ['positive', 'negative'] , 
		colWidths = [0.4, 0.4],
    	rowColours =["lightblue"] * 2,  
    	colColours =["lightblue"] * 2, 
    	cellLoc ='center',  
    	loc ='center')
	table.scale(1,1.5) 


###########################
# TESTING
###########################

# load and prepare data
filename = './Perceptron/data.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
str_column_to_int(dataset, len(dataset[0])-1)
dataset_divided = divide_data(dataset)
training_dataset = dataset_divided[0]
testing_dataset = dataset_divided[1]

# initial parameters
learning_rate = 0.01
n_epoch = 1 # will cycle up to 10

# prepare plot
fig, axs = plt.subplots(nrows = 1, ncols = 2,figsize=(12,4))

scores = list()
for n_epoch in range(10):
	score = run_algorithm(training_dataset, testing_dataset, learning_rate, n_epoch)
	scores.append(score)
	n_epoch = n_epoch + 1

# plot extra properties
axs[1].plot(range(10), scores)
axs[1].set_title('algorithm accuracy in relation to epoch number', fontweight ="bold")
axs[1].set_xlabel('N of epochs')
axs[1].set_ylabel('accuracy')

plt.gcf().text(0.2, 0.35, 'Accuracy rate: %.3f%%' % score, fontsize=12)
plt.show()