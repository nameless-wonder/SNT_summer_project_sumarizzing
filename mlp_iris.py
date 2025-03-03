# -*- coding: utf-8 -*-
"""MLP_iris.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10i3_cPokC5e4X6WAYXEwatkbz4OYA8AJ
"""

pip install scikit-learn

import numpy as np
import pandas as pd
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame from the data and feature names
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the target variable to the DataFrame with flower names
df['target'] = iris.target
df['target_names'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Print the first few rows of the DataFrame
print(df.head())

#removing virginica data
df_final = df[df['target_names'] != 'virginica']
print(df_final.shape)

#dropping the target_names column
df_final = df_final.drop('target_names', axis = 1)

# Splitting the data into features (X) and target (y)
X = df_final.drop('target', axis=1)
y = df_final['target']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

df_final.head()

#normalizing the data
#(mean zero, unit variance)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

#analysing the data a bit
import seaborn as sns
import matplotlib.pyplot as plt

# Concatenate the features and target into a single DataFrame
data_rep = pd.concat([X_train, y_train], axis=1)

# Set the style of the plot
sns.set(style="ticks")

# Create the scatter plot matrix
sns.pairplot(data_rep, vars=['sepal length (cm)',	'sepal width (cm)',	'petal length (cm)',	'petal width (cm)'], hue='target')

# Display the plot
plt.show()

"""# Model making starts from here"""

#Accuracy function
def acc_score(y_pred , y_true):
  accuracy = np.sum(y_pred == y_true, axis = 0)/len(y_true)
  return accuracy

#Binary Cross-Entropy (loss, accuracy and gradient) for optimization
class bin_cross_entropy():
  def __init__(self):
    pass

  def loss(self,y,p):
  # Avoiding division by zero by clipping
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return - (y * np.log(p) + (1 - y) * np.log(1 - p)) 
  
  def accuracy(self,y,p):
    threshold = 0.5
    predictions = np.where(p>= threshold , 1 , 0)
    return acc_score(predictions , y)

  def gradient(self,y,p):
    # Avoiding division by zero by clipping
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return - (y / p) + (1 - y) / (1 - p)

#defining sigmoid function
class sigmoid():
  def __call__(self,x):
    return 1/(1 + np.exp(-x))

  def gradient(self,x):
    return self.__call__(x) * (1 - self.__call__(x))

#Making multilayered perceptron

class MultilayerPerceptron():
  def __init__(self,n_hidden,n_iter,eta):
    self.n_hidden = n_hidden
    self.n_iter = n_iter
    self.eta = eta
    self.hid_activation = sigmoid()
    self.out_activation = sigmoid()
    self.loss = bin_cross_entropy()

  def _initialize_wts(self,X,y):
    n_samples , n_features = X.shape[0],X.shape[1]
    n_outputs = y.shape[1]

    #for hidden layer
    limit = 1/math.sqrt(n_features)      #keeping the weights within a range
    self.w1 = np.random.uniform(-limit , limit , (n_features,self.n_hidden))  # weights are randomly initialized in a matrix form 
    self.b1 = np.zeros((1,self.n_hidden))   # all biases are set to 0 initially

    #for output layer 
    limit = 1/math.sqrt(self.n_hidden)   #keeping the weights within a range
    self.w2 = np.random.uniform(-limit , limit , (self.n_hidden ,n_outputs))   # weights are randomly initialized in a matrix form 
    self.b2 = np.zeros((1,n_outputs))       # output layer bias is set to 0 initially


  def fit(self , X , y):

    self._initialize_wts(X,y)

    for i in range(self.n_iter):

      #forward propogation
      ##hidden layer
      hid_input = X.dot(self.w1) + self.b1
      hid_output = self.hid_activation(hid_input)

      ##output layer
      out_layer_input = hid_output.dot(self.w2) + self.b2
      out_layer_output = self.out_activation(out_layer_input)


      #back propogation
      ##output layer
      ###grad wrt input of output layer
      grad_wrt_out_l_input = self.loss.gradient(y,out_layer_output) * self.out_activation.gradient(out_layer_input)
      grad_w2 = hid_output.T.dot(grad_wrt_out_l_input)
      grad_b2 = np.sum(grad_wrt_out_l_input, axis=0)

      ##hidden layer 
      ###grad wrt input of hidden layer
      grad_wrt_hidden_l_input = grad_wrt_out_l_input.dot(self.w2.T) * self.hid_activation.gradient(hid_input)
      grad_w1 = X.T.dot(grad_wrt_hidden_l_input)
      grad_b1 = np.sum(grad_wrt_hidden_l_input, axis=0)

      #Update weights (by gradient descent)
      ##Move against the gradient to minimize loss
      self.w2  -= self.eta * grad_w2
      self.b2 -= self.eta * grad_b2
      self.w1  -= self.eta * grad_w1
      self.b1 -= self.eta * grad_b1

  
  #Prediction after iterations
  def predict(self,X):
    hid_input = X.dot(self.w1) + self.b1
    hid_output = self.hid_activation(hid_input)
    out_layer_input = hid_output.dot(self.w2) + self.b2
    out_layer_output = self.out_activation(out_layer_input)
    
    ##converting probabilities into 0 and 1 for binary classification using threshold
    threshold = 0.5
    binary_predictions = np.where(out_layer_output >= threshold, 1, 0)

    return binary_predictions

print(X_train.shape ,   y_train.shape)

"""# Training and making predictions"""

X_train = X_train.values  # Convert X_train DataFrame to NumPy array
y_train = y_train.values.reshape(-1, 1)  # Reshape y_train to have a second dimension

clf = MultilayerPerceptron(n_hidden=4, n_iter=3000, eta=0.01)
clf.fit(X_train, y_train)

# Predict on test data
X_test = X_test.values
y_pred = clf.predict(X_test)

# Flatten the y_test and y_pred arrays
y_test_flat = y_test.values.flatten()
y_pred_flat = y_pred.flatten()

# Calculate the accuracy
accuracy = acc_score(y_test_flat, y_pred_flat)
print("Accuracy:", accuracy)