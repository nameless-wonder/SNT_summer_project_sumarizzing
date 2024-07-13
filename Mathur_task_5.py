#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math


# In[2]:


get_ipython().system('pip install py7zr')


# In[2]:


import os

# Get the list of files in the current directory
files = os.listdir()
for f in files:
    print(f)


# In[3]:


# Get the current directory's name
current_dir = os.path.basename(os.getcwd())

# Print the current directory's name
print("Current directory name:", current_dir)


# In[5]:


import py7zr
file_path = 'train.7z'
dest_dir = 'Cifar-10_Mathur'

with py7zr.SevenZipFile(file_path, mode='r') as archive:
    archive.extractall(path=dest_dir)
    


# In[60]:


import argparse


# In[ ]:


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.5, help="Momentum value")
    parser.add_argument("--num_hidden", type=int, default=3, help="Number of hidden layers")
    parser.add_argument("--sizes", type=str, default="100,100,100", help="Hidden layer sizes")
    parser.add_argument("--activation", type=str, default="sigmoid", help="Activation function")
    parser.add_argument("--loss", type=str, default="sq", help="Loss function")
    parser.add_argument("--opt", type=str, default="adam", help="Optimizer")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
    parser.add_argument("--anneal", type=bool, default=True, help="Annealing")
    parser.add_argument("--save_dir", type=str, default="pa1/", help="Save directory")
    parser.add_argument("--expt_dir", type=str, default="pa1/exp1/", help="Experiment directory")
    parser.add_argument("--train", type=str, default="train", help="Training file")
    parser.add_argument("--test", type=str, default="test", help="Testing file")

    args = parser.parse_args()

    learning_rate = args.lr
    momentum = args.momentum
    num_hidden = args.num_hidden
    sizes =[int(size) for size in args.sizes.split(",")],
    activation = args.activation,
    loss = args.loss,
    opt = args.opt,
    batch_size = args.batch_size,
    anneal = args.anneal,
    save_dir = args.save_dir,
    expt_dir = args.expt_dir,
    train = args.train,
    test = args.test
get_ipython().run_line_magic('tb', '')


# In[4]:


get_ipython().system('pip install opencv-python')


# In[5]:


import cv2
images = []

folder_name = 'train'

pictures = [i for i in os.listdir(folder_name)]

for i in pictures:
    img = cv2.imread(folder_name + '/' + i)
    images.append(img)
    


# In[6]:


X_train = np.array(images)
X_train = X_train.astype('float32')/255
X_train


# In[7]:


print(X_train.shape)


# In[8]:


# Read the CSV file into a DataFrame
Y_train = pd.read_csv('trainLabels.csv')
Y_train = Y_train.to_numpy()
Y_train


# In[9]:


import sklearn


# In[10]:


from sklearn.preprocessing import OneHotEncoder

oe = OneHotEncoder()

Y_train_category = Y_train[:, 1].reshape(-1, 1)

Y_train_enc = oe.fit_transform(Y_train_category).toarray()

Y_train_enc


# In[11]:


print(Y_train_enc.shape)


# In[12]:


X_train_gray = np.zeros((X_train.shape[0], X_train.shape[1], X_train.shape[2]))

for i in range(X_train.shape[0]):
    X_train_gray[i] = cv2.cvtColor(X_train[i], cv2.COLOR_RGB2GRAY)



# In[13]:


print(X_train_gray.shape)


# In[14]:


X_train_flat = X_train_gray.reshape(X_train_gray.shape[0], -1)
print(X_train_flat.shape)


# In[48]:


class varMLP():
    def __init__(self,learning_rate,num_hidden,sizes,momentum,loss,opt,batch_size,anneal):
        self.learning_rate = learning_rate
        self.num_hidden = num_hidden
        self.sizes = sizes
        self.momentum = momentum
        if (loss == 'ce'):
            self.loss = self._cross_entropy_loss
        else:
            self.loss = self._squared_error_loss
        
        self.opt = opt
        self.batch_size = batch_size
        self.anneal = anneal
        self.weights = []
        self.biases = []
        
        
        # storing all layer sizes in a list 
        layers_sizes = [1024] + sizes + [10]
        
        #initializing the weights and biases randomly
        for i in range(1,num_hidden+2):
            weight_shape = (layers_sizes[i],layers_sizes[i-1])
            bias_shape = (self.batch_size,layers_sizes[i])
            self.weights.append(np.random.randn(*weight_shape))
            self.biases.append(np.random.randn(*bias_shape))
    
    def forward(self,X):
        activations = [X]
        outputs = []
        
        for i in range(self.num_hidden):
            z = np.dot(self.weights[i],activations[i].T) + self.biases[i]
            a = self._sigmoid(z)
            activations.append(a)
            outputs.append(z)
            
        z = np.dot(self.weights[-1],activations[-1]) + self.biases[-1]
        a = self._softmax(z)
        activations.append(a)
        outputs.append(z)
        
        return activations , outputs
    
    def backward(self, X, y, activations, outputs, learning_rate):
        gradients = []
        batch_size = self.batch_size

        # Compute gradients for output layer
        dZ = activations[-1] - y
        dW = (1 / batch_size) * np.dot(dZ, activations[-2].T)
        db = (1 / batch_size) * np.sum(dZ, axis=1, keepdims=True)
        gradients.append((dW, db))

        # Compute gradients for hidden layers
        for i in range(self.num_hidden, -1, -1):
            dA = np.dot(self.weights[i + 1].T, dZ)
            dZ = dA * self._sigmoid_derivative(outputs[i])
            dW = (1 / batch_size) * np.dot(dZ, activations[i].T)
            db = (1 / batch_size) * np.sum(dZ, axis=1, keepdims=True)
            gradients.append((dW, db))

        # Update weights and biases using gradients
            if (self.opt == 'gd'): 
                for i in range(self.num_hidden):
                    self.weights[i] -= learning_rate * gradients[-(i + 1)][0]
                    self.biases[i] -= learning_rate * gradients[-(i + 1)][1]
            
            if (self.opt == 'momentum'):
                for i in range(self.num_layers):
                    self.weights[i] -= learning_rate * gradients[-(i + 1)][0] + self.momentum * self.weights[i]
                    self.biases[i] -= learning_rate * gradients[-(i + 1)][1] + self.momentum * self.biases[i]    

    def train(self, X, y, num_epochs=100):
        
        batch_size = self.batch_size
        learning_rate = self.learning_rate
        num_samples = X.shape[0]
        
        anneal_factor = 0.5
                
        for epoch in range(num_epochs):
            # Shuffle training data
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[:, indices]


            # Split data into batches
            num_batches = num_samples // batch_size
            for batch in range(num_batches):
                start = batch * batch_size
                end = (batch + 1) * batch_size
                                
                X_batch = X_shuffled[start:end].T
                y_batch = y_shuffled[start:end].T[:10, :]

                '''y_batch = y_shuffled[start:end].T'''

                # Forward pass
                activations, outputs = self.forward(X_batch)

                # Backward pass
                self.backward(X_batch, y_batch, activations, outputs, learning_rate)

            #training loss and accuracy
            train_loss, train_accuracy = self.evaluate(X.T, y.T)
            print(f"Epoch {epoch + 1}: Loss = {train_loss:.4f}, Accuracy = {train_accuracy:.2%}")

    def evaluate(self, X, y):
        activations, _ = self.forward(X)
        y_pred = np.argmax(activations[-1], axis=0)
        y_true = np.argmax(y, axis=0)
        loss_value = self.loss(activations[-1], y)
        
        accuracy = np.mean(y_pred == y_true)
        return loss_value, accuracy

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _softmax(self, z):
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def _cross_entropy_loss(self, y_pred, y_true):
        epsilon = 1e-10
        clipped_y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(clipped_y_pred))

    def _squared_error_loss(self, y_pred, y_true):
        return 0.5 * np.mean((y_pred - y_true)**2)

            
            


# In[62]:


clf = varMLP(0.01,3,[4,5,6],0.5,'ce','gd',5,0.5)
clf.train(X_train_flat.T,Y_train_enc.T, num_epochs=100)
#self,learning_rate,num_hidden,sizes,momentum,loss,opt,batch_size,anneal

