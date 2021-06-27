"""
Aaron Safer-Rosenthal
20068164
17asr

Provides classes for the program to use
"""

import re
import numpy as np

#Perceptron class using simple feedback learning ([delta]w = cx)
class Perceptron:

    #Initializes the Perceptron object
    def __init__(self, learningRate=0.01, numIters=1000):
        self.learningRate = learningRate
        self.numIters = numIters
        self.activationFunction = self.unitStepFunction
        self.weights = None #will be set later
        self.bias = None #will be set later

    #returns 1 if x>=0, else 0
    def unitStepFunction(self, x):
        return np.where(x>=0, 1, 0)#Using np.where so it works on multiple
                                   #inupts

    #Input: training samples (X) and training labels (y)
    #X is n*d array of size m*n. M=rows/num of samples, N=num columns/features
    def fit(self, X, y):
        
        numSamples, numFeatures = X.shape

        #Initializing attributes
        self.weights = np.zeros(numFeatures)
        self.bias = 0

        #want to make sure y values are only 0 or 1
        y_ = np.array([1 if i>0 else 0 for i in y])

        bestAccuracy = 0

        #updating wieghts and bias
        for i in range(self.numIters):
            
            for index, currentSample in enumerate(X):
                linearOutput = np.dot(currentSample, self.weights) + self.bias
                yPrediction = self.activationFunction(linearOutput)                
                update = self.learningRate * (y_[index] - yPrediction)
                self.weights += update * currentSample
                self.bias += update
                        
    #Input: training samples
    #Output: prediction
    def predict(self, X):
        linearOutput = np.dot(X, self.weights) + self.bias #w.x + bias
        yPrediction = self.activationFunction(linearOutput)
        return yPrediction

#Pocket claass that uses the Pocket algorithm for training
#(Is also a perceptron)
class Pocket:

    #Initializes the Perceptron object
    def __init__(self, learningRate=0.01, numIters=1000):
        self.learningRate = learningRate
        self.numIters = numIters
        self.activationFunction = self.unitStepFunction
        self.weights = None #will be set later
        self.bias = None #will be set later
        self.pocket = None #will be set later
        self.longestRun = 0

    #returns 1 if x>=0, else 0
    def unitStepFunction(self, x):
        return np.where(x>=0, 1, 0) #Using np.where so it works on multiple
                                    #inupts

    #Input: training samples (X) and training labels (y)
    #X is n*d array of size m*n. M=rows/num of samples, N=num columns/features
    def fit(self, X, y):

        numSamples, numFeatures = X.shape

        #initializing attributes
        self.weights = np.zeros(numFeatures)
        self.pocket = np.zeros(numFeatures)
        self.bias = 0

        for _ in range(self.numIters):

            currentLongestRun = 0
            currentBestWeights= self.pocket.copy() #copying so it copies values
                                                   #and not the object itself
            self.weights = self.pocket.copy()

            for index, currentSample in enumerate(X):
                currentLongestRun += 1
                yPrediction = self.predict(currentSample)

                #print("currentSample:", currentSample) #flag
                #print("y[index]:", y[index], "| yPrediction:",yPrediction)#flag

                #the run is over
                if y[index] != yPrediction:
                    #print("run over") #flag
                    if y[index] > yPrediction:
                        update = self.learningRate
                    elif y[index] < yPrediction:
                        update = -self.learningRate

                    #current run is new longest
                    if currentLongestRun > self.longestRun:
                        currentBestWeights = self.weights
                        #print(currentBestWeights) #flag
                        self.longestRun = currentLongestRun
                        #print("new longest run") #flag

                    self.weights += update * currentSample
                    self.bias += update
                    
                    currentLongestRun = 0 #reset current run

                #print(currentBestWeights()) #flag
                self.pocket = currentBestWeights.copy() #update the pocket with the
                                                        #best weights                
                
    #Input: training samples
    def predict(self, X):
        linearOutput = np.dot(X, self.pocket) + self.bias #w.x + bias
        yPrediction = self.activationFunction(linearOutput)
        return yPrediction

#print("run the training file, not this one...\n") #reminder
