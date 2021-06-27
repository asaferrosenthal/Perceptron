"""
Aaron Safer-Rosenthal
20068164
17asr

Runs the program
"""

from Assignment1Classes import Perceptron
from Assignment1Classes import Pocket
import numpy as np
import pandas as pd

#to test the accuracy, relevance and recall
def testAccuracy(actual, predictions):
    actual = actual.tolist()
    accuracy = 0
    truePositives = 0
    falsePositives = 0
    falseNegatives = 0
    for i in range(len(actual)):
        if actual[i] == predictions[i]:
            accuracy += 1
            if predictions[i] == 1:
                truePositives += 1
        elif predictions[i] == 1:
            falsePositives += 1
        elif predictions[i] == 0:
            falseNegatives += 1
    accuracy /= len(actual)
    print("Accuracy: ", accuracyPrint(accuracy)) #flag
    relevance = truePositives / (truePositives + falsePositives)
    print("Relevance:", accuracyPrint(relevance))
    recall = truePositives / (truePositives + falseNegatives)
    print("Recall:", accuracyPrint(recall))

#purely for printing formatting
def accuracyPrint(accuracy):
    accuracy = accuracy*100
    accuracy = str(accuracy)[:5]
    accuracy += "%"
    return accuracy

#Input: fileName
#Output: panda data frame based on file
def getDataFrame(fileName):
    dataFrame = pd.read_csv(fileName, header=None) #reading the training data
    #dataFrame = dataFrame.sample(frac=1) #randomizing rows of training data
    return dataFrame

#Input: data frame
#Output: data frame of class names
def getClassNames(dataFrame):
    return dataFrame.iloc[:,4].values

#Input: data frame
#Output: data frame of features
def getFeatures(dataFrame):
    return dataFrame.iloc[:,0:4].values

#Input: data frame, only containing class names; target class name
#Output: data frame with binary class names (target=1, everything else=0)
def getTrainingData(classNames, className):
    testData = np.where(classNames == className, 1, 0)
    return testData

#flags
def dataFlags(setosaTrainingData, versicolorTrainingData, virginicaTrainingData):
    print("Training Data:\n")
    print("Setosa training data: " , setosaTrainingData, "\nVersicolor training\
 data: ", versicolorTrainingData, "\nVirginica training data: ",
          virginicaTrainingData) #flag

#printing my results!
def printResults(testSetosaData, setosaPrediction, testVersicolorData,
                 versicolorPrediction, testVirginicaData, virginicaPrediction):
    print("\nTesting results")

    print("\nSetosa")
    print("Actual: ", testSetosaData)
    print("Prediction: ", setosaPrediction)
    testAccuracy(testSetosaData, setosaPrediction) 

    print("\nVersicolor")
    print("Actual: ", testVersicolorData)
    print("Prediction: ", versicolorPrediction)
    testAccuracy(testVersicolorData, versicolorPrediction)


    print("\nVirginica")
    print("Actual: ", testVirginicaData)
    print("Prediction: ", virginicaPrediction)
    testAccuracy(testVirginicaData, virginicaPrediction) 

#Running the pocket perceptron training and testing
def PocketTrainingAndTesting(features, setosaTrainingData,
                             versicolorTrainingData, virginicaTrainingData,
                             testFeatures, testSetosaData, testVersicolorData,
                             testVirginicaData):

    print("\nPocket Perceptron\n")
    
    #training the Perceptron(s)
    trainSetosa = Pocket()
    print("Training Setosa...") #for printing ease of reading
    trainSetosa.fit(features, setosaTrainingData)

    trainVersicolor = Pocket()
    print("Training Versicolor...") #for printing ease of reading
    trainVersicolor.fit(features, versicolorTrainingData)

    trainVirginica = Pocket()
    print("Training Virginica...") #for printing ease of reading
    trainVirginica.fit(features, virginicaTrainingData)

    #predicting using the trained perceptrons
    setosaPrediction = trainSetosa.predict(testFeatures)
    versicolorPrediction = trainVersicolor.predict(testFeatures)
    virginicaPrediction = trainVirginica.predict(testFeatures)

    #printing results
    printResults(testSetosaData, setosaPrediction, testVersicolorData,
                 versicolorPrediction, testVirginicaData, virginicaPrediction)

#Running the w*cx perceptron training and testing
def PerceptronTrainingAndTesting(features, setosaTrainingData,
                                 versicolorTrainingData, virginicaTrainingData,
                                 testFeatures, testSetosaData,
                                 testVersicolorData, testVirginicaData):

    print("\nSimple Learning Feedback Perceptron\n")

    #training the Perceptron(s)
    trainSetosa = Perceptron()
    print("Training Setosa...") #for printing ease of reading
    trainSetosa.fit(features, setosaTrainingData)

    trainVersicolor = Perceptron()
    print("Training Versicolor...") #for printing ease of reading
    trainVersicolor.fit(features, versicolorTrainingData)

    trainVirginica = Perceptron()
    print("Training Virginica...") #for printing ease of reading
    trainVirginica.fit(features, virginicaTrainingData)
    
    #predicting using the trained perceptrons
    setosaPrediction = trainSetosa.predict(testFeatures)
    versicolorPrediction = trainVersicolor.predict(testFeatures)
    virginicaPrediction = trainVirginica.predict(testFeatures)

    #printing results
    printResults(testSetosaData, setosaPrediction, testVersicolorData,
                 versicolorPrediction, testVirginicaData, virginicaPrediction)

#main function which drives program
def main():

    #Importing and sorting the training data
    dataFrame = getDataFrame("iris_train.txt")
    #print(dataFrame) #flag
    classNames = getClassNames(dataFrame)
    features = getFeatures(dataFrame)

    #making list of binary values for each flower
    setosaTrainingData = getTrainingData(classNames, "Iris-setosa")
    versicolorTrainingData = getTrainingData(classNames, "Iris-versicolor")
    virginicaTrainingData = getTrainingData(classNames, "Iris-virginica")
    """dataFlags(setosaTrainingData, versicolorTrainingData,
    virginicaTrainingData) #flags"""

    #Importing and sorting the testing data
    dataFrame = getDataFrame("iris_test.txt")
    testClassNames = getClassNames(dataFrame)
    testFeatures = getFeatures(dataFrame)
    testSetosaData = getTrainingData(testClassNames, "Iris-setosa")
    testVersicolorData = getTrainingData(testClassNames, "Iris-versicolor")
    testVirginicaData = getTrainingData(testClassNames, "Iris-virginica")
    """dataFlags(setosaTrainingData, versicolorTrainingData,
    virginicaTrainingData) #flags"""
    
    PerceptronTrainingAndTesting(features, setosaTrainingData,
                                 versicolorTrainingData, virginicaTrainingData,
                                 testFeatures, testSetosaData,
                                 testVersicolorData, testVirginicaData)    
    
    PocketTrainingAndTesting(features, setosaTrainingData,
                             versicolorTrainingData, virginicaTrainingData,
                             testFeatures, testSetosaData, testVersicolorData,
                             testVirginicaData)

    #print("\ndone") #for printing ease of reading

main()
