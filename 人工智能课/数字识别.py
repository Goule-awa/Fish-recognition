# -*- coding: utf-8 -*-
import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier


class kNNClassifier(object):
    def __init__(self, k):
        self.k = k
        self.train_feature = None
        self.train_label = None

    def fit(self, feature, label):
        self.train_feature = feature
        self.train_label = label

    def predict(self, feature):
        predictions = []
        for test_instance in feature:
            distances = []
            for train_instance, train_label in zip(self.train_feature, self.train_label):
                distance = np.linalg.norm(test_instance - train_instance)
                distances.append((distance, train_label))
            nearest_neighbors = sorted(distances)[:self.k]
            labels = [neighbor[1] for neighbor in nearest_neighbors]
            prediction = max(set(labels), key=labels.count)
            predictions.append(prediction)
        return predictions


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def parseFiles():
    trainingLabels = []
    trainingFilelist = listdir('digits/trainingDigits')
    m = len(trainingFilelist)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFilelist[i]
        classNumber = int(fileNameStr.split('_')[0])
        trainingLabels.append(classNumber)
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % fileNameStr)

    testFilelist = listdir('digits/testDigits')
    mTest = len(testFilelist)
    testMat = np.zeros((mTest, 1024))
    testLabels = []
    for i in range(mTest):
        fileNameStr = testFilelist[i]
        classNumber = int(fileNameStr.split('_')[0])
        testLabels.append(classNumber)
        testMat[i, :] = img2vector('digits/testDigits/%s' % fileNameStr)

    return trainingMat, trainingLabels, testMat, testLabels


def evaluate_predictions(predictions, testLabels):
    errorCount = 0
    for i in range(len(predictions)):
        result = predictions[i]
        if result != testLabels[i]:
            errorCount += 1
    errorRate = errorCount / len(predictions)
    return errorCount, errorRate


def handwritingClassTest(k, trainingMat, trainingLabels, testMat, testLabels):
    clf = kNNClassifier(k)
    clf.fit(trainingMat, trainingLabels)
    predict = clf.predict(testMat)

    errorCount, errorRate = evaluate_predictions(predict, testLabels)
    print("Using custom kNN classifier:")
    print("The total number of errors is: %d" % errorCount)
    print("The total error rate is: %f" % errorRate)


def handwritingClassTest_sklearn(k, trainingMat, trainingLabels, testMat, testLabels):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(trainingMat, trainingLabels)
    predict = clf.predict(testMat)

    errorCount, errorRate = evaluate_predictions(predict, testLabels)
    print("Using sklearn kNN classifier:")
    print("The total number of errors is: %d" % errorCount)
    print("The total error rate is: %f" % errorRate)


if __name__ == '__main__':
    trainingMat, trainingLabels, testMat, testLabels = parseFiles()
    k = 3
    handwritingClassTest(k, trainingMat, trainingLabels, testMat, testLabels)
    handwritingClassTest_sklearn(k, trainingMat, trainingLabels, testMat, testLabels)