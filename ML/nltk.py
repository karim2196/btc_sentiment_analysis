from nltk import FreqDist,classify,NaiveBayesClassifier, parse


class BayesClassifier:
    
    def __init__(self,accuracy):
        self.accuracy = accuracy
        self.classifier = None

    def train(self,trainingData):
        self.classifier = NaiveBayesClassifier.train(trainingData)


    def setAccuracy(self,testData):
        self.accuracy = classify.accuracy(self.classifier,testData)


    def getAccuracy(self):
        return self.accuracy


    def avalueTweet(self,tweetTokens):
        return self.classifier.classify(dict([token, True] for token in tweetTokens))