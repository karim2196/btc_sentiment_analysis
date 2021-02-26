from ML.nltk import BayesClassifier
from nltk.corpus import stopwords
import pymongo
import tracemalloc
import repository.NewDataRepository as NewDataRepository
import repository.SamplesRepository as SampleRepository
import repository.KaggleRepository as KaggleRepository
import dataParser.DataParser as DataParser
from ML.nltk import BayesClassifier


def initializeMongoDB():
    client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
    db = client["sentiment_data"]
    return db


def getMemoryUsage():
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

def main():
    db = initializeMongoDB()
    tweetsTrainingCollection = db["tweetsTrainData"]
    fetchDataFrom= "Kaggle"
    data = []

    tracemalloc.start()
    getMemoryUsage()


    if (fetchDataFrom == "db"):
        tokensForModel = DataParser.parseTweets(True, NewDataRepository.getData(),"")
        NewDataRepository.saveTrainingAndTestData(tokensForModel,db,"False")
        positiveTrainingTweets = NewDataRepository.getPositiveDBTweets(tweetsTrainingCollection)
        negativeTrainingTweets = NewDataRepository.getNegativeDBTweets(tweetsTrainingCollection)
        data = DataParser.prepareDataForTraining(True,positiveTrainingTweets,negativeTrainingTweets)
   
    if (fetchDataFrom == "twitter_samples" ):
        #positive_tweets = twitter_samples.strings('positive_tweets.json')
        #negative_tweets = twitter_samples.strings('negative_tweets.json')
        positiveTrainingTweets = DataParser.parseTweets(False,SampleRepository.getPositiveTweets(),"positive")
        negativeTrainingTweets =  DataParser.parseTweets(False,SampleRepository.getNegativeTweets(),"negative")

        data = DataParser.prepareDataForTraining(False,positiveTrainingTweets,negativeTrainingTweets)

    if (fetchDataFrom == "Kaggle"):
        tweets = KaggleRepository.getExtensiveCSVTweetsForTraining()
        negativeTweets = []
        poistiveTweets = []
        for tweet in tweets:
            if (tweet[0] == 0):
                negativeTweets.append(tweet[1])
            if (tweet[0] == 4):
                poistiveTweets.append(tweet[1])
        del tweets
        negativeTrainingTweets = DataParser.parseTweets(True,negativeTweets[:10000],"negative")
        positiveTrainingTweets = DataParser.parseTweets(True,poistiveTweets[:10000],"positive")
        getMemoryUsage()
        del negativeTweets
        del poistiveTweets
        #gc.collect()
        getMemoryUsage()
        data = DataParser.prepareDataForTraining(False,positiveTrainingTweets,negativeTrainingTweets)
        del positiveTrainingTweets
        del negativeTrainingTweets
        getMemoryUsage()
    percentageOfTrainingData = 0.7
    trainingData = data[:(int(len(data)*percentageOfTrainingData))]
    testData = data[(int(len(data)*percentageOfTrainingData)):]
    del data
    getMemoryUsage()
    #KaggleRepository.saveTweetsInFile(trainingData,"Positive")
    #KaggleRepository.saveTweetsInFile(trainingData,"Negative")

    bayesClassifier = BayesClassifier(0)
    bayesClassifier.train(trainingData)
    bayesClassifier.setAccuracy(testData)
    print("accuracy : " , bayesClassifier.getAccuracy())
    #del trainingData
    #del testData
    #getMemoryUsage()
    #gc.collect
    #customTweet = 'Thank you for sending my baggage to CityX and flying me to CityY at the same time... Brilliant service. #thanksGenericAirline'
    customTweet = 'With this said, I think we are going to the moon'
    customTokens = DataParser.removeNoise(customTweet,stopwords.words('english'))
    print(bayesClassifier.avalueTweet(customTokens))
    customTweet = 'With this said, I think we are going all the way down'
    customTokens = DataParser.removeNoise(customTweet,stopwords.words('english'))
    print(bayesClassifier.avalueTweet(customTokens))

    #print(classifier.show_most_informative_features())

if __name__=="__main__":
    main()