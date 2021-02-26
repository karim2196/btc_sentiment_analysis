from nltk.corpus import twitter_samples, stopwords
import btc_tweet_getter
from nltk.tokenize import TweetTokenizer
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist,classify,NaiveBayesClassifier, parse
import re,string
import pymongo
import random
import pandas as pd
import gc
import tracemalloc

def getData():
    tweet = btc_tweet_getter.getTweets()
    return tweet


def tokenizeTweet(tweets):
    return TweetTokenizer().tokenize(tweets)
    

def removeNoise(tokens,stopWords=()):
    cleaned_tokens = []
    for token,tag in pos_tag(tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
        '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
        pos = lemmatize_sentence(tag)
        token = WordNetLemmatizer().lemmatize(token,pos)
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stopWords:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


def lemmatize_sentence(tag):
    #lemmatized_sentence = []
    #for word,tag in posTaggedTweet:
    if tag.startswith('NN'):
        pos = 'n'
    elif tag.startswith('VB'):
        pos = 'v'
    else:
        pos = 'a'
        
    return pos

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(allCleanedTokens):
    for tweet_tokens in allCleanedTokens:
        yield dict([token, True] for token in tweet_tokens)


def initializeMongoDB():
    client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
    db = client["sentiment_data"]
    return db


def saveTrainingAndTestData(tokensTrainingData,db, isTraining):
    tweetsTrainingCollection = db["tweetsTrainData"]
    modelToSave = [{"tokenTweet":"test1" , "label":" "}]
    if (isTraining == True):
        for tokenDictionary,label in tokensTrainingData:
            parsedTokenDictionary = { k.replace('.',',') if "." in k else k:v for k,v in tokenDictionary.items()}
            modelToSave.append({"tokenTweet":parsedTokenDictionary,"label":label})
    else:
        for tokenDictionary in tokensTrainingData:
            parsedTokenDictionary = { k.replace('.',',') if "." in k else k:v for k,v in tokenDictionary.items()}
            modelToSave.append({"tokenTweet":parsedTokenDictionary})
    tweetsTrainingCollection.insert_many(modelToSave)


def parseTweets(isNewData,tweets,posOrNeg):
    print("inside parse")
    allCleanedTokens = []
    if (isNewData == True):
        for tweet in tweets:
            tokenizedTweet = tokenizeTweet(tweet)
            cleanedTokens = removeNoise(tokenizedTweet, stopwords.words('english'))
            allCleanedTokens.append(cleanedTokens)
            #wordsAllTweets = get_all_words(allCleanedTokens)
            #print(FreqDist(wordsAllTweets).most_common(25))
        tokensForModel = get_tweets_for_model(allCleanedTokens)

    else:
        tweets = 'positive_tweets.json' if (posOrNeg == "positive") else 'negative_tweets.json'
        tweet_tokens = twitter_samples.tokenized(tweets)
        for tokens in tweet_tokens:
            allCleanedTokens.append(removeNoise(tokens, stopwords.words('english')))

        tokensForModel = get_tweets_for_model(allCleanedTokens)
        #wordsAllTweets = get_all_words(positive_cleaned_tokens_list)
        #print(FreqDist(wordsAllTweets).most_common(25))
    return tokensForModel
    
        
    


def saveTokenTweets(newTokenTweets,collection,db):
     db[collection].insert_many(newTokenTweets)

def getPositiveDBTweets(tweetsTrainingCollection):
    positiveTrainingTweets = tweetsTrainingCollection.find({'label':'Positive'})
    data = []
    for tweet in positiveTrainingTweets:
        data.append(tweet)
    return data


def getNegativeDBTweets(tweetsTrainingCollection):
    data = []
    negativeTrainingTweets = tweetsTrainingCollection.find({'label':'Negative'})
    for tweet in negativeTrainingTweets:
        data.append(tweet)
    return data
    

def getNegativeTweets():
    return twitter_samples.strings('negative_tweets.json')

def getPositiveTweets():
    return twitter_samples.strings('positive_tweets.json')
     


def prepareDataForTraining(isNewData,positiveTweets,negativeTweets):
    dataset = []
    
    if(isNewData == True):
        data = []
        for tweet in negativeTweets:
            data.append(tweet)

        for tweet in positiveTweets:
            data.append(tweet)
        
        random.shuffle(data)

        for tweet in data:
            element = (tweet['tokenTweet'],tweet['label'])
            dataset.append(element)

    else:
        positive_dataset = [(tweet_dict, "Positive") for tweet_dict in positiveTweets]
        negative_dataset = [(tweet_dict, "Negative") for tweet_dict in negativeTweets]
        dataset = positive_dataset + negative_dataset
        random.shuffle(dataset)

    return dataset

def getExtensiveCSVTweetsForTraining():
    data = pd.read_csv("/Users/karim/Desktop/karim/projects/btc_sentiment_analysis/training.1600000.processed.noemoticon.csv", encoding = "ISO-8859-1" , engine="python")
    data.columns = ["label", "time", "date", "query", "username", "text"]
    data = data[['label','text']]
    #label 0 is negative, 2 is neutral, 4 is positive
    return data.values.tolist()

def saveTweetsInFile(trainingData,posOrNeg):
    tokens = []
    saveTweets = []
   #positiveTrainingData = [item for item in trainingData if item[1]=="Positive"]
    parsedTrainingData = [item for item in trainingData if item[1]==posOrNeg]
    for element in parsedTrainingData:
        for token in element[0].keys():
            tokens.append(token)
        saveTweets.append("-|-".join(tokens))


    with open('/Users/karim/Desktop/karim/projects/btc_sentiment_analysis/btc_sentiment_analysis/ ' + posOrNeg + 'TrainingData.txt', mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n \n \n \n '.join(saveTweets))


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
        tokensForModel = parseTweets(True,getData(),"")
        saveTrainingAndTestData(tokensForModel,db,"False")
        negativeTrainingTweets = tweetsTrainingCollection.find({'label':'Negative'})
        positiveTrainingTweets = tweetsTrainingCollection.find({'label':'Positive'})
        data = prepareDataForTraining(True,positiveTrainingTweets,negativeTrainingTweets)
   
    if (fetchDataFrom == "twitter_samples" ):
        #positive_tweets = twitter_samples.strings('positive_tweets.json')
        #negative_tweets = twitter_samples.strings('negative_tweets.json')
        positiveTrainingTweets = parseTweets(False,getPositiveTweets(),"positive")
        negativeTrainingTweets =  parseTweets(False,getNegativeTweets(),"negative")

        data = prepareDataForTraining(False,positiveTrainingTweets,negativeTrainingTweets)

    if (fetchDataFrom == "Kaggle"):
        tweets = getExtensiveCSVTweetsForTraining()
        negativeTweets = []
        poistiveTweets = []
        for tweet in tweets:
            if (tweet[0] == 0):
                negativeTweets.append(tweet[1])
            if (tweet[0] == 4):
                poistiveTweets.append(tweet[1])
        del tweets
        negativeTrainingTweets = parseTweets(True,negativeTweets[:10000],"negative")
        positiveTrainingTweets = parseTweets(True,poistiveTweets[:10000],"positive")
        getMemoryUsage()
        del negativeTweets
        del poistiveTweets
        #gc.collect()
        getMemoryUsage()
        data = prepareDataForTraining(False,positiveTrainingTweets,negativeTrainingTweets)
        del positiveTrainingTweets
        del negativeTrainingTweets
        getMemoryUsage()
    percentageOfTrainingData = 0.7
    trainingData = data[:(int(len(data)*percentageOfTrainingData))]
    testData = data[(int(len(data)*percentageOfTrainingData)):]
    del data
    getMemoryUsage()
    #saveTweetsInFile(trainingData,"Positive")
    #saveTweetsInFile(trainingData,"Negative")

 
    classifier = NaiveBayesClassifier.train(trainingData)
    accuracy = classify.accuracy(classifier,testData)
    del trainingData
    del testData
    getMemoryUsage()
    print("Accuracy : " , accuracy)
    #gc.collect
    customTweet = 'Thank you for sending my baggage to CityX and flying me to CityY at the same time... Brilliant service. #thanksGenericAirline'
    
    customTokens = removeNoise(customTweet,stopwords.words('english'))
    print(classifier.classify(dict([token, True] for token in customTokens)))

    #print(classifier.show_most_informative_features())

if __name__=="__main__":
    main()