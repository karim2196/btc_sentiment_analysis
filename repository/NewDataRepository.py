
import btc_tweet_getter 



def getData():
    tweet = btc_tweet_getter.getTweets()
    return tweet


def saveTrainingAndTestData(tokensTrainingData,db):
    tweetsTrainingCollection = db["tweetsTrainData"]
    modelToSave = [{"tokenTweet":"test1" , "label":" "}]
    for tokenDictionary,label in tokensTrainingData:
        parsedTokenDictionary = { k.replace('.',',') if "." in k else k:v for k,v in tokenDictionary.items()}
        modelToSave.append({"tokenTweet":parsedTokenDictionary,"label":label})
    
    tweetsTrainingCollection.insert_many(modelToSave)


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