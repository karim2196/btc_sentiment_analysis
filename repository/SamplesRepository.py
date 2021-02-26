from nltk.corpus import twitter_samples


def getNegativeTweets():
    return twitter_samples.strings('negative_tweets.json')

def getPositiveTweets():
    return twitter_samples.strings('positive_tweets.json')
     