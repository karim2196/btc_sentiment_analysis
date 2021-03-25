
import random
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
import re,string
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import twitter_samples


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



def get_tweets_for_model(allCleanedTokens):
    for tweet_tokens in allCleanedTokens:
        yield dict([token, True] for token in tweet_tokens)



def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token
