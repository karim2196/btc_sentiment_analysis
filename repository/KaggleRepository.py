import pandas as pd


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

