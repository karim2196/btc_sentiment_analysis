import requests
import os
import json

# To set your enviornment variables in your terminal run the following line:
#export 'BEARER_TOKEN'='<your_bearer_token>'

def auth():
    token_exists = 'BEARER_TOKEN' in os.environ
    #print('token exists? ', token_exists)
    return "AAAAAAAAAAAAAAAAAAAAAHXlMQEAAAAArGFiJ28oRcty%2Fpg%2FW1edd6dgSAo%3DyqhgQnuohiIb9ZuakfHvAZllZHX68eVhtyNOEdsm8FJZfIbnu2"
    #return os.environ.get("BEARER_TOKEN")


def create_url():
    url = "https://api.twitter.com/2/tweets/search/recent?max_results=100&query=Bitcoin%20-is:retweet%20#Bitcoin&tweet.fields=created_at,author_id"
    return url


def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers


def connect_to_endpoint(url, headers):
    response = requests.request("GET", url, headers=headers)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

tweets = []

def setTweets(tweet):
    global tweets
    tweets.append(tweet)

def getTweets():
    return tweets

def main():
    bearer_token = auth()
    url = create_url()
    headers = create_headers(bearer_token)
    json_response = connect_to_endpoint(url, headers)
    for tweet in json_response['data']:
        setTweets(tweet['text'])

    


if __name__ == "__main__":
    main()
if __name__ == "btc_tweet_getter":
    print("nothing to get")
    #main()
