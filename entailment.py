import json
import requests
import logging

#from memoized import Memoize
from nltk import tokenize

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

def get_list(sen):
    """Converts the sentence into a List such that it can be used as an POST
    request json blob data.

    Args:
        sen : a string

    Returns:
        res: a list of list containing words and tags.
    """
    res = list()
    count = 0   
    for word in tokenize.word_tokenize(sen):
        l = [word, "Any", count]
        count += 1
        res.append(l)
    return res


#@Memoize
def get_ai2_textual_entailment(t, h):
    """Returns the output of POST request to AI2 textual entailment service

    Args:
        t, h : text and hypothesis (two strings)

    Returns:
        req : A text version of json response.
    """
    text = get_list(t)
    # print (text)
    hypothesis = get_list(h)
    # print(hypothesis)
    data = {"text": text, "hypothesis": hypothesis}
    headers = {'Content-type': 'application/json', 'Accept': 'application/json'}
    url = 'http://localhost:8191/api/entails'
    # url='https://api.github.com/events'
    req = requests.post(url, headers=headers, data=json.dumps(data))
    return req.json()


def main():
    text = input("Enter the text: ")
    hypothesis = input("Enter the hypothesis: ")

    print ("Response: ")
    print (get_ai2_textual_entailment(text, hypothesis))


if __name__ == "__main__":
    main()
