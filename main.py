import pickle
from urllib import request
from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
import joblib


import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


with open("Bad-Words.pkl" , 'rb') as f:
    myList = pickle.load(f)

# print(myList)

def word2vec(word):
    from collections import Counter
    from math import sqrt

    # count the characters in word
    cw = Counter(word)
    # precomputes a set of the different characters
    sw = set(cw)
    # precomputes the "length" of the word vector
    lw = sqrt(sum(c*c for c in cw.values()))

    # return a tuple
    return cw, sw, lw

def cosdis(v1, v2):
    # which characters are common to the two words?
    common = v1[1].intersection(v2[1])
    # by definition of cosine distance we have
    return sum(v1[0][ch]*v2[0][ch] for ch in common)/v1[2]/v2[2]


app = FastAPI()


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/checkStatus")
async def checkStatus(info : Request):
    req_info = await info.json()
    CurrString = dict(req_info)["comment"]

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(CurrString)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]

    filtered_sentence = []
  
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)


    status = False

    for i in filtered_sentence:
        for j in myList:
            s1 = word2vec(i)
            s2 = word2vec(j)
            res = cosdis(s1,s2)
            if res > 0.9:
                print(i,j)
                status = True
                break
    
    return {
        "status" : status
    }



