import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
# things we need for Tensorflow
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import pandas as pd
import pickle
import numpy as np
import tflearn
import tensorflow as tf
import random
import pickle
import json
from keras.models import load_model
from flask import Flask, render_template, request, make_response, redirect, url_for, session, abort
app = Flask(__name__)
app.config['SECRET_KEY'] = 'redsfsfsfsfis'

with open('intents.json') as json_data:
    intents = json.load(json_data)

words = []
classes = []
documents = []
ignore_words = ['?']
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remove duplicates
classes = sorted(list(set(classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)


@app.route('/', methods = ['GET','POST'])
def index_show():
    return render_template('chatui.html')
	
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

ERROR_THRESHOLD = 0.25
def classify(sentence):
    # generate probabilities from the model
    print("Wordsssss;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;",words)
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])
    model_file = load_model('my_model.h5')
    results = model_file.predict([input_data])[0]
    
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

# import our chat-bot intents file
@app.route('/chat_data', methods = ['GET','POST'])
def chat():
    if request.method == 'POST':
        msg = request.form['data']
        print("KKKKKKKKKKKKKKKKKKKKKK",msg)
        results = classify(msg)
        # if we have a classification then find the matching intent tag
        if results:
            # loop as long as there are matches to process
            while results:
                for i in intents['intents']:
                    # find a tag matching the first result
                    if i['tag'] == results[0][0]:
                        # a random response from the intent
                        msg = (random.choice(i['responses']))
                        return msg

                results.pop(0)
    return render_template('chatui.html')

if __name__ == '__main__':
    app.run(host='localhost', port=8182, debug=True)