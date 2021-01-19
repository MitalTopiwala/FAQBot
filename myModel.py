import re
import random
import nltk
#nltk.download('punkt')
#nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from FAQBotHelper import *

ignore_words = ['!', '?'] #not needed when lemmatizing

class myModel():
    def __init__(self):
        self.words = []#will hold all vocabulary words
        self.classes = []#will hold tags
        self.documents = []#combination between patterns and tags
        self.x_training = None
        self.y_training = None
        self.model = None

    def make_training_set(self, data_storage):
        #tokenize questions into words, and update documents and classes accordingly
        for j in range(len(data_storage[0])):
            for i in range(len(data_storage[1][j])):
                word = nltk.word_tokenize(data_storage[1][j][i])
                print(word)
                self.words.extend(word)
                self.documents.append((word, data_storage[0][j]))
                if data_storage[0][j] not in self.classes:
                    self.classes.append(data_storage[0][j])
    
        #Now lets lemmatize 
        self.words = [WordNetLemmatizer().lemmatize(word.lower()) for word in self.words if word not in ignore_words]
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))

        training = []
        output_empty = [0] * len(self.classes)
        # training set, pool of words for each sentence
        for doc in self.documents:
            pool = []
            # tokenized words for the pattern
            pattern_words = doc[0]
            # get base word
            pattern_words = [WordNetLemmatizer().lemmatize(word.lower()) for word in pattern_words]
            for word in self.words:
                #if word match found in current pattern, append 1 to pool array
                pool.append(1) if word in pattern_words else pool.append(0)
                # output is a '0' for each tag and '1' for current tag (for each pattern)
                output_row = list(output_empty)
                output_row[self.classes.index(doc[1])] = 1
                training.append([pool, output_row])

        #TODO: Fix line below 
        #training = np.array(random.shuffle(training))
        training_x = list(np.array(training).T[0]) #patterns
        training_y = list(np.array(training).T[1]) #intents
        print("Training data created")
        self.x_training = training_x
        self.y_training = training_y

    def train_model(self):
        # 3 later model: first layer 128 neurons, second layer 64 neurons, and 3rd output layer contains number of neurons 
        # equal to number of intents to predict output intent with softmax
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(len(self.x_training[0]),), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(self.y_training[0]), activation='softmax'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        self.model.fit(np.array(self.x_training), np.array(self.y_training), epochs=200, batch_size=5, verbose=1)
        print("model created")

    def predict_class(self, sentence):
        # filter out predictions below a threshold
        p = bow(sentence, self.words,show_details=False) #hepler function that 
        res = self.model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"tag": self.classes[r[0]], "probability": str(r[1])})
        return return_list
