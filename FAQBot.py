import os
import discord
import re
import pickle
import random
from dotenv import load_dotenv
import nltk
#nltk.download('punkt')
#nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD


load_dotenv()
#NOTE: saved token as environment variable, update token in .env file
TOKEN = os.getenv('DISCORD_TOKEN')

client = discord.Client()

tags = []
questions = [] # [ [where is the syllabus, is the syllbus posted, syllubus],  [when is A3 due, what is the due date of A3] ]
responses = [] # [*link to syllabus, september 29th]

data_storage = [tags, questions, responses]

words = []
classes = []
documents = []
ignore_words = ['!', '?'] #not needed when lemmatizing

MODEL = None

def make_training_set():
    global words
    global classes
    global documents
    #tokenize questions into words, and update documents and classes accordingly
    for j in range(len(data_storage[0])):
        for i in range(len(data_storage[1][j])):
            word = nltk.word_tokenize(data_storage[1][j][i])
            print(word)
            words.extend(word)
            documents.append((word, data_storage[0][j]))
            if data_storage[0][j] not in classes:
                classes.append(data_storage[0][j])
    
    #Now lets lemmatize 
    words = [WordNetLemmatizer().lemmatize(word.lower()) for word in words if word not in ignore_words]
    words = sorted(list(set(words)))
    # sort classes
    classes = sorted(list(set(classes)))
    # documents = combination between patterns and intents
    print (len(documents), "documents")
    # classes = intents
    print (len(classes), "classes", classes)
    # words = all words, vocabulary
    print (len(words), "unique lemmatized words", words)

    pickle.dump(words,open('words.pkl','wb'))
    pickle.dump(classes,open('classes.pkl','wb'))

    training = []
    output_empty = [0] * len(classes)

    # training set, pool of words for each sentence
    for doc in documents:
        pool = []
        # tokenized words for the pattern
        pattern_words = doc[0]
        # get base word
        pattern_words = [WordNetLemmatizer().lemmatize(word.lower()) for word in pattern_words]

        for word in words:
            #if word match found in current pattern, append 1 to pool array
            pool.append(1) if word in pattern_words else pool.append(0)
            # output is a '0' for each tag and '1' for current tag (for each pattern)
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1
            training.append([pool, output_row])
            

    #TODO: Fix line below 
    #training = np.array(random.shuffle(training))
    print(training)
    training_x = list(np.array(training).T[0]) #patterns
    training_y = list(np.array(training).T[1]) #intents
    print("Training data created")
    return training_x, training_y

def train_model(x, y):
    global MODEL
    # 3 later model: first layer 128 neurons, second layer 64 neurons, and 3rd output layer contains number of neurons 
    # equal to number of intents to predict output intent with softmax
    MODEL = Sequential()
    MODEL.add(Dense(128, input_shape=(len(x[0]),), activation='relu'))
    MODEL.add(Dropout(0.5))
    MODEL.add(Dense(64, activation='relu'))
    MODEL.add(Dropout(0.5))
    MODEL.add(Dense(len(y[0]), activation='softmax'))

    # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    MODEL.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    #fitting and saving the model 
    m = MODEL.fit(np.array(x), np.array(y), epochs=200, batch_size=5, verbose=1)

    print("model created")
    return m

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [WordNetLemmatizer().lemmatize(word.lower()) for word in sentence_words]
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
                    print("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"tag": classes[r[0]], "probability": str(r[1])})
    return return_list


@client.event
async def on_ready():
    print(f'{client.user.name} has connected to Discord!')

@client.event
async def on_member_join(member):
    await member.create_dm()
    await member.dm_channel.send(f'Hi {member.name}, welcome to Mitals Discord server!')

@client.event
async def on_message(message):
    global MODEL
    #to prevent recursive case
    if message.author == client.user:
        return

    if re.search("Tag##.+ ##Question##.+##Response##.+", message.content.strip()) != None:
        message_segments = re.split("##", message.content)
        current = -1
        for segment in message_segments:
            if segment.strip() == "Tag":
                current = 0
            elif segment.strip() == "Question":
                current = 1
                data_storage[current].append([])
            elif segment.strip() == "Response":
                current = 2
            else:
                if current == 1:
                    data_storage[current][-1].append(segment.strip())
                else:
                    data_storage[current].append(segment.strip())
        x_train, y_train = make_training_set()
        m_history = train_model(x_train, y_train)
        await message.channel.send("FAQ Added")
    else:
        probs = predict_class(message.content, MODEL)
        tag = probs[0]['tag']
        prob = probs[0]['probability']
        for i in range(len(data_storage[0])):
            if(data_storage[0][i] == tag):
                if (float(prob) > 0.995):
                    response = data_storage[2][i]
                    await message.channel.send(response)
                    break
        

client.run(TOKEN)
