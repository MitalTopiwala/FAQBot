import os
import discord
import re
import random
from dotenv import load_dotenv
from myModel import *

load_dotenv()
#NOTE: saved token as environment variable, update token in .env file
TOKEN = os.getenv('DISCORD_TOKEN')
client = discord.Client()

tags = []
questions = [] # [ [where is the syllabus, is the syllbus posted, syllubus],  [when is A3 due, what is the due date of A3] ]
responses = [] # [*link to syllabus, september 29th]
data_storage = [tags, questions, responses]

MODEL = myModel()

@client.event
async def on_ready():
    print(f'{client.user.name} has connected to Discord!')

@client.event
async def on_message(message):
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
        MODEL.make_training_set(data_storage)
        MODEL.train_model()
        await message.channel.send("FAQ Added")
    else:
        if MODEL.model != None:
            probs = MODEL.predict_class(message.content)
            tag = probs[0]['tag']
            prob = probs[0]['probability']
            for i in range(len(data_storage[0])):
                if(data_storage[0][i] == tag):
                    if (float(prob) > 0.995):
                        response = data_storage[2][i]
                        await message.channel.send(response)
                        break
        

client.run(TOKEN)
