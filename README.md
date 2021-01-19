# FAQBot

A bot that asnwers FAQs.

## Intro

Whether its a student trying to locate the syllabus, or a novice gammer inquiring about game mechanics, in almost discord server, there are select questions that are repeatedly asked. I created the FAQBot to eliminate the need for users to repeatedly answer the same questions, and can instead rely on a bot to do it for them.

## Requirements

- Install [Python 3.6+](https://www.python.org/downloads/)
- Install [Discord.py](https://pypi.org/project/discord.py/)
- Install [NLTK](https://www.nltk.org/data.html)
- Install [numpy](https://numpy.org/install/)
- A [Discord API Key](https://discord.com/developers/docs/intro)

## Installation

```
# git clone into your root folder
git clone https://github.com/MitalTopiwala/FAQBot.git

#Navigate to the FAQBot folder, 
#and create a file named .env, which contains the following line:
#DISCORD_TOKEN=[TOKEN]
#where [TOKEN] is replaced with your Discord API token/key

# Finally, run FAQBot.py
python FAQBot.py

#Now your bot is ready to be used!

```

## Usage

In order for a member to add a new FAQ, they must send a message in the chat that is formatted in a manner similar to: 

Tag## A3 due date ##Question## Did he post the A3 due date? ## Whats the due date for A3?##Whats the deadline for A3?##Response## September 26th

There must be exactly 1 tag and 1 response, and there could be multiple variations of the question.

The user is encouraged to provide as many variations of the question as possible, as it improves the accuracy of the bot.
