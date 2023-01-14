"""
@Desc:
@Reference:
- chatgpt api
https://gitcode.net/mirrors/acheong08/ChatGPT?utm_source=csdn_github_accelerator
@Notes:

"""

import logging


from tqdm import tqdm
from revChatGPT.ChatGPT import Chatbot

def chatgpt_batch_generate():
    chatbot = Chatbot({
      "session_token": "<YOUR_TOKEN>"
    }, conversation_id=None, parent_id=None) # You can start a custom conversation

    response = chatbot.ask("Prompt", conversation_id=None, parent_id=None)

if __name__ == '__main__':
    pass



