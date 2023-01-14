"""
@Desc:
@Reference:
- chatgpt api
https://blog.csdn.net/qq_42902997/article/details/128274658
https://gitcode.net/mirrors/acheong08/ChatGPT?utm_source=csdn_github_accelerator
@Notes:

"""

import logging


from tqdm import tqdm
from revChatGPT.ChatGPT import Chatbot


def chatgpt_batch_generate(query: str):
    chatbot = Chatbot({
      "session_token": "https%3A%2F%2Fchat.openai.com%2Fchat"
    }, conversation_id=None, parent_id=None) # You can start a custom conversation

    response = chatbot.ask(query, conversation_id=None, parent_id=None)
    return response

if __name__ == '__main__':
    print(chatgpt_batch_generate("hellow world."))



