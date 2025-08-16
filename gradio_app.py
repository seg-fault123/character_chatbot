import gradio as gr

from chatbot import ChatBot

import os
from dotenv import load_dotenv
import pathlib

load_dotenv()


base_model = 'meta-llama/Meta-Llama-3-8B-Instruct'
base_model_path = os.getenv('cache_dir')
peft_adapter_path = pathlib.Path(r"D:\Naruto_Project\models\result\peft")
# peft_adapter_path = os.getenv('save_dir')
max_tokenizer_len = 124
max_new_tokens=50
temperature=0.4
top_p=0.9
# print(peft_adapter_path)

chatbot = ChatBot(base_model, base_model_path, peft_adapter_path,
                  max_tokenizer_len, max_new_tokens, temperature, top_p)
# chatbot = None

def chat_with_naruto(message, history):
    message = message.strip()
    response = chatbot.chat(message, history)
    return response.strip()


def main():
    with gr.Blocks() as iface:
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Chat with Naruto</h1>")
                gr.ChatInterface(chat_with_naruto, type='messages')
    
    iface.launch(share=True)

if __name__ == '__main__':
    main()
