import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig, pipeline, AutoTokenizer, AutoModelForCausalLM
from nltk import sent_tokenize
import os
import re
from nltk import sent_tokenize

class ChatBot:
    def __init__(self, base_model, base_model_path, peft_adapter_path=None, max_tokenizer_len=124,
                 max_new_tokens=50, temperature=0.4, top_p=0.9):
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                        bnb_4bit_compute_dtype=torch.bfloat16)
        
        self.model = AutoModelForCausalLM.from_pretrained(base_model, cache_dir = base_model_path,
                                                          quantization_config = bnb_config, device_map = 'auto',
                                                          torch_dtype = torch.bfloat16, attn_implementation='sdpa')
        
        self.tokenizer = AutoTokenizer.from_pretrained(peft_adapter_path)

        self.model = PeftModel.from_pretrained(self.model, peft_adapter_path)

        self.task_pipeline = pipeline(task='text-generation', model=self.model, tokenizer=self.tokenizer)

        self.terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids('<|eot_id|>')]

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokenizer_len = max_tokenizer_len


    def chat(self, message, history):
        if len(history) > 8:
            history = history[-8:]
        
        messages = []
        messages.append({'role': 'system', 'content': 'You are Naruto from the anime "Naruto". Your responses should reflect his personality and speech patterns.'})
        messages.extend(history)
        # for prev_message in history:
        #     messages.append({'role': 'user', 'content': prev_message[0]})
        #     messages.append({'role': 'assistant', 'content': prev_message[1]})
        
        messages.append({'role': 'user', 'content': message})

        result = self.task_pipeline(messages, max_new_tokens=50, eos_token_id = self.terminators,
                                    temperature = self.temperature, top_p = 0.9, do_sample=True)
        
        response = result[0]['generated_text'][-1]['content']
        response = ' '.join(sent_tokenize(response)[:-1])
        response = re.sub('\*.*?\*', "", response)
        return response
