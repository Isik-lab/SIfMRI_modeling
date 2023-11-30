from transformers import AutoTokenizer, AutoModel
import torch

model = AutoModel.from_pretrained('openai-gpt')
tokenizer = AutoTokenizer.from_pretrained('openai-gpt')
# tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
# model = OpenAIGPTModel.from_pretrained("openai-gpt")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)

print('done')