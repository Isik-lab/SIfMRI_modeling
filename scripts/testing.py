from transformers import AutoTokenizer, AutoModel
import torch
from deepjuice.systemops import sysreport

model = AutoModel.from_pretrained('openai-gpt')
tokenizer = AutoTokenizer.from_pretrained('openai-gpt')
# tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
# model = OpenAIGPTModel.from_pretrained("openai-gpt")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)

print('cpu report:')
print(sysreport.get_cpu_info())
print('\n gpu report:')
print(sysreport.get_gpu_info())
print('\n cuda report:')
print(sysreport.count_cuda_devices())

print('done')