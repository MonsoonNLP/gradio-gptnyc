import gradio as gr
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("monsoon-nlp/gpt-nyc")
model = GPT2LMHeadModel.from_pretrained("monsoon-nlp/gpt-nyc", pad_token_id=tokenizer.eos_token_id)

def hello(question, context):
    inp = question + ' - ' + context + ' %%'
    input_ids = torch.tensor([tokenizer.encode(inp)])
    output = model.generate(input_ids, max_length=50, early_stopping=True)
    resp = tokenizer.decode(output[0], skip_special_tokens=True)
    if '%%' in resp:
        resp = resp[resp.index('%%') + 2 : ]
    return resp

io = gr.Interface(fn=hello,
    inputs=[
        gr.inputs.Textbox(label="Question"),
        gr.inputs.Textbox(lines=3, label="Context"),
    ],
    outputs=gr.outputs.Textbox(label="Reply"),
    verbose=True,
    title='GPT-NYC Input',
    description='Learn more at https://huggingface.co/monsoon-nlp/gpt-nyc',
    #thumbnail='https://github.com/MonsoonNLP/gradio-gptnyc',
    analytics_enabled=True)

io.launch(debug=True)
