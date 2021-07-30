import gradio as gr
from transformers import AutoModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("monsoon-nlp/gpt-nyc")
model = AutoModel.from_pretrained("monsoon-nlp/gpt-nyc", pad_token_id=tokenizer.eos_token_id)

def hello(question, context):
    inp = question + ' - ' + context + ' %%'
    input_ids = tokenizer.encode(inp)
    output = model.generate(input_ids, max_length=50, early_stopping=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

io = gr.Interface(fn=hello,
    inputs=[
        gr.inputs.Textbox(label="Question"),
        gr.inputs.Textbox(lines=3, label="Context"),
    ],
    outputs=gr.outputs.Textbox(label="Reply"),
    verbose=True,
    title='GPT-NYC Input',
    description='Learn more at https://monsoonnlp.com/gpt-nyc/index.html',
    #thumbnail='https://github.com/MonsoonNLP/gradio-gptnyc',
    analytics_enabled=True)

io.launch(debug=True)
