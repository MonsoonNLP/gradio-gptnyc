import gradio as gr
from transformers import AutoModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("monsoon-nlp/gpt-nyc")
model = AutoModel.from_pretrained("monsoon-nlp/gpt-nyc", pad_token_id=tokenizer.eos_token_id)

def hello(question, context):
    inp = question + ' - ' + context + ' %%'
    input_ids = tokenizer.encode(inp)
    beam_output = model.generate(input_ids, max_length=100, num_beams=5,
        no_repeat_ngram_size=2, early_stopping=True)
    output = tokenizer.decode(beam_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return ".".join(output.split(".")[:-1]) + "."

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
