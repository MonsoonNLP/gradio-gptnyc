import gradio as gr
from transformers import AutoModel, AutoTokenizer

tokenizer = GPT2Tokenizer.from_pretrained("monsoon-nlp/gpt-nyc")
model = AutoModel.from_pretrained("monsoon-nlp/gpt-nyc", pad_token_id=tokenizer.eos_token_id)

def hello(input):
    input_ids = tokenizer.encode(inp)
    beam_output = model.generate(input_ids, max_length=100, num_beams=5,
        no_repeat_ngram_size=2, early_stopping=True)
    output = tokenizer.decode(beam_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return ".".join(output.split(".")[:-1]) + "."

io = gr.Interface(fn=hello,
    inputs='text',
    outputs='text',
    verbose=True,
    title='GPT-NYC Input',
    description='Format chars: Question? - Details %%',
    thumbnail='https://github.com/MonsoonNLP/gradio-gptnyc',
    analytics_enabled=True)

io.launch(debug=True)
