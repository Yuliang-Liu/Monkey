import re
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from monkey_model.modeling_textmonkey import TextMonkeyLMHeadModel
from monkey_model.tokenization_qwen import QWenTokenizer
from monkey_model.configuration_monkey import MonkeyConfig
from argparse import ArgumentParser

def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=None,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--share", action="store_true", default=True,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--server-port", type=int, default=7680,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="127.0.0.1",
                        help="Demo server name.")

    args = parser.parse_args()
    return args
args = _get_args()
checkpoint_path = args.checkpoint_path
device_map = "cuda"
# Create model
config = MonkeyConfig.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
    )
model = TextMonkeyLMHeadModel.from_pretrained(checkpoint_path,
    config=config,
    device_map=device_map, trust_remote_code=True).eval()
tokenizer = QWenTokenizer.from_pretrained(checkpoint_path,
                                            trust_remote_code=True)
tokenizer.padding_side = 'left'
tokenizer.pad_token_id = tokenizer.eod_id
tokenizer.IMG_TOKEN_SPAN = config.visual["n_queries"]

title = "TextMonkey : An OCR-Free Large Multimodal Model for Understanding Document"

description = """
<font size=4>
Welcome to TextMonkey

Hello! I'm TextMonkey, a Large Language and Vision Assistant developed by HUST VLRLab and KingSoft.

You can click on the examples below the demo to display them.

## Example prompts for different tasks
You need to replace "Question" with your question.

1.**Read All Text:** Read all the text in the image.

2.**Text Spotting:** OCR with grounding:

3.**Position of Text:** &lt;ref&gt;"Question"&lt;/ref&gt;

4.**VQA:** "Question" Answer:

5.**VQA with Grounding:** "Question" Provide the location coordinates of the answer when answering the question.

6.**Output Json**: Convert the chart in this image to json format. Answer:(Convert the document in this image to json format. Answer:)(Convert the table in this image to json format. Answer:)
</font>
"""

def inference(input_str, input_image):    
    input_str = f"<img>{input_image}</img> {input_str}"
    input_ids = tokenizer(input_str, return_tensors='pt', padding='longest')

    attention_mask = input_ids.attention_mask
    input_ids = input_ids.input_ids
    
    pred = model.generate(
    input_ids=input_ids.cuda(),
    attention_mask=attention_mask.cuda(),
    do_sample=False,
    num_beams=1,
    max_new_tokens=2048,
    min_new_tokens=1,
    length_penalty=1,
    num_return_sequences=1,
    output_hidden_states=True,
    use_cache=True,
    pad_token_id=tokenizer.eod_id,
    eos_token_id=tokenizer.eod_id,
    )
    response = tokenizer.decode(pred[0][input_ids.size(1):].cpu(), skip_special_tokens=False).strip()
    image = Image.open(input_image).convert("RGB").resize((1000,1000))
    font = ImageFont.truetype('NimbusRoman-Regular.otf', 22)
    bboxes = re.findall(r'<box>(.*?)</box>', response, re.DOTALL)
    refs = re.findall(r'<ref>(.*?)</ref>', response, re.DOTALL)
    if len(refs)!=0:
        num = min(len(bboxes), len(refs))
    else:
        num = len(bboxes)
    for box_id in range(num):
        bbox = bboxes[box_id]
        matches = re.findall( r"\((\d+),(\d+)\)", bbox)
        draw = ImageDraw.Draw(image)
        point_x = (int(matches[0][0])+int(matches[1][0]))/2
        point_y = (int(matches[0][1])+int(matches[1][1]))/2
        point_size = 8
        point_bbox = (point_x - point_size, point_y - point_size, point_x + point_size, point_y + point_size)
        draw.ellipse(point_bbox, fill=(255, 0, 0))
        if len(refs)!=0:
            text = refs[box_id]
            text_width, text_height = font.getsize(text)
            draw.text((point_x-text_width//2, point_y+8), text, font=font, fill=(255, 0, 0))
    response = tokenizer.decode(pred[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()
    output_str = response
    output_image = image
    print(f"{input_str}   {response}")
    
    return output_image, output_str

demo = gr.Interface(
    inference,
    inputs=[
        gr.Textbox(lines=1, placeholder=None, label="Question"),
        gr.Image(type="filepath", label="Input Image"),
    ],
    outputs=[
        gr.Image(type="pil", label="Output Image"),
        gr.Textbox(lines=1, placeholder=None, label="TextMonkey's response"),
    ],
    title=title,
    description=description,
    allow_flagging="auto",
)

demo.queue()
demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share
    )
