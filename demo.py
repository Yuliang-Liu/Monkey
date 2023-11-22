
from argparse import ArgumentParser
from pathlib import Path

import copy
import gradio as gr
import os
import re
import secrets
import tempfile

from PIL import Image
from monkey_model.modeling_monkey import MonkeyLMHeadModel
from monkey_model.tokenization_qwen import QWenTokenizer
from monkey_model.configuration_monkey import MonkeyConfig

import shutil
from pathlib import Path
import json
DEFAULT_CKPT_PATH = '/home/zhangli/demo/'
BOX_TAG_PATTERN = r"<box>([\s\S]*?)</box>"
PUNCTUATION = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
title_markdown = ("""
# Welcome to Monkey

Hello! I'm Monkey, a Large Language and Vision Assistant. Before talking to me, please read the **Operation Guide** and **Terms of Use**.
你好！我是Monkey，一个大型语言和视觉助理。在与我交谈之前，请阅读**操作指南**和**使用条款**。
## Operation Guide 操作指南

Click the **Upload** button to upload an image. Then, you can get Monkey's answer in two ways:点击**Upload**上传图像。你可以通过两种方式得到Monkey的回答：
 - Click the **Generate** and Monkey will generate a description of the image. 点击**Generate**，Monkey将生成图像的描述。
 - Enter the question in the dialog box, click the **Submit**, and Monkey will answer the question based on the image. 在对话框中输入问题，点击**Submit**，Monkey会根据图片回答问题。
 - Click **Clear History** to clear the current image and Q&A content.点击**Clear History**，清除当前图片和问答内容。
> Note: Monkey does not have a multi-round dialogue function. Perhaps we will further develop its capabilities in the future. 注意：Monkey没有多轮对话功能，或许我们在未来会进一步开发它的能力。
> Monkey支持中文,但使用英文提问会比使用中文效果明显好.""")

policy_markdown = ("""
## Terms of Use

By using this service, users are required to agree to the following terms:

 - Monkey is for research use only and unauthorized commercial use is prohibited. For any query, please contact the author.
 - Monkey's generation capabilities are limited, so we recommend that users do not rely entirely on its answers.
 - Monkey's security measures are limited, so we cannot guarantee that the output is completely appropriate. We strongly recommend that users do not intentionally guide Monkey to generate harmful content, including hate speech, discrimination, violence, pornography, deception, etc.

""")
def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")

    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--server-port", type=int, default=8000,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="127.0.0.1",
                        help="Demo server name.")

    args = parser.parse_args()
    return args


def _load_model_tokenizer(args):
    tokenizer = QWenTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True)

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "cuda"

    model = MonkeyLMHeadModel.from_pretrained(
        args.checkpoint_path,
        device_map=device_map,
        trust_remote_code=True,
    ).eval()
    # model.generation_config = GenerationConfig.from_pretrained(
    #     args.checkpoint_path, trust_remote_code=True, resume_download=True,
    # )
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id
    return model, tokenizer


def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def _launch_demo(args, model, tokenizer):
    def predict(_chatbot, task_history):
        chat_query = _chatbot[-1][0]
        query = task_history[-1][0]
        question =  _parse_text(query)
        print("User: " + _parse_text(query))
        full_response = ""


        img_path = _chatbot[0][0][0]
        try:
            Image.open(img_path)
        except:
            response = "Please upload a picture."
            _chatbot[-1] = (_parse_text(chat_query), response)
            full_response = _parse_text(response)

            task_history[-1] = (query, full_response)
            print("Monkey: " + _parse_text(full_response))
            return _chatbot

        query = f'<img>{img_path}</img> {question} Answer: '
        print(query)

        input_ids = tokenizer(query, return_tensors='pt', padding='longest')
        attention_mask = input_ids.attention_mask
        input_ids = input_ids.input_ids
        
        pred = model.generate(
            input_ids=input_ids.cuda(),
            attention_mask=attention_mask.cuda(),
            do_sample=False,
            num_beams=1,
            max_new_tokens=512,
            min_new_tokens=1,
            length_penalty=3,
            num_return_sequences=1,
            output_hidden_states=True,
            use_cache=True,
            pad_token_id=tokenizer.eod_id,
            eos_token_id=tokenizer.eod_id,
            )
        response = tokenizer.decode(pred[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()

        _chatbot[-1] = (_parse_text(chat_query), response)
        full_response = _parse_text(response)
        task_history[-1] = (query, full_response)
        print("Monkey: " + _parse_text(full_response))
        return _chatbot
    
    def caption(_chatbot, task_history):

        
        query = "Generate the detailed caption in English:"
        chat_query = "Generate the detailed caption in English:"
        question =  _parse_text(query)
        print("User: " + _parse_text(query))

        full_response = ""
        
        try:
            img_path = _chatbot[0][0][0]
            Image.open(img_path)
        except:
            response = "Please upload a picture."

            _chatbot.append((None, response))
            full_response = _parse_text(response)

            task_history.append((None, full_response))
            print("Monkey: " + _parse_text(full_response))
            return _chatbot
        img_path = _chatbot[0][0][0]
        query = f'<img>{img_path}</img> {chat_query} '
        print(query)
        input_ids = tokenizer(query, return_tensors='pt', padding='longest')
        attention_mask = input_ids.attention_mask
        input_ids = input_ids.input_ids
        

        pred = model.generate(
            input_ids=input_ids.cuda(),
            attention_mask=attention_mask.cuda(),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=250,
            min_new_tokens=1,
            length_penalty=3,
            num_return_sequences=1,
            output_hidden_states=True,
            use_cache=True,
            pad_token_id=tokenizer.eod_id,
            eos_token_id=tokenizer.eod_id,
            )
        response = tokenizer.decode(pred[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()

        
        _chatbot.append((None, response))
        full_response = _parse_text(response)

        task_history.append((None, full_response))
        print("Monkey: " + _parse_text(full_response))
        return _chatbot
   


    def add_text(history, task_history, text):
        task_text = text
        if len(text) >= 2 and text[-1] in PUNCTUATION and text[-2] not in PUNCTUATION:
            task_text = text[:-1]
        history = history + [(_parse_text(text), None)]
        task_history = task_history + [(task_text, None)]
        print(history, task_history, text)
        return history, task_history, ""

    def add_file(history, task_history, file):
        history =  [((file.name,), None)]
        task_history = [((file.name,), None)]
        print(history, task_history, file)
        return history, task_history

    def reset_user_input():
        return gr.update(value="")

    def reset_state(task_history):
        task_history.clear()
        return []


    with gr.Blocks() as demo:
        gr.Markdown(title_markdown)

        chatbot = gr.Chatbot(label='Monkey', elem_classes="control-height", height=600,avatar_images=("https://ooo.0x0.ooo/2023/11/09/OehsLx.png","https://ooo.0x0.ooo/2023/11/09/OehGBC.png"),layout="bubble",bubble_full_width=False,show_copy_button=True)
        query = gr.Textbox(lines=1, label='Input')
        task_history = gr.State([])

        with gr.Row():
            empty_bin = gr.Button("Clear History (清空)")
            submit_btn = gr.Button("Submit (提问)")
            
            generate_btn_en = gr.Button("Generate")
            addfile_btn = gr.UploadButton("Upload (上传图片)", file_types=["image"])

        submit_btn.click(add_text, [chatbot, task_history, query], [chatbot, task_history]).then(
            predict, [chatbot, task_history], [chatbot], show_progress=True
        )
        generate_btn_en.click(caption, [chatbot, task_history], [chatbot], show_progress=True)
        
        submit_btn.click(reset_user_input, [], [query])
        empty_bin.click(reset_state, [task_history], [chatbot], show_progress=True)
        
        addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn], [chatbot, task_history], show_progress=True,scroll_to_output=True)
        


        gr.Markdown(policy_markdown)
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7681
    )


def main():
    args = _get_args()

    model, tokenizer = _load_model_tokenizer(args)
    _launch_demo(args, model, tokenizer)


if __name__ == '__main__':
    main()
