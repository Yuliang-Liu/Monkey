import json
import openai
from argparse import ArgumentParser
#You can replace this step with Open Source LMM
openai.api_base = ""
openai.api_key = ''
def save_json(json_list,save_path):
    with open(save_path, 'w') as file:
        json.dump(json_list, file, indent=4)
def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--ann_path", type=str, default="./outputs/ann_all.json")
    parser.add_argument("--output_path", type=str, default="./outputs/detailed_caption.json")
    args = parser.parse_args()
    return args

def get_question(ann):
    sentence1 = "I want you to act as an intelligent image captioner. You should generate a descriptive, coherent and logical description of the image based on the given descriptions from different people for the same image. The position is represented by normalized top-left and bottom-right coordinates."
    sentence2 = "\n Overall Image Caption: "
    sentece3 = "\n Dense Caption1 (Region Description: Area Location): "
    sentece4 = "\n Dense Caption2 (Region Description: Area Location): "
    sentece5 = "\n Texts in the image: "
    sentence6 = "\n There are some rules for your response: Provide context of the image. \n Merging the descriptions of the same object at the same position and the texts belonging to the same sentence. \n Show main objects with their attributes (e.g. position, color, shape). \n Show relative position between main objects. \n Less than 6 sentences. \n Do not show any numbers or coordinates. \n Do not describe any individual letter."
    
    if ann['ocr']!="":
        question = f"{sentence1}{sentence2}{ann['blip2cap']}{sentece3}{ann['grit']}{sentece4}{ann['sam_blip']}{sentece5}{ann['ocr']}{sentence6}"
    else:
        question = f"{sentence1}{sentence2}{ann['blip2cap']}{sentece3}{ann['grit']}{sentece4}{ann['sam_blip']}{sentence6}"
    return question

if __name__=="__main__":
    args = _get_args()
    json_save=[]
    with open(args.ann_path,"r") as f:
        data = json.load(f)
    
    for i in range(len(data)):
        question = get_question(data[i])
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", #gpt-4-1106-preview
            messages=[
            {"role": "user", "content": question},
            ]
        )
        answer = response['choices'][0]['message']['content']
        json_save.append({"img_id":data[i]['img_id'],"detailed_caption":answer})
    save_json(json_save, args.output_path)