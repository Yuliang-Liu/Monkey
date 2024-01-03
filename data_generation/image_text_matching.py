from typing import Any
import torch
from PIL import Image
from argparse import ArgumentParser
from lavis.models import load_model_and_preprocess
import os
import json
from tqdm import tqdm
import re
def save_json(json_list,save_path):
    with open(save_path, 'w') as file:
        json.dump(json_list, file, indent=4)
class blip_matching:
    def __init__(self, name, device) -> None:
        if "blip2" in name:
            model, vis_processors, text_processors = load_model_and_preprocess(name, "pretrain", device=device, is_eval=True)
        else:
            model, vis_processors, text_processors = load_model_and_preprocess(name, "large", device=device, is_eval=True)
        self.model=model
        self.vis_processors=vis_processors
        self.text_processors=text_processors
        self.device=device
    def match_score(self, img_src, caption, crop_box=None):
        raw_image = Image.open(img_src).convert("RGB")
        w,h=raw_image.size
        if crop_box is not None:
            raw_image = raw_image.crop((int(crop_box[0]*w), int(crop_box[1]*h), int(crop_box[2]*w), int(crop_box[3]*h)))
        img = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        txt = self.text_processors["eval"](caption)
        itm_output = self.model({"image": img, "text_input": txt}, match_head="itm")
        itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
        return round(itm_scores[:, 1].item(), 3)
    
def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="./images")
    parser.add_argument("--ann_path", type=str, default="./outputs/sam_blip2.json")
    parser.add_argument("--output_path", type=str, default="./outputs/sam_blip2_score.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    return args
if __name__=="__main__":
    args = _get_args()
    # blip_image_text_matching or blip2_image_text_matching
    model = blip_matching(name="blip2_image_text_matching", device=args.device)
    with open(args.ann_path, 'r') as f:
        data = json.load(f)
    for i in tqdm(range(len(data))):
        img_id = data[i]["img_id"]
        path=os.path.join(args.image_folder, img_id)
        for j in range(len(data[i]['objects'])):
           score = model.match_score(img_src=path,caption=data[i]['objects'][j]['caption'],crop_box=data[i]['objects'][j]['box'])
           data[i]['objects'][j]['score']=score
    save_json(json_list=data, save_path=args.output_path)