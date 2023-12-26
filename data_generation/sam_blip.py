import json
import os
from pycocotools import mask as mask_utils
from PIL import Image
import cv2
from operator import itemgetter
import numpy as np
import random
from lavis.models import load_model_and_preprocess
import torch
from tqdm import tqdm
from argparse import ArgumentParser
def generate_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return [r, g, b]

def get_json_files(folder_path):
    json_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def add_mask(img, mask):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    color = generate_random_color()
    converted_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    converted_mask[np.where(mask == 1)] = color
    converted_mask[np.where(mask == 0)] = [0, 0, 0]
    dst = cv2.addWeighted(img, 1, converted_mask, 0.2, 0)
    cv2.imwrite("./test.jpg",dst)
    return dst

def crop_image(img, mask, bbox):
    masked_image_array = np.array(img)
    masked_image_array[mask == 0] = [255, 255, 255]
    masked_image = Image.fromarray(masked_image_array)
    masked_image = masked_image.crop(bbox)
    return masked_image

def get_image_files(folder_path):  
    image_files = []  
    for root, dirs, files in os.walk(folder_path):  
        for file in files:  
            if file.endswith('.jpg') or file.endswith('.png'):  
                image_files.append(os.path.join(root, file))  
    return image_files

def save_json(json_list,save_path):
    with open(save_path, 'w') as file:
        json.dump(json_list, file, indent=4)

def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="./images")
    parser.add_argument("--json_folder", type=str, default="./masks")
    parser.add_argument("--output_path", type=str, default="./outputs/sam_blip2.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    return args
if __name__=="__main__":
    args = _get_args()
    images_path=get_image_files(args.image_folder)

    model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device = args.device)
    ann_list=[]
    for i in tqdm(range(len(images_path))):
        image_path = images_path[i]
        img=Image.open(image_path).convert('RGB')
        json_path = os.path.join(args.json_folder,image_path.split('/')[-1].split('.')[0])+'.json'
        width, height = img.size
        print(json_path)
        with open(json_path, "r") as f:
            data=json.load(f)
        if data==[]:
            continue
        num_obj = min(len(data), 20)
        sorted_data = sorted(data, key=itemgetter('area'), reverse=True)
        image_list = []
        norm_bbox_list = []
        for j in range(num_obj):
            mask_rle = sorted_data[j]['segmentation']
            mask = mask_utils.decode(mask_rle)
            bbox = sorted_data[j]['bbox']
            x1=bbox[0]
            x2=bbox[0]+bbox[2]
            y1=bbox[1]
            y2=bbox[1]+bbox[3]
            bbox = [x1,y1,x2,y2]
            norm_bbox = [round(float(x1)/width, 3), round(float(y1)/height, 3), round(float(x2)/width, 3), round(float(y2)/height, 3)]
            norm_bbox_list.append(norm_bbox)
            masked_image = crop_image(img, mask, bbox)
            image_list.append(vis_processors["eval"](masked_image).to(args.device))
        batch_img = torch.stack(image_list, dim=0, out=None)
        answer = model.generate({"image": batch_img}, num_beams=1, max_length=30)
        objects=[]
        for k in range(len(answer)):
            objects.append({"caption": answer[k], "box":norm_bbox_list[k]})
        ann_list.append({"img_id": image_path.split('/')[-1], "objects":objects})
    save_json(json_list=ann_list, save_path=args.output_path)