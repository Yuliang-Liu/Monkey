from PIL import Image
import os
import json
from tqdm import tqdm
from argparse import ArgumentParser
from paddleocr import PaddleOCR
def save_json(json_list,save_path):
    with open(save_path, 'w') as file:
        json.dump(json_list, file, indent=4)
def get_image_files(folder_path):  
    image_files = []  
    for root, dirs, files in os.walk(folder_path):  
        for file in files:  
            if file.endswith('.jpg') or file.endswith('.png'):  
                image_files.append(os.path.join(root, file))  
    return image_files
def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="./images")
    parser.add_argument("--output_path", type=str, default="./outputs/ppocr.json")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    return args 

if __name__=="__main__":
    args = _get_args()
    ocr_model=PaddleOCR(use_angle_cls=True, lang="en") 
    images = get_image_files(args.image_folder)
    json_save=[]
    for i in tqdm(range(len(images))):
        print(f"num{i}")
        img_src=images[i]
        result=ocr_model.ocr(img_src, cls=True)[0]
        if result==None:
            print("no text")
            continue
        img = Image.open(img_src)
        width, height = img.size
        objects=[]
        for j in range(len(result)):
            box_xy = result[j][0]
            norm_box_xy=[round(float(box_xy[0][0])/width, 3), round(float(box_xy[0][1])/height, 3), round(float(box_xy[2][0])/width, 3), round(float(box_xy[2][1])/height, 3)]
            text= result[j][1][0]
            score = result[j][1][1]
            objects.append({"caption": text, "box":norm_box_xy, "score":score})
        json_save.append({"img_id":img_src.split('/')[-1], "objects":objects})
    save_json(json_save, args.output_path)