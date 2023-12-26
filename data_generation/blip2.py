import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from argparse import ArgumentParser
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.utils as vutils
import json
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
    parser.add_argument("--output_path", type=str, default="./outputs/blip2_cap.json")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    return args
class lazydataset(Dataset):
    def __init__(self, data_path, processor) -> None:
        super(lazydataset).__init__()
        self.image_paths = get_image_files(data_path)
        self.processor = processor
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, i):
        image_path = self.image_paths[i]
        raw_image = Image.open(image_path).convert('RGB')
        image = self.processor["eval"](raw_image)
        return {'image':image, 'img_id': image_path.split('/')[-1]}
def collate_fn(batch):
    image = [item['image'].squeeze(0) for item in batch]
    image = torch.stack(image)
    img_id = [item['img_id'] for item in batch]
    return {'image':image, 'img_id':img_id}

if __name__=="__main__":
    json_save = []
    args = _get_args()
    device = args.device
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device)
    dataset = lazydataset(data_path=args.image_folder, processor = vis_processors)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    for batch in tqdm(dataloader):
        image = batch['image'].to(device)
        captions = model.generate({"image": image})
        img_id = batch['img_id']
        for i in range(len(img_id)):
            json_save.append({'img_id':img_id[i],'blip2_caption':captions[i]})
    save_json(json_save, args.output_path)
