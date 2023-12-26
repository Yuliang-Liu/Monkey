import argparse
import multiprocessing as mp
import os
import time
import cv2
from tqdm import tqdm
import sys
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
sys.path.insert(0, 'grit/third_party/CenterNet2/projects/CenterNet2/')
sys.path.append('./grit')
from centernet.config import add_centernet_config
from grit.config import add_grit_config
from grit.predictor import VisualizationDemo, BatchVisualizationDemo
from torch.utils.data import Dataset, DataLoader
from PIL import Image
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

# constants
WINDOW_NAME = "GRiT"

def norm_xy(xy, width, height):
    xy[0]=round(xy[0]/width,3)
    xy[2]=round(xy[2]/width,3)
    xy[1]=round(xy[1]/height,3)
    xy[3]=round(xy[3]/height,3)
    return xy
def dense_pred_to_normcaption(predictions, width, height):
    boxes = predictions["instances"].pred_boxes if predictions["instances"].has("pred_boxes") else None
    object_description = predictions["instances"].pred_object_descriptions.data
    objects = []
    for i in range(len(object_description)):
        xy = [a for a in boxes[i].tensor.cpu().detach().numpy()[0]]
        box = norm_xy(xy, width, height)
        objects.append({"caption":object_description[i],"box":box})
    return objects

def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    cfg.MODEL.DEVICE=args.device
    add_centernet_config(cfg)
    add_grit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    if args.test_task:
        cfg.MODEL.TEST_TASK = args.test_task
    cfg.MODEL.BEAM_SIZE = 1
    cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
    cfg.USE_ACT_CHECKPOINT = False
    cfg.freeze()
    return cfg


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="./grit/configs/GRiT_B_DenseCap_ObjectDet.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--cpu", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--test-task",
        type=str,
        default='DenseCap',
        help="Choose a task to have GRiT perform",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "./grit/model_weight/grit_b_densecap.pth"],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--image_folder", type=str, default="./images")
    parser.add_argument("--output_path", type=str, default="./outputs/grit.json")
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()
    return args

class lazydataset(Dataset):
    def __init__(self, data_path) -> None:
        super(lazydataset).__init__()
        self.image_paths = get_image_files(data_path)
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, i):
        image_path = self.image_paths[i]
        image = read_image(image_path, format="BGR")
        return {'image':image, 'img_id': image_path.split('/')[-1]}
def collate_fn(batch):
    image = [item['image'] for item in batch]
    img_id = [item['img_id'] for item in batch]
    return {'image':image, 'img_id':img_id}
if __name__ == "__main__":
    json_save=[]
    args = _get_args()
    cfg = setup_cfg(args)
    demo = BatchVisualizationDemo(cfg)
    dataset=lazydataset(args.image_folder)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,collate_fn=collate_fn)
    for batch in tqdm(dataloader):
        predictions = demo.run_on_images(batch['image'])
        for i in range(len(predictions)):
            height, width = batch['image'][i].shape[0], batch['image'][i].shape[1]
            objects = dense_pred_to_normcaption(predictions[i], width, height)
            json_save.append({"img_id":batch['img_id'][i], "objects":objects})
    save_json(json_save, args.output_path)