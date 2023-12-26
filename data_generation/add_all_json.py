import json
from argparse import ArgumentParser
def open_json(path):
    with open(path,"r") as f:
        data=json.load(f)
    return data
def save_json(json_list,save_path):
    with open(save_path, 'w') as file:
        json.dump(json_list, file, indent=4)
def caculate_IOU(box1,box2):
    Ax1=box1[0]
    Ay1=box1[1]
    Ax2=box1[2]
    Ay2=box1[3]

    Bx1=box2[0]
    By1=box2[1]
    Bx2=box2[2]
    By2=box2[3]

    Ix1 = max(Ax1, Bx1)
    Iy1 = max(Ay1, By1)
    Ix2 = min(Ax2, Bx2)
    Iy2 = min(Ay2, By2)
    IntersectionArea = max(0, Ix2 - Ix1 + 1) * max(0, Iy2 - Iy1 + 1)
    BoxAArea = (Ax2 - Ax1 + 1) * (Ay2 - Ay1 + 1)
    BoxBArea = (Bx2 - Bx1 + 1) * (By2 - By1 + 1)
    UnionArea = BoxAArea + BoxBArea - IntersectionArea
    IOU = IntersectionArea / UnionArea
    return IOU
def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--blip2_caption", type=str, default="./outputs/blip2_cap.json")
    parser.add_argument("--ori_caption", type=str, default=None)
    parser.add_argument("--grit", type=str, default="./outputs/grit_score.json")
    parser.add_argument("--ppocr", type=str, default="./outputs/ppocr.json")
    parser.add_argument("--sam_blip2", type=str, default="./outputs/sam_blip2_score.json")
    parser.add_argument("--output", type=str, default="./outputs/ann_all.json")
    args = parser.parse_args()
    return args
if __name__=="__main__":
    json_save = []
    args = _get_args()
    if args.ori_caption is not None:
        ori_cap = open_json(args.ori_caption)
    blip2_caption = open_json(args.blip2_caption)
    grit = open_json(args.grit)
    ppocr = open_json(args.ppocr)
    sam_blip2 = open_json(args.sam_blip2)
    
    blip2_caption_dict = {}
    grit_dict = {}
    ppocr_dict = {}
    sam_blip2_dict = {}
    for i in range(len(blip2_caption)):
        img_id = blip2_caption[i]['img_id']
        caption = blip2_caption[i]['blip2_caption']
        blip2_caption_dict[img_id]=caption
    for i in range(len(grit)):
        img_id = grit[i]['img_id']
        objects = grit[i]['objects']
        caption = ""
        for j in range(len(objects)):
            if objects[j]['score']>0.4:
                caption = caption + f"{objects[j]['caption']}: {objects[j]['box']}; "
        grit_dict[img_id] = caption
    
    for i in range(len(ppocr)):
        img_id = ppocr[i]['img_id']
        objects = ppocr[i]['objects']
        caption = ""
        for j in range(len(objects)):
            if objects[j]['score']>0.85:
                caption = caption + f"{objects[j]['caption']}: {objects[j]['box']}; "
        ppocr_dict[img_id] = caption
    
    for i in range(len(sam_blip2)):
        img_id = sam_blip2[i]['img_id']
        objects = sam_blip2[i]['objects']
        caption = ""
        iou_filter = {}
        for j in range(len(objects)):
            if objects[j]['score']>0.5:
                if iou_filter.get(objects[j]['caption'], 0)==0:
                    iou_filter[objects[j]['caption']]=objects[j]['box']
                    caption = caption + f"{objects[j]['caption']}: {objects[j]['box']}; "
                else:
                    if caculate_IOU(iou_filter[objects[j]['caption']], objects[j]['box'])<0.6:
                        caption = caption + f"{objects[j]['caption']}: {objects[j]['box']}; "
        sam_blip2_dict[img_id] = caption
    for key in blip2_caption_dict.keys():
        ocr_result = ppocr_dict.get(key,"")
        json_save.append({"img_id":key, "blip2cap":blip2_caption_dict[key], "grit":grit_dict[key], "ocr":ocr_result, "sam_blip":sam_blip2_dict[key]})
    save_json(json_save, args.output)