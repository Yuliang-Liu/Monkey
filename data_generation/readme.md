The requested images should be placed in the "./images" directory, and the results will be stored in the "./outputs" directory.

Follow https://github.com/salesforce/LAVIS, https://github.com/facebookresearch/segment-anything, https://github.com/JialianW/GRiT and https://github.com/PaddlePaddle/PaddleOCR.git to prepare the enverionment.

Download GRiT(Dense Captioning on VG Dataset) and place it under ./grit/model_weight.

Download SAM and place it under ./model_weight.


Generation Steps:
1. Generate global description for each image. 
```python blip2.py```

2. Use the Grit model to generate dense captions for each image.
```python grit.py```

3. Generate segmentation maps for each image using the SAM model, and save the segmentation maps in the "./masks" directory.
```python amg.py --checkpoint ./model_weight/<pth name>  --model-type <model_type>  --input ./images  --output ./data_gen/masks --convert-to-rle```

4. Generate corresponding descriptions for the segmentation maps. 
```python sam_blip.py```

5.  Compute the similarity score.
```python image_text_matching.py --ann_path ./outputs/sam_blip2.json --output_path ./outputs/sam_blip2_score.json```

6. Compute the similarity score.
```python image_text_matching.py --ann_path ./outputs/grit.json --output_path ./outputs/grit_score.json```

7. Use ppocr to detect text in images.   
```python ocr_ppocr.py```

8. Integrate the generated annotations into ann_all.json.  
```python add_all_json.py```  

9. Use ChatGPT API to generate the final detailed description and save it in ./outputs/ann_all.json.   
```python chatgpt.py```       
