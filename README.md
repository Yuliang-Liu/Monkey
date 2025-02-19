<p align="center">
    <img src="https://v1.ax1x.com/2024/08/13/7GXwAh.png" width="500" style="margin-bottom: 0.2;"/>
<p>

<h3 align="center"> <a href="https://arxiv.org/abs/2311.06607">Monkey: Image Resolution and Text Label Are Important Things for Large Multi-modal Models</a></h3>
<h2></h2>

<h5 align="center"> Please give us a star ⭐ for the latest update.  </h5>

<h5 align="center">

 
[![arXiv](https://img.shields.io/badge/Arxiv-2311.06607-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2311.06607) 
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/Monkey/blob/main/LICENSE) 
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/Monkey?color=critical&label=Issues)](https://github.com/Yuliang-Liu/Monkey/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/Monkey?color=success&label=Issues)](https://github.com/Yuliang-Liu/Monkey/issues?q=is%3Aissue+is%3Aclosed)  <br>
</h5>




> [**Monkey: Image Resolution and Text Label Are Important Things for Large Multi-modal Models**](https://arxiv.org/abs/2311.06607)<br>
> Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, Xiang Bai <br>
[![Source_code](https://img.shields.io/badge/Code-Available-white)](README.md)
[![Detailed Caption](https://img.shields.io/badge/Detailed_Caption-yellow)](http://huggingface.co/datasets/echo840/Detailed_Caption)
[![Model Weight](https://img.shields.io/badge/Model_Weight-gray)](http://huggingface.co/echo840/Monkey)
[![Model Weight in Wisemodel](https://img.shields.io/badge/Model_Weight_in_Wisemodel-gray)](https://www.wisemodel.cn/models/HUST-VLRLab/Monkey/)
[![Demo in Wisemodel](https://img.shields.io/badge/Demo_in_Wisemodel-blue)](https://wisemodel.cn/space/gradio/huakeMonkey)



> [**TextMonkey: An OCR-Free Large Multimodal Model for Understanding Document**](https://arxiv.org/abs/2403.04473)<br>
> Yuliang Liu, Biao Yang, Qiang Liu, Zhang Li, Zhiyin Ma, Shuo Zhang, Xiang Bai <br>
[![arXiv](https://img.shields.io/badge/Arxiv-2403.04473-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2403.04473) 
[![Source_code](https://img.shields.io/badge/Code-Available-white)](monkey_model/text_monkey/README.md)
[![Data](https://img.shields.io/badge/Data-yellow)](https://huggingface.co/datasets/MelosY/TextMonkey_Data/tree/main)
[![Model Weight](https://img.shields.io/badge/Model_Weight-gray)](https://www.modelscope.cn/models/lvskiller/TextMonkey)

> [**Mini-Monkey: Multi-Scale Adaptive Cropping for Multimodal Large Language Models**](https://arxiv.org/pdf/2408.02034)<br>
> Mingxin Huang, Yuliang Liu, Dingkang Liang, Lianwen Jin, Xiang Bai <br>
[![arXiv](https://img.shields.io/badge/Arxiv-2408.02034-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2408.02034)
[![Source_code](https://img.shields.io/badge/Code-Available-white)](project/mini_monkey)
[![Model Weight](https://img.shields.io/badge/Model_Weight-gray)](https://www.wisemodel.cn/models/HUST-VLRLab/Mini-Monkey)
[![Model Weight](https://img.shields.io/badge/Model_Weight-gray)](https://huggingface.co/mx262/MiniMokney)
[![Model Weight in Wisemodel](https://img.shields.io/badge/Model_Weight-gray)](https://www.wisemodel.cn/models/HUST-VLRLab/Mini-Monkey)


## News 
* ```2025.1.23 ``` 🚀 Mini-Monkey is accepted by ICLR 2025. 
* ```2024.11.27``` 🚀 Thanks to [Fahd Mirza](https://www.youtube.com/@fahdmirza) for sharing a [video](https://www.youtube.com/watch?v=NY3YzrhD4EM) on how to run Monkey.
* ```2024.8.13 ``` 🚀 Sourced code for [Mini-Monkey](project/mini_monkey) is released.
* ```2024.8.6  ``` 🚀 We release the paper [Mini-Monkey](https://arxiv.org/abs/2408.02034).
* ```2024.4.13 ``` 🚀 Sourced code for [TextMonkey](monkey_model/text_monkey/README.md) is released.
* ```2024.4.5  ``` 🚀 Monkey is nominated as CVPR 2024 Highlight paper.
* ```2024.3.8  ``` 🚀 We release the paper [TextMonkey](https://arxiv.org/abs/2403.04473).
* ```2024.2.27 ``` 🚀 Monkey is accepted by CVPR 2024. 
* ```2024.1.3  ``` 🚀 Release the basic data generation pipeline. [Data Generation](./data_generation)
* ```2023.11.06``` 🚀 We release the paper [Monkey](https://arxiv.org/abs/2311.06607).

## 🐳 Model Zoo

Monkey-Chat
| Model|Language Model|Transformers(HF) |MMBench-Test|CCBench|MME|SeedBench_IMG|MathVista-MiniTest|HallusionBench-Avg|AI2D Test|OCRBench|
|---------------|---------|-----------------------------------------|---|---|---|---|---|---|---|---|
|Monkey-Chat|Qwev-7B|[🤗echo840/Monkey-Chat](https://huggingface.co/echo840/Monkey-Chat)|72.4|48|1887.4|68.9|34.8|39.3|68.5|534|
|Mini-Monkey|internlm2-chat-1_8b|[Mini-Monkey](https://huggingface.co/mx262/MiniMokney)|---|75.5|1881.9|71.3|47.3|38.7|74.7|802|


## Environment

```python
conda create -n monkey python=3.9
conda activate monkey
git clone https://github.com/Yuliang-Liu/Monkey.git
cd ./Monkey
pip install -r requirements.txt
```
You can download the corresponding version of flash_attention from https://github.com/Dao-AILab/flash-attention/releases/ and use the following code to install:
```python
pip install flash_attn-2.3.5+cu117torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl --no-build-isolation
```


## Train

We also offer Monkey's model definition and training code, which you can explore above. You can execute the training code through executing `finetune_ds_debug.sh` for Monkey and `finetune_textmonkey.sh` for TextMonkey.

The json file used for Monkey training can be downloaded at [Link](https://drive.google.com/file/d/18z_uQTe8Jq61V5rgHtxOt85uKBodbvw1/view?usp=sharing).


## Inference
Run the inference code for Monkey and Monkey-Chat:
```
python ./inference.py --model_path MODEL_PATH  --image_path IMAGE_PATH  --question "YOUR_QUESTION"
```


## Demo

Demo is fast and easy to use. Simply uploading an image from your desktop or phone, or capture one directly. 
[Demo_chat](http://vlrlab-monkey.xyz:7681) is also launched as an upgraded version of the original demo to deliver an enhanced interactive experience.

We also provide the source code and the model weight for the original demo, allowing you to customize certain parameters for a more unique experience. The specific operations are as follows:
 1. Make sure you have configured the [environment](#environment).
 2. You can choose to use the demo offline or online:
- **Offline:** 
	- Download the [Model Weight](http://huggingface.co/echo840/Monkey). 
	- Modify `DEFAULT_CKPT_PATH="pathto/Monkey"` in the `demo.py` file to your model weight path. 
	- Run the demo using the following command: 
	```
	python demo.py
	```
- **Online:** 
	- Run the demo and download model weights online with the following command: 
	```
	python demo.py -c echo840/Monkey 
	```

For TextMonkey you can download the model weight from [Model Weight](https://www.modelscope.cn/models/lvskiller/TextMonkey)  and run the demo code:
``` python
python demo_textmonkey.py -c model_path
```

Before 14/11/2023, we have observed that for some random pictures Monkey can achieve more accurate results than GPT4V.  
<br>
<p align="center">
    <img src="https://v1.ax1x.com/2024/04/13/7yS2yq.jpg" width="666"/>
<p>
<br>

Before 31/1/2024, Monkey-chat achieved the fifth rank in the Multimodal Model category on [OpenCompass](https://opencompass.org.cn/home). 
<br>
<p align="center">
    <img src="https://v1.ax1x.com/2024/04/13/7yShXL.jpg" width="666"/>
<p>
<br>

 
## Dataset
You can download the training and testing data used by monkey from [Monkey_Data](https://huggingface.co/datasets/echo840/Monkey_Data).

The json file used for Monkey training can be downloaded at [Link](https://drive.google.com/file/d/18z_uQTe8Jq61V5rgHtxOt85uKBodbvw1/view?usp=sharing).

The data from our multi-level description generation method is now open-sourced and available for download at [Link](https://huggingface.co/datasets/echo840/Detailed_Caption). We already upload the images used in multi-level description. Examples:

<br>
<p align="center">
    <img src="https://v1.ax1x.com/2024/04/13/7yS6Ss.jpg" width="666"/>
<p>
<br>
	
You can download train images of Monkey from [Train](https://pan.baidu.com/s/1svSjXTxWpI-3boALgSeLlw). Extraction code: 4hdh

You can download test images and jsonls of Monkey from [Test](https://pan.baidu.com/s/1ABrQKeE9QBeKvtGzXfM8Eg). Extraction code: 5h71

The images are from CC3M, COCO Caption, TextCaps, VQAV2, OKVQA, GQA, ScienceQA, VizWiz, TextVQA, OCRVQA, ESTVQA, STVQA, AI2D and DUE_Benchmark. When using the data, it is necessary to comply with the protocols of the original dataset.

## Evaluate

We offer evaluation code for 14 Visual Question Answering (VQA) datasets in the `evaluate_vqa.py` file, facilitating a quick verification of results.  The specific operations are as follows:

 1. Make sure you have configured the [environment](#environment).
 2. Modify `sys.path.append("pathto/Monkey")`  to the project path.
 3. Prepare the datasets required for evaluation. 
 4. Run the evaluation code.

 Take ESTVQA as an example:
 - Prepare data according to the following directory structure:
```
├── data
|	├── estvqa
|		├── test_image
|			├── {image_path0}
|			├── {image_path1}
|				  ·
|				  ·
|	├── estvqa.jsonl
```
 - Example of the format of each line of the annotated `.jsonl` file:
```
{"image": "data/estvqa/test_image/011364.jpg", "question": "What is this store?", "answer": "pizzeria", "question_id": 0}
```
 - Modify the dictionary `ds_collections`:
```
ds_collections = {
	'estvqa_test': {
		'test': 'data/estvqa/estvqa.jsonl',
		'metric': 'anls',
		'max_new_tokens': 100,
	},
	...
}
```
 - Run the following command:
```
bash eval/eval.sh 'EVAL_PTH' 'SAVE_NAME'
```


## Citing Monkey
If you wish to refer to the baseline results published here, please use the following BibTeX entries:

```BibTeX
@inproceedings{li2023monkey,
  title={Monkey: Image Resolution and Text Label Are Important Things for Large Multi-modal Models},
  author={Li, Zhang and Yang, Biao and Liu, Qiang and Ma, Zhiyin and Zhang, Shuo and Yang, Jingxu and Sun, Yabo and Liu, Yuliang and Bai, Xiang},
  booktitle={proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  year={2024}
}
@article{liu2024textmonkey,
  title={TextMonkey: An OCR-Free Large Multimodal Model for Understanding Document},
  author={Liu, Yuliang and Yang, Biao and Liu, Qiang and Li, Zhang and Ma, Zhiyin and Zhang, Shuo and Bai, Xiang},
  journal={arXiv preprint arXiv:2403.04473},
  year={2024}
}
@article{huang2024mini,
  title={Mini-Monkey: Multi-Scale Adaptive Cropping for Multimodal Large Language Models},
  author={Huang, Mingxin and Liu, Yuliang and Liang, Dingkang and Jin, Lianwen and Bai, Xiang},
  journal={arXiv preprint arXiv:2408.02034},
  year={2024}
}
@article{deng2024r,
  title={R-CoT: Reverse Chain-of-Thought Problem Generation for Geometric Reasoning in Large Multimodal Models},
  author={Deng, Linger and Liu, Yuliang and Li, Bohan and Luo, Dongliang and Wu, Liang and Zhang, Chengquan and Lyu, Pengyuan and Zhang, Ziyang and Zhang, Gang and Ding, Errui and others},
  journal={arXiv preprint arXiv:2410.17885},
  year={2024}
}
```

## Acknowledgement
The Monkey series is primarily focused on exploring techniques such as image resolution enhancement and token compression methods to improve the performance of existing multimodal large models. For instance, earlier versions of Monkey and TextMonkey were based on QwenVL, while MiniMonkey is based on InternVL2 and miniCPM, among others. Thanks to
[Qwen-VL](https://github.com/QwenLM/Qwen-VL.git), [LLAMA](https://github.com/meta-llama/llama), [LLaVA](https://github.com/haotian-liu/LLaVA), [OpenCompass](https://github.com/open-compass/opencompass), [InternLM](https://github.com/InternLM/InternLM), and [InternVL](https://github.com/OpenGVLab/InternVL).  


## Copyright
Monkey project is intended for non-commercial use only. For commercial inquiries or to explore more advanced versions of the Monkey series LMMs (<1b, 2b, 7b, 72b), please contact Prof. Yuliang Liu at ylliu@hust.edu.cn. 
