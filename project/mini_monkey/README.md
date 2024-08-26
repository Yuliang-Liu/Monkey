# Mini-Monkey: Multi-Scale Adaptive Cropping for Multimodal Large Language Models

<br>

<p align="center">
    <img src="https://v1.ax1x.com/2024/08/13/7GXu34.png" width="300"/>
<p>

> [**Mini-Monkey: Multi-Scale Adaptive Cropping for Multimodal Large Language Models**](https://arxiv.org/abs/2408.02034)<br>
> Mingxin Huang, Yuliang Liu, Dingkang Liang, Lianwen Jin, Xiang Bai <br>

[![arXiv](https://img.shields.io/badge/Arxiv-2408.02034-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2408.02034) 
[![Demo](https://img.shields.io/badge/Demo-blue)](http://vlrlab-monkey.xyz:7685)
[![Model Weight](https://img.shields.io/badge/Model_Weight-gray)](https://huggingface.co/mx262/MiniMokney)
[![Model Weight in Wisemodel](https://img.shields.io/badge/Model_Weight_in_Wisemodel-gray)](https://www.wisemodel.cn/models/HUST-VLRLab/Mini-Monkey)


-----

**Mini-Monkey** is a lightweight MLLM that incorporates a plug-and-play method called multi-scale adaptive cropping strategy (MSAC). Mini-Monkey adaptively generates multi-scale representations, allowing it to select non-segmented objects from various scales. To mitigate the computational overhead introduced by MSAC, we propose a Scale Compression Mechanism (SCM), which effectively compresses image tokens. Mini-Monkey achieves state-of-the-art performance among 2B-parameter MLLMs. It not only demonstrates leading performance on a variety of general multimodal understanding tasks but also shows consistent improvements in document understanding capabilities. On the OCRBench, Mini-Monkey achieves a score of 802, outperforming 8B-parameter state-of-the-art model InternVL2-8B. Besides, our model and training strategy are very efficient, which can be trained with only eight RTX 3090.


# TODO

- [x] Open source code, weight, and data
- [x] Support training using 3090 GPUs (24Gb video memory)
- [ ] Mini-Monkey with different LLMs


# Model Zoo

Mini-Monkey was trained using 8 3090 GPUs on a dataset 

| Model | #param | MME | RWQA | AI2D | CCB | SEED | HallB | POPE | MathVista | DocVQA | ChartQA | InfoVQA$ | TextVQA | OCRBench |
|-------|---------|-----|------|------|-----|------|-------|------|-----------|-------------------|-------------------|-------------------|----------------|----------|
| Mini-Gemini | 35B | 2141.0 | - | - | - | - | - | - | 43.3 | - | - | - | - | - |
| LLaVA-NeXT | 35B | 2028.0 | - | 74.9 | 49.2 | 75.9 | 34.8 | 89.6 | 46.5 | - | - | - | - | - |
| InternVL 1.2 | 40B | 2175.4 | 67.5 | 79.0 | 59.2 | 75.6 | 47.6 | 88.0 | 47.7 | - | - | - | - | - |
| InternVL 1.5 | 26B | 2187.8 | 66.0 | 80.7 | 69.8 | 76.0 | 49.3 | 88.3 | 53.5 | 90.9 | 83.8 | 72.5 | 80.6 | 724 |
| DeepSeek-VL | 1.7B | 1531.6 | 49.7 | 51.5 | 37.6 | 43.7 | 27.6 | 85.9 | 29.4 | - | - | - | - | - |
| Mini-Gemini | 2.2B | 1653.0 | - | - | - | - | - | - | 29.4 | - | - | - | - | - |
| Bunny-StableLM-2 | 2B | 1602.9 | - | - | - | 58.8 | - | 85.9 | - | - | - | - | - | - |
| MiniCPM-V-2 | 2.8B | 1808.6 | 55.8 | 62.9 | 48.0 | - | 36.1 | 86.3 | 38.7 | 71.9 | 55.6 | - | 74.1 | 605 |
| InternVL 2 | 2B | 1876.8 | 57.3 | 74.1 | 74.7 | 70.9 | 37.9 | 85.2 | 46.3 | 86.9 | 76.2 | 58.9 | 73.4 | 784 |
| Mini-Monkey (ours) | 2B | 1881.9 | 57.5 | 74.7 | 75.5 | 71.3 | 38.7 | 86.7 | 47.3 | 87.4 | 76.5 | 60.1 | 75.7 | 802 |


## Environment

```python
conda create -n minimonkey python=3.10
conda activate minimonkey
git clone https://github.com/Yuliang-Liu/Monkey.git
cd ./Monkey/project/mini_monkey
pip install -r requirements.txt
```
Install `flash-attn==2.3.6`:
```bash
pip install flash-attn==2.3.6 --no-build-isolation
```

Alternatively you can compile from source:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.3.6
python setup.py install
```


## Evaluate

We use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) repositories for model evaluation. 

## Inference
We provide an example of inference code [here](https://github.com/Yuliang-Liu/Monkey/blob/main/project/mini_monkey/demo.py)

## Train

### Prepare Training Datasets

Inspired by InternVL 1.2, we adopted a [LLaVA-ZH](https://huggingface.co/datasets/openbmb/llava_zh), [DVQA](https://github.com/kushalkafle/DVQA_dataset), [ChartQA](https://github.com/vis-nlp/ChartQA), [AI2D](https://allenai.org/data/diagrams), [DocVQA](https://www.docvqa.org/datasets), [GeoQA+](https://github.com/SCNU203/GeoQA-Plus), and [SynthDoG-EN](https://huggingface.co/datasets/naver-clova-ix/synthdog-en). Most of the data remains consistent with InternVL 1.2.

First, download the [annotation files](https://huggingface.co/OpenGVLab/InternVL/resolve/main/playground.zip) and place them in the `playground/opensource/` folder.

Second, download all the images we used.

- AI2D: [ai2d_images](https://drive.google.com/file/d/1dqqa3MnrxMXaU_K9JA6C83je32ibwdOY/view?usp=sharing) (provided by InternLM-XComposer)
- ChartQA: [ChartQA Dataset](https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/main/ChartQA%20Dataset.zip)
- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- DocVQA: [train](https://datasets.cvc.uab.es/rrc/DocVQA/train.tar.gz), [val](https://datasets.cvc.uab.es/rrc/DocVQA/val.tar.gz), [test](https://datasets.cvc.uab.es/rrc/DocVQA/test.tar.gz)
- DVQA: [images](https://drive.google.com/file/d/1iKH2lTi1-QxtNUVRxTUWFvUvRHq6HAsZ/view)
- LLaVA-Pretrain: [images](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip)
- SynthDoG-EN: We only use 00000~00004 parquet files for now, with a total of 30K images. We provide the converted [images](https://huggingface.co/OpenGVLab/InternVL/resolve/main/synthdog-en-images.zip).
- GeoQA+: [GeoQA+](https://drive.google.com/file/d/1KL4_wIzr3p8XSKMkkLgYcYwCbb0TzZ9O/view) [images](https://huggingface.co/OpenGVLab/InternVL/resolve/main/geoqa%2B_images.zip)

Then, organize the data as follows in `playground/data`:

```none
playground/
├── opensource
│   ├── ai2d_train_12k.jsonl
│   ├── chartqa_train_18k.jsonl
│   ├── docvqa_train_10k.jsonl
│   ├── dvqa_train_200k.jsonl
│   ├── geoqa+.jsonl
│   ├── llava_instruct_150k_zh.jsonl
│   └── synthdog_en.jsonl
├── data
│   ├── ai2d
│   │   ├── abc_images
│   │   └── images
│   ├── chartqa
│   │   ├── test
│   │   ├── train
│   │   └── val
│   ├── coco
│   │   └── train2017
│   ├── docvqa
│   │   ├── test
│   │   ├── train
│   │   └── val
│   ├── dvqa
│   │   └── images
│   ├── llava
│   │   └── llava_pretrain
│   │       └── images
│   ├── synthdog-en
│   │   └── images
│   ├── geoqa+
│   │   └── images
```

Execute the training code:
```python
sh shell/minimonkey/minimonkey_finetune_full.sh
```



## Citing Mini-Monkey

If you wish to refer to the baseline results published here, please use the following BibTeX entries:

```BibTeX
@article{huang2024mini,
  title={Mini-Monkey: Multi-Scale Adaptive Cropping for Multimodal Large Language Models},
  author={Huang, Mingxin and Liu, Yuliang and Liang, Dingkang and Jin, Lianwen and Bai, Xiang},
  journal={arXiv preprint arXiv:2408.02034},
  year={2024}
}
```


## Copyright

We welcome suggestions to help us improve the Mini-Monkey. For any query, please contact Dr. Yuliang Liu: ylliu@hust.edu.cn. If you find something interesting, please also feel free to share with us through email or open an issue.
