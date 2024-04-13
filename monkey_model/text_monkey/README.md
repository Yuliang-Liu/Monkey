# TextMonkey: An OCR-Free Large Multimodal Model for Understanding Document

<br>

<p align="center">
    <img src="https://v1.ax1x.com/2024/04/13/7ySD7w.png" width="300"/>
<p>

> [**TextMonkey: An OCR-Free Large Multimodal Model for Understanding Document**](https://arxiv.org/abs/2403.04473)<br>
> Yuliang Liu, Biao Yang, Qiang Liu, Zhang Li, Zhiyin Ma, Shuo Zhang, Xiang Bai <br>

[![arXiv](https://img.shields.io/badge/Arxiv-2403.04473-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2403.04473) 
[![Source_code](https://img.shields.io/badge/Code-Available-white)](monkey_model/text_monkey/README.md)
[![Demo](https://img.shields.io/badge/Demo-blue)](http://vlrlab-monkey.xyz:7684/)
[![Data](https://img.shields.io/badge/Data-yellow)](https://www.modelscope.cn/datasets/lvskiller/TextMonkey_data)
[![Model Weight](https://img.shields.io/badge/Model_Weight-gray)](https://www.modelscope.cn/models/lvskiller/TextMonkey)


-----

**TextMonkey** is a multi-modal large model (LMM) focused on text-related tasks, including document question answering and scene text question answering. Compared with Monkey, TextMonkey has been improved in many aspects: by using zero-initialized Shifted Window Attention, TextMonkey realizes information interaction between windows at a higher input resolution; by calculating similarity to filter out important image features, not only can it simplify the input, but it can also improve the performance of the model. Furthermore, TextMonkey enhances interpretability and reduces hallucinations by extending multiple text-related tasks and incorporating location information into responses. At the same time, after fine-tuning, TextMonkey can also have the ability to understand user instructions and click on the corresponding location in the APP Agent, demonstrating its huge potential for downstream applications. 


# TODO

- [x] Open source code, weight, and data
- [ ] Improve Chinese language proficiency
- [ ] TextMonkey with different LLMs


# Model Zoo
| Method | LLM | STVQA | TextVQA | OCRVQA | DocVQA | InfoVQA | ChartQA | FUNSD | SROIE | POIE | OCRBench |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BLIP2-OPT-6.7B | OPT-6.7B | 20.9 | 23.5 | 9.7 | 3.2 | 11.3 | 3.4 | 0.2 | 0.1 | 0.3 | 235 |
| mPLUG-Owl | LLaMA-7B | 30.5 | 34.0 | 21.1 | 7.4 | 20.0 | 7.9 | 0.5 | 1.7 | 2.5 | 297 |
| InstructBLIP | Vircuna-7B | 27.4 | 29.1 | 41.3 | 4.5 | 16.4 | 5.3 | 0.2 | 0.6 | 1.0 | 276 |
| LLaVAR | Vircuna-7B | 39.2 | 41.8 | 24.0 | 12.3 | 16.5 | 12.2 | 0.5 | 5.2 | 5.9 | 346 |
| BLIVA | Vircuna-7B | 32.1 | 33.3 | 50.7 | 5.8 | 23.6 | 8.7 | 0.2 | 0.7 | 2.1 | 291 |
| mPLUG-Owl2 | LLaMA-7B | 49.8 | 53.9 | 58.7 | 17.9 | 18.9 | 19.4 | 1.4 | 3.2 | 9.9 | 366 |
| LLaVA1.5-7B$ | Vircuna-7B | 38.1 | 38.7 | 58.1 | 8.5 | 14.7 | 9.3 | 0.2 | 1.7 | 2.5 | 297 |
| TGDoc$ | Vircuna-7B | 36.3 | 46.2 | 37.2 | 9.0 | 12.8 | 12.7 | 1.4 | 3.0 | 22.2 | - |
| UniDoc | Vircuna-7B | 35.2 | 46.2 | 36.8 | 7.7 | 14.7 | 10.9 | 1.0 | 2.9 | 5.1 | - |
| DocPedia | Vircuna-7B | 45.5 | 60.2 | 57.2 | 47.1 | 15.2 | 46.9 | 9.9 | 21.4 | 39.9 | - |
| Monkey | Qwen-7B | 54.7 | 64.3 | 64.4 | 50.1 | 25.8 | 54.0 | 24.1 | 41.9 | 19.9 | 514 |
| InternVL | - | 62.2 | 59.8 | 30.5 | 28.7 | 23.6 | 45.6 | 6.5 | 26.4 | 25.9 | 517 |
| InternLM-XComposer2 | InternLM-7B | 59.6 | 62.2 | 49.6 | 39.7 | 28.6 | 51.6 | 15.3 | 34.2 | 49.3 | 511 |
| TextMonkey (40k data)| Qwen-7B | 61.8 | 65.9 | 71.3 | 64.3 | 28.2 | 58.2 | 32.3 | 47.0 | 27.9 | 561 |
| TextMonkey (50k data) | Qwen-7B | 61.2 | 64.3 | 72.2 | 66.7 | 28.6 | 59.9 | 42.9 | 46.2 | 32.0 | 558 |



## Environment

```python
conda create -n textmonkey python=3.10
conda activate textmonkey
git clone https://github.com/MelosY/TextMonkey.git
cd ./TextMonkey
pip install -r requirements.txt
```

## Evaluate

We also offer TextMonkey's model testing code, which you can explore above. You can execute the training code through executing:

```python
bash eval/eval_doc.sh
```

## Train

Execute the training code:
```python
bash finetune/finetune_textmonkey.sh
```

## Cases

TextMonkey can accurately locate and recognize text in both scene images and document images. In addition, the natural image in (a), the document in (b), the diagram in (c), and the table in (d) all demonstrate TextMonkey’s ability to identify, understand, and locate text information in a variety of scenarios.

<br>
<p align="center">
    <img src="https://v1.ax1x.com/2024/04/13/7ySSXO.png" width="700"/>
<p>
<br>

TextMonkey has shown strong feasibility as an agent for smartphone applications. After fine-tuning using 15k user click data from the Rico dataset, TextMonkey was able to understand user intent and click the corresponding icon.

<br>
<p align="center">
    <img src="https://v1.ax1x.com/2024/04/13/7ySOV6.png" width="700"/>
<p>
<br>



## Citing TextMonkey

If you wish to refer to the baseline results published here, please use the following BibTeX entries:

```BibTeX
@article{liu2024textmonkey,
  title={TextMonkey: An OCR-Free Large Multimodal Model for Understanding Document},
  author={Liu, Yuliang and Yang, Biao and Liu, Qiang and Li, Zhang and Ma, Zhiyin and Zhang, Shuo and Bai, Xiang},
  journal={arXiv preprint arXiv:2403.04473},
  year={2024}
}
```


## Copyright

We welcome suggestions to help us improve the TextMonkey. For any query, please contact Dr. Yuliang Liu: ylliu@hust.edu.cn. If you find something interesting, please also feel free to share with us through email or open an issue.
