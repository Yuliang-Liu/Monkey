import argparse
import itertools
import json
import os
import random
import time
from functools import partial
from typing import Optional
import sys
import torch
from tqdm import tqdm

from vqa import VQA
from vqa_eval import VQAEval

sys.path.append("pathto/Monkey/")
from monkey_model.modeling_monkey import MonkeyLMHeadModel

from monkey_model.tokenization_qwen import QWenTokenizer
import numpy as np
from pathlib import Path
time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
ds_collections = {
    'estvqa_test': {
        'train': 'data/ESTVQA/estvqa.jsonl',
        'test': 'data/estvqa/estvqa.jsonl',
        'metric': 'anls',
        'max_new_tokens': 100,
    },
    'docvqa_test': {
        'train': 'data/docvqa/train.jsonl',
        'test': 'data/docvqa/test_ans.jsonl',
        'metric': 'anls',
        'max_new_tokens': 100,
    },
    'chartqa': {
        'train': 'data/chartqa/train_augmented.jsonl',
        'test': 'data/chartqa/chartqa.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'infovqa_test': {
        'train': 'data/infographicVQA/infovqa.jsonl',
        'test': 'data/infographicVQA/infovqa_test.jsonl',
        'metric': 'anls',   
        'max_new_tokens': 100,
    },
    'vizwiz_val': {
        'train': 'data/vizwiz/vizwiz_train.jsonl',
        'test': 'data/vizwiz/vizwiz_val.jsonl',
        'question': 'data/vizwiz/vizwiz_val_questions.json',
        'annotation': 'data/vizwiz/vizwiz_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'deepform': {
        'train': '',
        'test': 'data/test_DeepForm.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'KLC': {
        'train': '',
        'test': 'data/test_KleisterCharity.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'WTQ': {
        'train': '',
        'test': 'data/test_WikiTableQuestions.jsonl',
        'metric': 'accuracy',   
        'max_new_tokens': 100,
    },
    'gqa_testdev': {
        'train': 'data/gqa/train.jsonl',
        'test': 'data/gqa/gqa_testdev_new.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'okvqa_val': {
        'train': 'data/okvqa/okvqa_train.jsonl',
        'test': 'data/okvqa/okvqa_val.jsonl',
        'question': 'data/okvqa/OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'data/okvqa/mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'textvqa_val': {
        'train': 'data/textvqa/textvqa_train.jsonl',
        'test': 'data/textvqa/textvqa_val.jsonl',
        'question': 'data/textvqa/textvqa_val_questions.json',
        'annotation': 'data/textvqa/textvqa_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'stvqa_test': {
        'train': 'data/STVQA/stvqa.jsonl',
        'test': 'data/STVQA/stvqa.jsonl',
        'metric': 'anls',
        'max_new_tokens': 100,
    },
    'ai2diagram_test': {
        'train': 'data/ai2d/train.jsonl',
        'test': 'data/ai2d/test.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'vqav2_val': {
        'train': 'data/vqav2/vqav2_train.jsonl',
        'test': 'data/vqav2/vqav2_val.jsonl',
        'question': 'data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'data/vqav2/v2_mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },

}



def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def normANLS(s1,s2):
    dist = levenshtein_distance(s1.lower().strip(),s2.lower().strip())
    length = max(len(s1),len(s2))
    value =  0.0 if length == 0 else float(dist) / float(length) 
    return value 

def evaluateANLS(ans_list):
    anls_threshold = 0.5
    anls_list = []
    for predict_pair in ans_list:
        answer = predict_pair["answer"].strip()
        gt_list = predict_pair["annotation"]
        
        value_list = []
        for gt_single in gt_list:
            value_list.append(normANLS(gt_single,answer))
        question_result = 1 - min(value_list)

        if (question_result < anls_threshold) :
            question_result = 0
        anls_list.append(question_result)
    return np.mean(anls_list)
# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            relaxed_correctness(elem['answer'].strip(), ann)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def evaluate_exact_match_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            (1.0 if
             (ann.strip().lower() in  elem['answer'].strip().lower().replace(".","") ) else 0.0)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def collate_fn(batches, tokenizer):
    image_paths = [_['image_path'] for _ in batches]
    questions = [_['question'] for _ in batches]
    question_ids = [_['question_id'] for _ in batches]
    annotations = [_['annotation'] for _ in batches]

    input_ids = tokenizer(questions, return_tensors='pt', padding='longest')

    return image_paths,question_ids, input_ids.input_ids, input_ids.attention_mask, annotations


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, train, test, prompt, few_shot):
        self.test = open(test).readlines()
        self.prompt = prompt

        self.few_shot = few_shot
        if few_shot > 0:
            self.train = open(train).readlines()

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = json.loads(self.test[idx].strip())
        image, question, question_id, annotation = data['image'], data[
            'question'], data['question_id'], data.get('answer', None)

        few_shot_prompt = ''
        if self.few_shot > 0:
            few_shot_samples = random.sample(self.train, self.few_shot)
            for sample in few_shot_samples:
                sample = json.loads(sample.strip())
                few_shot_prompt += self.prompt.format(
                    sample['image'],
                    sample['question']) + f" {sample['answer']}"

        return {
            'image_path':image,
            'question': few_shot_prompt + self.prompt.format(image, question),
            'question_id': question_id,
            'annotation': annotation
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)



def evaluate(model,tokenizer,prompt,args,dataset_name):
    dataset_info = ds_collections[dataset_name]
    dataset = VQADataset(
        train=dataset_info['train'],
        test=dataset_info['test'],
        prompt=prompt,
        few_shot=args.few_shot,
    )
    len_dataset = len(dataset)
    if torch.distributed.get_rank() == 0:
        print(f"there have {len(dataset)} in {dataset_name}")

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len_dataset),
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    outputs = []
    for image_paths,question_ids, input_ids, attention_mask,annotations in tqdm(dataloader):
        pred = model.generate(
            input_ids=input_ids.cuda(),
            attention_mask=attention_mask.cuda(),
            do_sample=False,
            num_beams=1,
            max_new_tokens=dataset_info['max_new_tokens'],
            min_new_tokens=1,
            length_penalty=1,
            num_return_sequences=1,
            output_hidden_states=True,
            use_cache=True,
            pad_token_id=tokenizer.eod_id,
            eos_token_id=tokenizer.eod_id,
        )

        answers = [
            tokenizer.decode(_[input_ids.size(1):].cpu(),
                             skip_special_tokens=True).strip() for _ in pred
        ]

        for image_path,question_id, answer, annotation in zip(image_paths,question_ids, answers,
                                                   annotations):
            if dataset_name in ['vqav2_val', 'okvqa_val', 'textvqa_val', 'vizwiz_val']:
                outputs.append({
                    'image_path':image_path,
                    'question_id': question_id,
                    'answer': answer,
                })
            elif dataset_name in ['docvqa_test', 'gqa_testdev',"stvqa_test","infovqa_test"]:
                outputs.append({
                    'image_path':image_path,
                    'questionId': question_id,
                    'answer': answer,
                    'annotation': annotation,
                })
                
            elif dataset_name in ['ai2diagram_test',"WTQ","deepform","KLC"]:
                outputs.append({
                    'image_path':image_path,
                    'image': question_id,
                    'answer': answer,
                    'annotation': annotation,
                })
            elif dataset_name in ['estvqa_test']:
                outputs.append({
                    'image_path':image_path,
                    'questionId': question_id,
                    'answer': answer,
                    'annotation': [annotation],
                })
            elif dataset_name in ["chartqa"]:
                outputs.append({
                    'image_path':image_path,
                    'answer': answer,
                    'annotation': annotation,
                })

            else:
                raise NotImplementedError

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

    merged_outputs = [json.loads(_) for _ in merged_outputs]
    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

    if torch.distributed.get_rank() == 0:
        print(f"Evaluating {dataset_name} ...")
        results_file = f'{dataset_name}.json'
        root_path = os.path.join("result",args.save_name)
        Path(root_path).mkdir(exist_ok=True,parents=True)

        results_file = os.path.join(root_path,results_file)
        json.dump(merged_outputs, open(results_file, 'w',encoding="utf-8"), ensure_ascii=False,indent=2)

        if dataset_info['metric'] == 'vqa_score':
            vqa = VQA(dataset_info['annotation'],dataset_info['question'])
            results = vqa.loadRes(
                resFile=results_file,
                quesFile=dataset_info['question'])
            vqa_scorer = VQAEval(vqa, results, n=2)
            question_id_list = [item["question_id"]for item in merged_outputs]
            vqa_scorer.evaluate(question_id_list)

            print(vqa_scorer.accuracy)

            results_file = results_file.replace("json","txt")
            with open(results_file,"w") as fp:
                fp.write(dataset_name+"\n")
                fp.writelines(str(vqa_scorer.accuracy["overall"])+'\n')  
        elif dataset_info['metric'] == 'anls':
            json.dump(merged_outputs,
                      open(results_file, 'w'),
                      ensure_ascii=False)
            anls_res = evaluateANLS(merged_outputs)
            print(anls_res)
            results_file = results_file.replace("json","txt")
            with open(results_file,"w") as fp:
                fp.write(dataset_name+"\n")
                fp.writelines(str(anls_res)+'\n')  
        elif dataset_info['metric'] == 'relaxed_accuracy':
            print({
                'relaxed_accuracy': evaluate_relaxed_accuracy(merged_outputs)
            })
            results_file = results_file.replace("json","txt")
            with open(results_file,"w") as fp:
                fp.write(dataset_name+"\n")
                fp.writelines(str(evaluate_relaxed_accuracy(merged_outputs))+'\n') 
        elif dataset_info['metric'] == 'accuracy':
            if 'gqa' in dataset_name:
                for entry in merged_outputs:
                    response = entry['answer']
                    response = response.strip().split('.')[0].split(
                        ',')[0].split('!')[0].lower()
                    if 'is ' in response:
                        response = response.split('is ')[1]
                    if 'are ' in response:
                        response = response.split('are ')[1]
                    if 'a ' in response:
                        response = response.split('a ')[1]
                    if 'an ' in response:
                        response = response.split('an ')[1]
                    if 'the ' in response:
                        response = response.split('the ')[1]
                    if ' of' in response:
                        response = response.split(' of')[0]
                    response = response.strip()
                    entry['answer'] = response
            acc = evaluate_exact_match_accuracy(merged_outputs)
            print({'accuracy': acc})
            results_file = results_file.replace("json","txt")
            with open(results_file,"w") as fp:
                fp.write(dataset_name+"\n")
                fp.writelines(str(acc)+'\n') 


    torch.distributed.barrier()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--few-shot', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument("--save_name",type=str,default="test")
    args = parser.parse_args()

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model = MonkeyLMHeadModel.from_pretrained(
        args.checkpoint, device_map='cuda', trust_remote_code=True).eval()
    tokenizer = QWenTokenizer.from_pretrained(args.checkpoint,
                                              trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id

    random.seed(args.seed)
    for k,_ in ds_collections.items():
        if "vizwiz_val" in k:
            # prompt = '<img>{}</img> {} When the provided information is insufficient, respond with "Unanswerable". Answer:'  #vizwiz 61.2
            prompt = '<img>{}</img> {} When the provided information is insufficient, respond with "unanswerable". Answer: '  #vizwiz 68.3 # this prompt will achieve better results.
        else:
            # prompt = '<img>{}</img>{} Answer:'
            prompt = '<img>{}</img>{} Answer: '  # this prompt will achieve better results.
        evaluate(model,tokenizer,prompt,args,k)
