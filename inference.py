from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="echo840/Monkey-Chat") #echo840/Monkey-Chat  echo840/Monkey
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--question", type=str, default=None)
    args = parser.parse_args()

    checkpoint = args.model_path
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map='cuda', trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id
    img_path = args.image_path
    question = args.question
    if question == "Generate the detailed caption in English:" and "Monkey-Chat" not in checkpoint:
        query = f'<img>{img_path}</img> Generate the detailed caption in English: ' #detailed caption
    else:
        query = f'<img>{img_path}</img> {question} Answer: ' #VQA
    
    input_ids = tokenizer(query, return_tensors='pt', padding='longest')
    attention_mask = input_ids.attention_mask
    input_ids = input_ids.input_ids

    pred = model.generate(
                input_ids=input_ids.cuda(),
                attention_mask=attention_mask.cuda(),
                do_sample=False,
                num_beams=1,
                max_new_tokens=512,
                min_new_tokens=1,
                length_penalty=1,
                num_return_sequences=1,
                output_hidden_states=True,
                use_cache=True,
                pad_token_id=tokenizer.eod_id,
                eos_token_id=tokenizer.eod_id,
                )
    response = tokenizer.decode(pred[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()
    print(f"Question: {question} Answer: {response}")
