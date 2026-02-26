import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os

# os.environ["HF_HOME"] = "/root/autodl-tmp/cache/"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from PIL import Image
import math

from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
from Qwen_VL.modeling_qwen import QWenLMHeadModel
from nolan_utils.nolan_sample import evolve_nolan_sampling

evolve_nolan_sampling()


def eval_model(args):
    # Model
    # template = " Please answer this question with one word."
    template = ""
    disable_torch_init()
    print(args.use_cd, args.question_file, args.answers_file)
    print(args.top_k, args.top_p)
    model_path = os.path.expanduser(args.model_path)
    model_name = "qwen-vl"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eod_id
    model = QWenLMHeadModel.from_pretrained(
        model_path, device_map="cuda", trust_remote_code=True
    ).eval()

    questions = [
        json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")
    ]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        if "\nAnswer the question using a single word or phrase." in line["text"]:
            line["text"] = (
                line["text"]
                .replace("\nAnswer the question using a single word or phrase.", "")
                .strip()
            )

        question = line["text"]

        image_path = os.path.join(args.image_folder, image_file)
        question = "<img>{}</img>{} Answer:".format(image_path, question + template)
        input_ids = tokenizer([question], return_tensors="pt", padding="longest")

        image_tensor = Image.open(image_path).convert("RGB")
        image_tensor = (
            model.transformer.visual.image_transform(image_tensor)
            .unsqueeze(0)
            .to(model.device)
        )

        if args.use_cd:
            question_cd = "{} Answer:".format(line["text"] + template)
            input_ids_cd = tokenizer(
                [question_cd], return_tensors="pt", padding="longest"
            )
        else:
            input_ids_cd = None

        pred = model.generate(
            input_ids=input_ids.input_ids.cuda(),
            input_ids_cd=input_ids_cd.input_ids.cuda(),
            attention_mask=input_ids.attention_mask.cuda(),
            do_sample=True,
            max_new_tokens=20,
            min_new_tokens=1,
            length_penalty=1,
            num_return_sequences=1,
            output_hidden_states=True,
            use_cache=True,
            pad_token_id=tokenizer.eod_id,
            eos_token_id=tokenizer.eod_id,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            images=image_tensor,
            cd_beta=args.cd_beta,
            cd_alpha=args.cd_alpha,
        )

        outputs = [
            tokenizer.decode(
                _[input_ids.input_ids.size(1) :].cpu(), skip_special_tokens=True
            ).strip()
            for _ in pred
        ][0]
        outputs = outputs.strip()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": question,
                    "text": outputs,
                    "model_id": model_name,
                    "image": image_file,
                    "metadata": {},
                }
            )
            + "\n"
        )
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/workspace/ckpt/Qwen-VL")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", action="store_true", default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
