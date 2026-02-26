import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
from transformers import set_seed
from accelerate import infer_auto_device_map, dispatch_model

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

from lavis.models import load_model_and_preprocess
from nolan_utils.nolan_sample import evolve_nolan_sampling

evolve_nolan_sampling()

import subprocess, re


def get_gpu_with_max_free_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits"
    output = subprocess.check_output(command.split(), universal_newlines=True)
    gpu_memory_info = output.strip().split("\n")
    gpu_memory_info = [int(re.findall(r"\d+", info)[0]) for info in gpu_memory_info]
    gpu_with_max_free_memory = gpu_memory_info.index(max(gpu_memory_info))
    return gpu_with_max_free_memory


def get_blip_model(args, device="cuda", dtype=torch.bfloat16, use_multi_gpus=False):
    cuda_number = get_gpu_with_max_free_memory()
    print(f"cuda number: {cuda_number}")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip2_vicuna_instruct",
        model_type=args.model_base,
        is_eval=True,
        device=f"cuda:{cuda_number}",
    )
    if use_multi_gpus:
        device_map = infer_auto_device_map(
            model,
            max_memory={0: "16GiB", 1: "16GiB"},
            no_split_module_classes=["LlamaDecoderLayer", "VisionTransformer"],
        )
        device_map["llm_model.model.embed_tokens"] = device_map["llm_model.lm_head"] = (
            device_map["llm_proj"]
        ) = 1
        print(device_map)
        model = dispatch_model(model, device_map=device_map)
        torch.cuda.empty_cache()
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    return model, vis_processors, txt_processors


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # template=" Please answer this question with one word."
    template = ""
    # Model
    disable_torch_init()
    print(args.use_cd, args.question_file, args.answers_file)
    print(args.top_k, args.top_p)

    # single gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # loads InstructBLIP model
    # single gpu
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_vicuna_instruct",
        model_type=args.model_base,
        is_eval=True,
        device=device,
    )
    model = model.to(device)

    # multi gpus
    # model, vis_processors, _ = get_blip_model(args=args, use_multi_gpus=True)

    questions = [
        json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")
    ]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        prompt = qs + template

        raw_image = Image.open(os.path.join(args.image_folder, image_file)).convert(
            "RGB"
        )
        # prepare the image
        image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        ## create a white image for contrastive decoding
        if args.use_cd:
            input_ids_cd = prompt
        else:
            input_ids_cd = None

        with torch.inference_mode():
            outputs = model.generate(
                {"image": image_tensor, "prompt": prompt},
                input_ids_cd=input_ids_cd,
                use_nucleus_sampling=True,
                num_beams=1,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=1,
                cd_beta=args.cd_beta,
                cd_alpha=args.cd_alpha,
            )

        outputs = outputs[0]
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": prompt,
                    "text": outputs,
                    "model_id": "instruct_blip",
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
