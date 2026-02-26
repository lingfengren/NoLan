seed=${1:-55}
dataset_name=${2:-"coco"}
# popular random adversarial _no_imagecd
type=${3:-"random"}
model_path=${4:-"liuhaotian/llava-v1.5-7b"}
cd_alpha=${5:-1}
cd_beta=${6:-0.9}
noise_step=${7:-500}
if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder=/root/autodl-tmp/val2014
else
  image_folder=/local_home/renlingfeng/data_repo/gqa/images
fi

qs=./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json
ans=./output/llava15/llava15_${dataset_name}_pope_${type}_answers_seed${seed}.jsonl

python ./eval/object_hallucination_vqa_llava.py \
  --model-path ${model_path} \
  --question-file $qs \
  --image-folder ${image_folder} \
  --answers-file  $ans\
  --cd_alpha $cd_alpha \
  --cd_beta $cd_beta \
  --noise_step $noise_step \
  --seed ${seed} \
  --use_cd

# python eval/eval_pope.py \
# --gt_files $qs \
# --gen_files $ans