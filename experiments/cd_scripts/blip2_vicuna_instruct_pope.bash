seed=${1:-55}
dataset_name=${2:-"coco"}
# popular random adversarial _no_imagecd
type=${3:-"random"}
model_path=${4:-"blip2_vicuna_instruct"}
cd_alpha=${5:-1}
cd_beta=${6:-0.9}
noise_step=${7:-500}
if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder=/root/autodl-tmp/val2014
else
  image_folder=/local_home/renlingfeng/data_repo/gqa/images
fi

python ./eval/object_hallucination_vqa_instructblip.py \
--question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/blip2_vicuna_instruct/blip2_vicuna_instruct_${dataset_name}_pope_${type}_answers_yes_cd_no_imagecd_seed${seed}_${cd_beta}_auto.jsonl \
--cd_alpha $cd_alpha \
--cd_beta $cd_beta \
--noise_step $noise_step \
--seed ${seed} \
--use_cd

