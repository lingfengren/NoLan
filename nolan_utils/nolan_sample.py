import copy
import csv
import inspect
import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import SampleOutput


def sa(q, kv):
    q = q.unsqueeze(-1)
    kv = kv.unsqueeze(-1)
    Q = q
    K = kv
    V = kv

    # 计算注意力得分
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(q.shape[-2], dtype=torch.float32)
    )
    attention = nn.functional.softmax(scores, dim=-1)

    # 应用注意力得分到 V
    output = torch.matmul(attention, V)
    return output.squeeze(-1)


def calculate_kl_divergence_file(logits1, logits2):

    # # 将logits转换为概率分布
    # probs1 = F.softmax(logits1, dim=-1)
    # probs2 = F.softmax(logits2, dim=-1)

    # # 计算两个方向的KL散度
    # kl_div_1_to_2 = F.kl_div(probs1.log(), probs2, reduction="batchmean")
    # kl_div_2_to_1 = F.kl_div(probs2.log(), probs1, reduction="batchmean")

    # # 计算平均KL散度
    # kl_div_mean = (kl_div_1_to_2 + kl_div_2_to_1) / 2

    logits1 = logits1.to(torch.float32)
    logits2 = logits2.to(torch.float32)

    # 将logits转换为概率分布
    probs1 = F.softmax(logits1, dim=-1)
    probs2 = F.softmax(logits2, dim=-1)

    # 添加一个小常数以防止对数零值
    epsilon = 1e-10
    probs1 = probs1 + epsilon
    probs2 = probs2 + epsilon

    # 计算两个方向的KL散度
    kl_div_1_to_2 = F.kl_div(probs1.log(), probs2, reduction="batchmean")
    kl_div_2_to_1 = F.kl_div(probs2.log(), probs1, reduction="batchmean")
    kl_div_mean = (kl_div_1_to_2 + kl_div_2_to_1) / 2

    m = 0.5 * (probs1 + probs2)

    # 计算 JS 散度
    kl_div_1_to_m = F.kl_div(probs1.log(), m, reduction="batchmean")  # KL(P || M)
    kl_div_2_to_m = F.kl_div(probs2.log(), m, reduction="batchmean")  # KL(Q || M)

    js_divergence = 0.5 * (kl_div_1_to_m + kl_div_2_to_m)

    output_file = "./output/llava15/llava15.csv"
    file_exists = os.path.exists(output_file)
    with open(output_file, "a", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        # 如果文件不存在，写入表头
        if not file_exists:
            csvwriter.writerow(
                ["kl_div_1_to_2", "kl_div_2_to_1", "kl_div_mean", "js_divergence"]
            )
        csvwriter.writerow(
            [
                kl_div_1_to_2.item(),
                kl_div_2_to_1.item(),
                kl_div_mean.item(),
                js_divergence.item(),
            ]
        )
    return kl_div_1_to_2, kl_div_2_to_1, kl_div_mean, js_divergence


def calculate_kl_divergence(logits1, logits2):

    # # 将logits转换为概率分布
    # probs1 = F.softmax(logits1, dim=-1)
    # probs2 = F.softmax(logits2, dim=-1)

    # # 计算两个方向的KL散度
    # kl_div_1_to_2 = F.kl_div(probs1.log(), probs2, reduction="batchmean")
    # kl_div_2_to_1 = F.kl_div(probs2.log(), probs1, reduction="batchmean")

    # # 计算平均KL散度
    # kl_div_mean = (kl_div_1_to_2 + kl_div_2_to_1) / 2

    logits1 = logits1.to(torch.float32)
    logits2 = logits2.to(torch.float32)

    # 将logits转换为概率分布
    probs1 = F.softmax(logits1, dim=-1)
    probs2 = F.softmax(logits2, dim=-1)

    # 添加一个小常数以防止对数零值
    epsilon = 1e-10
    probs1 = probs1 + epsilon
    probs2 = probs2 + epsilon

    # 计算两个方向的KL散度
    kl_div_1_to_2 = F.kl_div(probs1.log(), probs2, reduction="batchmean")
    kl_div_2_to_1 = F.kl_div(probs2.log(), probs1, reduction="batchmean")
    kl_div_mean = (kl_div_1_to_2 + kl_div_2_to_1) / 2

    m = 0.5 * (probs1 + probs2)

    # 计算 JS 散度
    kl_div_1_to_m = F.kl_div(probs1.log(), m, reduction="batchmean")  # KL(P || M)
    kl_div_2_to_m = F.kl_div(probs2.log(), m, reduction="batchmean")  # KL(Q || M)

    js_divergence = 0.5 * (kl_div_1_to_m + kl_div_2_to_m)

    return (
        kl_div_1_to_2.item(),
        kl_div_2_to_1.item(),
        kl_div_mean.item(),
        js_divergence.item(),
    )


def sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    # init values
    logits_processor = (
        logits_processor if logits_processor is not None else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = (
        logits_warper if logits_warper is not None else LogitsProcessorList()
    )
    pad_token_id = (
        pad_token_id
        if pad_token_id is not None
        else self.generation_config.pad_token_id
    )
    eos_token_id = (
        eos_token_id
        if eos_token_id is not None
        else self.generation_config.eos_token_id
    )

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = (
        torch.tensor(eos_token_id).to(input_ids.device)
        if eos_token_id is not None
        else None
    )
    output_scores = (
        output_scores
        if output_scores is not None
        else self.generation_config.output_scores
    )
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.generation_config.output_hidden_states
    )

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(
        input_ids.shape[0], dtype=torch.long, device=input_ids.device
    )

    this_peer_finished = False  # used by synced_gpus only

    # auto-regressive generation
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(
                0.0 if this_peer_finished else 1.0
            ).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        ## For contrastive decoding initial
        use_cd = model_kwargs.get("input_ids_cd") != None
        output_attentions_wo_img = (
            output_attentions
            if output_attentions is not None
            else self.generation_config.output_attentions
        )
        output_hidden_states_wo_img = (
            output_hidden_states
            if output_hidden_states is not None
            else self.generation_config.output_hidden_states
        )
        # print("--------", use_cd)
        # use_cd = False
        if use_cd:
            if (
                "past_key_values" not in model_kwargs
                or model_kwargs["past_key_values"] is None
            ):
                model_kwargs_cd = model_kwargs.copy()
                input_ids_cd = model_kwargs_cd.get("input_ids_cd")

            ## cd_comments: forward pass of the model with distorted image input
            model_inputs_cd = self.prepare_inputs_for_generation_cd(
                input_ids_cd, **model_kwargs_cd
            )

            outputs_cd = self(
                **model_inputs_cd,
                return_dict=True,
                output_attentions=output_attentions_wo_img,
                output_hidden_states=output_hidden_states_wo_img,
            )
            next_token_logits_cd = outputs_cd.logits[:, -1, :]

            ## cd_comments: pre-process logits from contrastive inputs
            cd_alpha = (
                model_kwargs.get("cd_alpha")
                if model_kwargs.get("cd_alpha") is not None
                else 0.5
            )
            # print(cd_alpha)
            # assert 1 == 2
            cd_beta = (
                model_kwargs.get("cd_beta")
                if model_kwargs.get("cd_beta") is not None
                else 0.1
            )
            # # nolan
            # diffs = (1 + cd_alpha) * next_token_logits - cd_alpha * next_token_logits_cd
            # cd_logits = diffs if cd_alpha != 0 else next_token_logits

            # nolan++
            kl_div_1_to_2, kl_div_2_to_1, kl_div_mean, js_divergence = (
                calculate_kl_divergence(next_token_logits, next_token_logits_cd)
            )
            cd_auto = 1 / (kl_div_mean)
            # cd_auto = (torch.tanh(torch.tensor(cd_auto))) * 1.6
            cd_auto = (torch.tanh(torch.tensor(cd_auto)) + 1) * 0.8

            # nolan++2
            # selected_logits1 = next_token_logits.to(torch.float32)
            # selected_logits2 = next_token_logits_cd.to(torch.float32)
            # # 将logits转换为概率分布
            # selected_logits1 = F.softmax(selected_logits1, dim=-1)
            # selected_logits2 = F.softmax(selected_logits2, dim=-1)
            # cos = torch.nn.CosineSimilarity(dim=-1)
            # similarity = cos(selected_logits1, selected_logits2).item()
            # cd_auto = similarity
            # cd_auto = (torch.tanh(torch.tensor(cd_auto)) + 1) * 0.90

            diffs = (1 + cd_auto) * next_token_logits - cd_auto * next_token_logits_cd
            cd_logits = diffs if cd_auto != 0 else next_token_logits

            ## cd_comments: apply temperature warping and top-k filtering in contrastive decoding
            cd_logits = logits_processor(input_ids, cd_logits)
            cd_logits = logits_warper(input_ids, cd_logits)

            next_token_scores = cd_logits
            cd_probs = nn.functional.softmax(cd_logits, dim=-1)
            next_tokens = torch.multinomial(cd_probs, num_samples=1).squeeze(1)
        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError(
                    "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                )
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences
            )

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        ## cd_comments: update model_kwargs_cd for contrastive decoding
        if use_cd:
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd,
                model_kwargs_cd,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            # print(input_ids_cd.shape)
            if len(input_ids_cd.shape) == 3:
                input_ids_cd = input_ids[:, :-1]
            input_ids_cd = torch.cat([input_ids_cd, next_tokens[:, None]], dim=-1)
            # model_kwargs["input_ids_cd"] = None

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids


def evolve_nolan_sampling():
    transformers.generation.utils.GenerationMixin.sample = sample
