import os
import torch
import sys
sys.path.insert(0, "/data/pinci/code/mllms_fine_grained")
from PIL import Image

from transformers import AutoTokenizer

from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, select_best_resolution, get_model_name_from_path
from utils import get_possible_resolutions

from llava.model.builder import load_pretrained_model
# Backends
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig as LlavaLlamaConfig
try:
    from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM, LlavaQwenConfig
    HAS_QWEN = True
except Exception:
    LlavaQwenForCausalLM, LlavaQwenConfig, HAS_QWEN = None, None, False

from torch.nn import CrossEntropyLoss
from utils import process_images


def load_llava_model(model_path, vision_tower_path, device, multi_gpu, attn_implementation="flash_attention_2", torch_dtype=torch.float16, padding_side="left"):
    model_name = get_model_name_from_path(model_path)
    model_name += 'llava'
    model_base = None
    use_qwen = ("qwen" in model_path.lower()) and HAS_QWEN
    device_map = device if not multi_gpu else "auto"
    if use_qwen:
        default_conv_type = "qwen_1_5"
    else:
        default_conv_type = "v1"
    
    if vision_tower_path is not None:
        overwrite_config={"mm_vision_tower": vision_tower_path}
    else:
        overwrite_config = None
    # Use load_pretrained_model with customized_config to avoid config conflict
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        model_base, 
        model_name, 
        device_map=device_map,
        attn_implementation=attn_implementation,
        overwrite_config=overwrite_config,
        torch_dtype=torch_dtype
    )

    model.config.tokenizer_padding_side = padding_side

    return model, tokenizer, image_processor, default_conv_type, use_qwen

# def load_llava_model(model_path: str, device: str = "cuda:0", torch_dtype=torch.float16, load_8bit=False, load_4bit=False, attn_implementation="flash_attention_2", padding_side: str = "left", vision_tower_path: str = None, multi_gpu: bool = False):
#     disable_torch_init()

#     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
#     tokenizer.padding_side = padding_side

#     # Decide backend
#     use_qwen = ("qwen" in model_path.lower()) and HAS_QWEN

#     if use_qwen:
#         llava_cfg = LlavaQwenConfig.from_pretrained(model_path)
#         if vision_tower_path is not None:
#             llava_cfg.mm_vision_tower = vision_tower_path
#         model = LlavaQwenForCausalLM.from_pretrained(
#             model_path,
#             low_cpu_mem_usage=True,
#             torch_dtype=torch_dtype,
#             attn_implementation=attn_implementation,
#             device_map=device if not multi_gpu else "auto",
#             config=llava_cfg,
#             load_8bit=load_8bit,
#             load_4bit=load_4bit,
#         )
#         default_conv_type = "qwen_1_5"
#     else:
#         llava_cfg = LlavaLlamaConfig.from_pretrained(model_path)
#         if "v1.5" in model_path.lower():
#             llava_cfg.delay_load = True
#         if vision_tower_path is not None:
#             # Support specifying external vision tower for llama variants as well
#             llava_cfg.mm_vision_tower = vision_tower_path
#         model = LlavaLlamaForCausalLM.from_pretrained(
#             model_path,
#             low_cpu_mem_usage=True,
#             torch_dtype=torch_dtype,
#             attn_implementation=attn_implementation,
#             device_map=device if not multi_gpu else "auto",
#             config=llava_cfg,
#             load_8bit=load_8bit,
#             load_4bit=load_4bit,
#         )
#         default_conv_type = "v1"

#     model.config.tokenizer_padding_side = padding_side
#     if not multi_gpu:
#         model.to(device)

#     vision_tower = model.get_vision_tower()
#     if not vision_tower.is_loaded:
#         if multi_gpu:
#             vision_tower.load_model(device_map="auto")
#         else:
#             vision_tower.load_model(device_map=device)
#             vision_tower.to(device=device, dtype=torch.float16)
#     image_processor = vision_tower.image_processor

#     return model, tokenizer, image_processor, default_conv_type, use_qwen


def format_question(question, option_str):
    return question + "\n" + option_str + "Answer the option letter directly."


def format_question_multichoice(question, options):
    ret = question
    for o in options:
        ret += "\n"
        ret += o
    ret += "\nSelect the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option.\nThe best answer is:"
    return ret

@torch.inference_mode()
def multiple_choices_inference(model, tokenizer, image_processor, image, question, options, conv_type="v1", use_qwen=False):
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=model.device) for _image in image_tensor]
    image_sizes = [image.size]
    # if use_qwen:
    #     possible_resolutions = get_possible_resolutions(384, model.config.image_grid_pinpoints)
    #     image_sizes = [select_best_resolution(image.size, possible_resolutions)]

    conv = conv_templates[conv_type].copy()
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{question}")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    question_input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    output_question = model(
        question_input_ids,
        use_cache=True,
        images=image_tensor,
        image_sizes=image_sizes
    )

    question_logits = output_question.logits
    question_past_key_values = output_question.past_key_values

    loss_list = []
    for option in options:
        full_conv = conv_templates[conv_type].copy()
        full_conv.append_message(full_conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{question}")
        full_conv.append_message(full_conv.roles[1], option)
        full_prompt = full_conv.get_prompt()
        full_input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        option_answer_input_ids = full_input_ids[:, question_input_ids.shape[1]:]

        output_option = model(
            input_ids=option_answer_input_ids,
            use_cache=True,
            attention_mask=torch.ones(1, question_logits.shape[1]+option_answer_input_ids.shape[1], device=full_input_ids.device),
            past_key_values=question_past_key_values
        )

        logits = torch.cat([question_logits[:, -1:], output_option.logits[:, :-1]], 1)
        loss_fct = CrossEntropyLoss()
        logits = logits.view(-1, model.config.vocab_size)
        labels = option_answer_input_ids.view(-1)
        loss = loss_fct(logits, labels)
        loss_list.append(loss)

    option_chosen = torch.stack(loss_list).argmin()
    
    return option_chosen.detach().cpu().item()


@torch.inference_mode()
def free_form_response(model, tokenizer, image_processor, image, question, conv_type="v1", use_qwen=False):
    conv = conv_templates[conv_type].copy()
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{question}")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=model.device) for _image in image_tensor]
    image_sizes = [image.size]
    if use_qwen:
        possible_resolutions = get_possible_resolutions(384, model.config.image_grid_pinpoints)
        image_sizes = [select_best_resolution(image.size, possible_resolutions)]

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=256,
    )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return {"text": outputs}


@torch.inference_mode()
def option_list_response(model, tokenizer, image_processor, image, question, options, conv_type="v1", use_qwen=False):
    answers = []
    for option_str in options:
        q = format_question(question, option_str)
        resp = free_form_response(model, tokenizer, image_processor, image, q, conv_type, use_qwen)
        answers.append(resp["text"])
    return answers


@torch.inference_mode()
def multiple_choice_letter_response(model, tokenizer, image_processor, image, question, options, conv_type="v1", use_qwen=False):
    q = format_question_multichoice(question, options)
    resp = free_form_response(model, tokenizer, image_processor, image, q, conv_type, use_qwen)
    return resp["text"]
