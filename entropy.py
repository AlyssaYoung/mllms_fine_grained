from transformers.generation.logits_process import LogitsProcessorList
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import pdb


import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
import seaborn as sns
from tqdm import tqdm
import json
import os

available_fonts = set(f.name for f in font_manager.fontManager.ttflist)
if 'Noto Sans CJK JP' in available_fonts:
    matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
else:
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

def visualize_entropy_heatmap(entropy_matrix, generated_ids, tokenizer, save_path=None, annotate_values=False, prompt_length=None):
    """
    entropy_matrix: [T, L] entropy list
    generated_ids: [1, S] tensor of token ids, including prompt + generated tokens
    tokenizer: tokenizer to decode ids
    save_path: optional path to save the figure
    annotate_values: if True, display entropy values in each grid cell
    prompt_length: length of the prompt (to extract only generated tokens)
    """
    entropy_matrix = np.array(entropy_matrix)  # shape: [T, L]
    entropy_matrix = entropy_matrix.T          # -> shape: [L, T]
    # entropy_matrix = entropy_matrix.T[::2] 
    num_layers, num_steps = entropy_matrix.shape

    # 解码 token（只保留生成的 token 段）
    # Extract only the newly generated tokens, not the prompt
    if prompt_length is not None:
        # Only get tokens after the prompt (exactly num_steps tokens)
        end_idx = min(prompt_length + num_steps, generated_ids.shape[1])
        generated_token_ids = generated_ids[0, prompt_length:end_idx].tolist()
        # Pad or truncate to match num_steps
        if len(generated_token_ids) < num_steps:
            # Pad with placeholder if we don't have enough tokens
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            generated_token_ids.extend([pad_id] * (num_steps - len(generated_token_ids)))
        elif len(generated_token_ids) > num_steps:
            # Truncate if we have too many
            generated_token_ids = generated_token_ids[:num_steps]
    else:
        # Fallback: try to get the last num_steps tokens
        if generated_ids.shape[1] >= num_steps:
            generated_token_ids = generated_ids[0, -num_steps:].tolist()
        else:
            generated_token_ids = generated_ids[0, :].tolist()
            # Pad if needed
            if len(generated_token_ids) < num_steps:
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                generated_token_ids.extend([pad_id] * (num_steps - len(generated_token_ids)))
            else:
                generated_token_ids = generated_token_ids[:num_steps]
    
    # Decode tokens
    if len(generated_token_ids) == num_steps:
        generated_tokens = tokenizer.convert_ids_to_tokens(generated_token_ids, skip_special_tokens=False)
    else:
        # Fallback: use step numbers if not enough tokens
        generated_tokens = [f"Step{i}" for i in range(num_steps)]

    # 处理 token 显示（加入换行、旋转、空格优化等）
    # Better cleaning for matplotlib compatibility
    def clean_token_for_matplotlib(tok):
        """Clean token text to be safe for matplotlib display"""
        # Remove sentencepiece prefix
        if tok.startswith("▁"):
            tok = tok[1:]
        
        # Replace special characters that matplotlib might misinterpret
        # Replace angle brackets with full-width versions
        if tok == "<|im_end|>":
            tok = "<END>"
        if tok == "<|im_start|>":
            tok = "<START>"
        tok = tok.replace("<", "＜").replace(">", "＞")
        # Replace other problematic characters
        tok = tok.replace("|", "｜")
        tok = tok.replace("Ġ", " ")
        
        # Remove control characters and non-printable characters
        tok = ''.join(char for char in tok if char.isprintable() or char in [' ', '\t'])
        
        # If empty after cleaning, return a placeholder
        if not tok.strip():
            return "[?]"
        
        return tok.strip()
    
    display_tokens = []
    for i, tok in enumerate(generated_tokens):
        # Clean the token
        cleaned_tok = clean_token_for_matplotlib(tok)
        
        # If still empty or problematic, try decoding the raw token ID
        if cleaned_tok == "[?]" and i < len(generated_token_ids):
            # Try decoding the raw token to see what it is
            try:
                raw_tok = tokenizer.decode([generated_token_ids[i]], skip_special_tokens=False)
                if raw_tok.strip():
                    cleaned_tok = clean_token_for_matplotlib(raw_tok)
                else:
                    cleaned_tok = f"T{generated_token_ids[i]}"  # Show token ID if can't decode
            except:
                cleaned_tok = f"T{generated_token_ids[i]}"
        
        display_tokens.append(cleaned_tok)

    plt.figure(figsize=(len(display_tokens) * 0.5 + 2, 10))
    
    # Create annotation matrix if requested
    annot = None
    fmt = '.3f' if annotate_values else None
    if annotate_values:
        # Only annotate if matrix is not too large (to avoid clutter)
        if entropy_matrix.shape[0] <= 30 and entropy_matrix.shape[1] <= 20:
            # annot = np.array([[f'{val:.3f}' for val in row] for row in entropy_matrix])
            annot = entropy_matrix.copy()
        else:
            print("Matrix too large for value annotations, skipping...")
            annot = None
    
    # Create y-axis labels for layers (0 to num_layers-1)
    # Show all layers, but if there are too many, show every Nth layer
    if num_layers * 2 <= 50:
        # Show all layers
        # yticklabels = [str(i) for i in range(num_layers)]
        yticklabels = [str(i) for i in range(0, num_layers * 2, 2)]
    else:
        # Show every 5th layer if too many
        step = max(1, num_layers // 50)
        #yticklabels = [str(i) if i % step == 0 else '' for i in range(num_layers)]
        yticklabels = [str(i) if i % step == 0 else '' for i in range(0, num_layers * 2, 2)]
    
    ax = sns.heatmap(
        entropy_matrix,
        cmap="YlGn",
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={"label": "Uncertainty"},
        xticklabels=display_tokens,
        yticklabels=yticklabels,
        vmin=0,
        vmax=1,
        annot=annot,
        fmt=fmt,
    )

    # Reverse y-axis so layer 0 is at bottom, highest layer at top
    ax.invert_yaxis()
    
    # Set axis labels
    ax.set_ylabel("Layer Index", fontsize=12)

    # 调整 token label 样式 - place x-axis labels at bottom
    ax.set_xticklabels(display_tokens, rotation=30, ha='right', fontsize=10)
    ax.xaxis.set_ticks_position('top')  # Ensure x-axis labels are at bottom
    ax.xaxis.set_label_position('top')

    # 上下留白 & 美化
    plt.subplots_adjust(bottom=0.2, top=0.95)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

def register_entropy_hooks(model):
    def hook_fn(layer, input, output):
        # output: hidden_states, shape: [B, seq_len, hidden_dim]
        hidden = output[0]  # 只取 hidden_states
        normed = model.model.norm(hidden)  # norm 是在语言模型上
        logits = model.lm_head(normed)  # [B, seq_len, vocab]
        last_logits = logits[:, -1, :]  # 当前 token logits
        top_k = 10
        topk_logits, _ = torch.topk(last_logits, top_k, dim=-1)
        probs = F.softmax(topk_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1) / torch.log(torch.tensor(10.0))
        layer_entropy_list.append(entropy.item())  # append 每一层的值

    # hook 偶数层
    for idx, layer in enumerate(model.model.layers):
        if idx % 2 == 0:
            layer.register_forward_hook(hook_fn)

def main(model, processor, tokenizer, annotation, messages_fn, image_folder, vis_dir):
    input_image = annotation["input_image"]
    image_path = os.path.join(image_folder, input_image)
    question = annotation["question"]
    options = annotation.get('options', [])
    gt_answer = options[0]
    shared_messages = messages_fn(image_path, question)
    question_prompt = processor.apply_chat_template(
        shared_messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(shared_messages)
    question_inputs = processor(
        text=[question_prompt],
        images=image_inputs,
        return_tensors="pt",
    )
    question_inputs = question_inputs.to(model.device)
    input_ids = question_inputs.input_ids
    generated = input_ids
    attention_mask = question_inputs.attention_mask
    answer_ids = tokenizer(gt_answer, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    print(f"Answer length: {answer_ids.shape[1]}")
    
    global layer_entropy_list  # Make it global so the hook can access it
    layer_entropy_list = []
    entropy_matrix = []
    past_key_values = None
    
    with torch.no_grad():
        for i in range(answer_ids.shape[1]):
            layer_entropy_list.clear()

            next_token = answer_ids[:, i:i+1]

            # Forward pass with current generated sequence
            if past_key_values is None:
                outputs = model(
                    input_ids=generated, 
                    attention_mask=attention_mask, 
                    use_cache=True,
                    pixel_values=question_inputs.pixel_values,
                    image_grid_thw=question_inputs.image_grid_thw
                )
            else:
                outputs = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    pixel_values=None,  # Don't reprocess image
                    image_grid_thw=question_inputs.image_grid_thw  # Keep image grid info
                )
            past_key_values = outputs.past_key_values
            
            entropy_matrix.append(layer_entropy_list.copy())

            # Get next token from logits
            logits = outputs.logits[:, -1, :].float()  # current time step's logits, last token
            
            # Append new token to generated sequence
            generated = torch.cat((generated, next_token), dim=1)
            attention_mask = torch.cat((attention_mask, torch.ones_like(next_token)), dim=1)

            token_text = tokenizer.decode(next_token[0], skip_special_tokens=False)
            print(f"Step {i+1}: token = '{token_text}'")
    
    del outputs, logits, past_key_values
    torch.cuda.empty_cache()
    entropy_array = np.array(entropy_matrix)  # [T, L]
    vis_save_path = os.path.join(vis_dir, input_image)
    visualize_entropy_heatmap(
        entropy_matrix, 
        generated, 
        tokenizer, 
        save_path=vis_save_path, 
        annotate_values=True,
        prompt_length=input_ids.shape[1]  # Length of the initial prompt
    )

if __name__ == "__main__":
    model_name = "/data1/pinci/ckpt/huggingface/Qwen2.5-VL-3B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
                attn_implementation="flash_attention_2"
    ).eval()

    register_entropy_hooks(model)

    messages = lambda img, question: [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": question}]}]

    annotation_file = "/data1/pinci/datasets/zoom_eye_data/vstar/annotation_vstar.json"
    with open(annotation_file, 'r') as f:
        all_annotations = json.load(f)
    image_folder = "/data1/pinci/datasets/zoom_eye_data/vstar"
    vis_dir = "vis/entropy/vstar"
    os.makedirs(vis_dir, exist_ok=True)

    for annotation in tqdm(all_annotations):
        main(model, processor, tokenizer, annotation, messages, image_folder, vis_dir)