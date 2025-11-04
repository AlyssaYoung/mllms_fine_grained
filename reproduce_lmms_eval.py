import re
import os
import torch
import json
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

def build_prompt(entry: dict, processor, image_folder) -> str:
    question_text = entry["question"]
    options = entry["options"]
    # 提取括号选项，例如 "(A) rubber (B) cotton ..."
    # options = re.findall(r"\([A-D]\)\s*([^()]+?)(?=\s*\([A-D]\)|$)", question)
    if options:
        # 切掉 "(A)" 之前的题干
        # question_text = question.split("(A)")[0].strip()
        # 重建为多行格式：A. xxx
        option_letters = ["A", "B", "C", "D"]
        options_formatted = [f"{letter}. {opt.strip()}"
                             for letter, opt in zip(option_letters, options)]
        question = question_text + "\n" + "\n".join(options_formatted)
    # 拼接后缀提示语， instruct 模型直接生成字母
    post_prompt = "\nAnswer with the option's letter from the given choices directly."
    content = question + post_prompt

    system_prompt = "You are a helpful assistant."
    image_path = os.path.join(image_folder, entry["input_image"])
    message = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": content},
            ],
        },
    ]

    prompt_str = processor.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt_str, message

def extract_answer_letter(response: str) -> str | None:
    """
    模仿 lmms‑eval 的 extract_answer_letter:contentReference[oaicite:9]{index=9}，
    从模型输出中提取答案字母（A–D）。
    """
    resp = response.strip().upper()
    patterns = [
        r"^([A-D])\s*[\.)\]]*",                       # A  或 A.
        r"(?:THE\s+)?(?:ANSWER|CHOICE|OPTION)(?:\s+IS)?[\s:]+([A-D])",
        r"\(([A-D])\)",                               # (A)
        r"([A-D])\s*(?:\.|\)|\])",                    # A) 或 A.
        r"(?:^|\s)([A-D])(?:\s|$)",
    ]
    for pattern in patterns:
        m = re.search(pattern, resp, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    letters = re.findall(r"[A-D]", resp)
    if len(letters) == 1:
        return letters[0]
    if resp and resp[0] in "ABCD":
        return resp[0]
    return None

if __name__ == "__main__":
    model_name = "/root/autodl-tmp/ckpt/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name, max_pixels=12845056)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
                attn_implementation="flash_attention_2"
    ).eval()

    annotation_file = "/root/autodl-tmp/dataset/zoom_eye_data/zoom_eye_data/vstar/annotation_vstar.json"
    image_folder = "/root/autodl-tmp/dataset/zoom_eye_data/zoom_eye_data/vstar"
    with open(annotation_file, 'r') as f:
        all_annotations = json.load(f)

    correct = 0
    res_dict = {"A": 0, "B": 1, "C": 2, "D": 3}
    answers_file = "eval/answers/vstar/Qwen2.5-VL-7B-Instruct/reproduce_lmms_eval.jsonl"
    results_file = open(answers_file, 'w')
    for sample in tqdm(all_annotations):
        prompt_str, message = build_prompt(sample, processor, image_folder)
        # print(prompt_str)
        # print(message)
        image_inputs, video_inputs = process_vision_info([message])
        inputs = processor(text=[prompt_str],
                        images=image_inputs,
                        return_tensors="pt")
        # 按 vstar_bench 配置生成答案:contentReference[oaicite:10]{index=10}
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            temperature=0.0,
            top_p=1.0,
            num_beams=1,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        # 只取新生成的部分
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = processor.tokenizer.decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # print(response)
        pred_letter = extract_answer_letter(response)
        # print(pred_letter)
        if pred_letter is None:
            print(response)
            sample["output"] = -1
        if pred_letter is not None:
            sample["output"] = res_dict[pred_letter]
        # if pred_letter == sample["label"].strip().upper():
        # print(pred_letter)
        if pred_letter == "A":
            correct += 1
        results_file.write(json.dumps(sample) + "\n")
    results_file.close()

    accuracy = correct / len(all_annotations)
    print(f"V*-Bench accuracy: {accuracy:.4f}")
