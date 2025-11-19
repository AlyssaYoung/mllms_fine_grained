# prompt_builder.py

import torch


def build_question_prompt(shared_messages, processor):
    """Build question prompt without generation prompt."""
    return processor.apply_chat_template(
        shared_messages, tokenize=False, add_generation_prompt=True
    )


def build_option_prompt(shared_messages, option, processor):
    """Append an assistant message with the candidate option."""
    option_messages = shared_messages + [{"role": "assistant", "content": option}]
    return processor.apply_chat_template(
        option_messages, tokenize=False, add_generation_prompt=False
    )


def prepare_inputs(processor, prompt, image_inputs, device):
    """
    Apply processor to (text prompt + images).
    This avoids repeating processor(text.... images...) all over the code.
    """
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        return_tensors="pt",
    )
    return inputs.to(device)


def extract_option_ids(full_inputs, q_len):
    """
    From full prompt including the answer,
    extract just the option token ids and length.
    """
    option_ids = full_inputs.input_ids[:, q_len:]
    o_len = option_ids.shape[1]
    return option_ids, o_len
