# option_evaluator.py

import torch
import torch.nn.functional as F

from prompt_builder import (
    build_option_prompt,
    prepare_inputs,
    extract_option_ids,
)


def evaluate_all_options(
    model,
    processor,
    question_inputs,
    shared_messages,
    options,
    image_inputs,
):
    """
    Step 1: Run question-only forward to obtain question_logits + KV cache.
    Step 2: For each option, attach option prompt and compute loss based on KV cache.
    """

    # Step-1: Question only forward (with use_cache=True)
    with torch.no_grad():
        q_out = model(
            input_ids=question_inputs.input_ids,
            attention_mask=question_inputs.attention_mask,
            pixel_values=question_inputs.pixel_values,
            image_grid_thw=question_inputs.image_grid_thw,
            use_cache=True,
            return_dict=True,
        )
        question_logits = q_out.logits
        past_kv = q_out.past_key_values
        rope_deltas = q_out.rope_deltas

        model.rope_deltas = rope_deltas.to(model.device)

        q_len = question_inputs.input_ids.shape[1]

        losses = []

        for opt in options:
            # Prepare option prompt
            opt_prompt = build_option_prompt(shared_messages, opt, processor)
            full_inputs = prepare_inputs(processor, opt_prompt, image_inputs, model.device)

            # Extract option ids and len
            opt_ids, opt_len = extract_option_ids(full_inputs, q_len)

            # Build full attention mask (question + option)
            opt_attn_mask = torch.ones_like(opt_ids, dtype=question_inputs.attention_mask.dtype)
            full_mask = torch.cat([question_inputs.attention_mask, opt_attn_mask], dim=1)

            # KV cache reuse
            cache_position = torch.arange(
                q_len,
                q_len + opt_ids.shape[1],
                device=model.device,
                dtype=torch.long
            )

            out = model(
                input_ids=opt_ids,
                attention_mask=full_mask,
                past_key_values=past_kv,
                cache_position=cache_position,
                use_cache=False,
                pixel_values=None,
                image_grid_thw=question_inputs.image_grid_thw,
                return_dict=True,
            )

            # Compose logits: first token from question_logits, remaining from option forward
            logits = torch.cat([question_logits[:, -1:], out.logits[:, :-1]], dim=1)
            logits = logits.reshape(-1, model.config.vocab_size)

            loss = F.cross_entropy(logits.float(), opt_ids.reshape(-1))
            losses.append(loss)

    losses = torch.stack(losses)
    chosen = losses.argmin().item()
    return chosen
