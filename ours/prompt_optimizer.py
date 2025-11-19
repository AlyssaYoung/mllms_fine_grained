# prompt_optimizer.py (Corrected Adam implementation)

import torch
import torch.nn.functional as F
from prompt_builder import build_option_prompt, prepare_inputs, extract_option_ids


def optimize_visual_prompt(
    model,
    processor,
    annotation,
    shared_messages,
    image_inputs,
    question_inputs,
    visual_prompt,
    cfg,
):
    """
    Run T-1 rounds of gradient-based optimization for visual_prompt.
    Uses strict Adam with bias correction (matching original script).
    """

    gt_option = annotation["options"][0]

    # ===== Prepare full input with ground truth option =====
    gt_prompt = build_option_prompt(shared_messages, gt_option, processor)
    full_inputs = prepare_inputs(processor, gt_prompt, image_inputs, device=model.device)

    q_len = question_inputs.input_ids.shape[1]
    gt_o_ids, gt_o_len = extract_option_ids(full_inputs, q_len)

    # ===== Adam states (exactly like original) =====
    state = {
        "m": torch.zeros_like(visual_prompt),
        "s": torch.zeros_like(visual_prompt),
    }
    hyperparams = {"lr": cfg.lr, "t": 1}

    beta1, beta2, eps = cfg.beta1, cfg.beta2, cfg.eps

    with torch.set_grad_enabled(True):
    # ===== T-1 update steps =====
        for _ in range(cfg.T - 1):

            out = model(
                input_ids=full_inputs.input_ids,
                attention_mask=full_inputs.attention_mask,
                pixel_values=full_inputs.pixel_values,
                image_grid_thw=full_inputs.image_grid_thw,
                use_cache=False,
            )
            logits = out.logits

            option_logits = logits[:, q_len - 1 : q_len + gt_o_len - 1]
            option_logits = option_logits.reshape(-1, model.config.vocab_size)

            loss = F.cross_entropy(option_logits.float(), gt_o_ids.reshape(-1))

            grad = torch.autograd.grad(loss, visual_prompt, retain_graph=False)[0]

            # ===== Strict Adam like original code =====
            state["m"] = beta1 * state["m"] + (1 - beta1) * grad
            state["s"] = beta2 * state["s"] + (1 - beta2) * grad.pow(2)

            # Bias correction with exponent
            m_hat = state["m"] / (1 - beta1 ** hyperparams["t"])
            s_hat = state["s"] / (1 - beta2 ** hyperparams["t"])

            visual_prompt.data -= hyperparams["lr"] * m_hat / (torch.sqrt(s_hat) + eps)

            hyperparams["t"] += 1
