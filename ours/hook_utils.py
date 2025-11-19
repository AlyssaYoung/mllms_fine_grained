from contextlib import contextmanager


def visual_prompt_hook(model, visual_prompt):
    """
    Inject visual_prompt into model.visual output using forward hook.
    Automatically removed when exiting 'with' block.
    """

    def _hook(module, inp, out):
        return out + visual_prompt.to(out.device)

    handle = model.visual.register_forward_hook(_hook)

    @contextmanager
    def manager():
        try:
            yield
        finally:
            handle.remove()

    return manager()
