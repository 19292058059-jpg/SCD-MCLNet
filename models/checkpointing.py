from contextlib import contextmanager, nullcontext
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

@contextmanager
def _isolated_bn_buffers(module):
    originals = []
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            saved = (child.running_mean, child.running_var, child.num_batches_tracked)
            originals.append((child, saved))
            (child.running_mean, child.running_var, child.num_batches_tracked) = (value.clone() if value is not None else None for value in saved)
    try:
        yield
    finally:
        for (child, saved) in originals:
            (child.running_mean, child.running_var, child.num_batches_tracked) = saved

class _RepeatableBNContext:

    def __init__(self, module):
        self.module = module
        self.contexts = []

    def __enter__(self):
        context = _isolated_bn_buffers(self.module)
        result = context.__enter__()
        self.contexts.append(context)
        return result

    def __exit__(self, *exception):
        return self.contexts.pop().__exit__(*exception)

def run_block(module, *inputs, enabled=False, concatenate=False):
    function = (lambda *parts: module(torch.cat(parts, dim=1))) if concatenate else module
    if not enabled or not module.training or (not torch.is_grad_enabled()):
        return function(*inputs)
    return checkpoint(function, *inputs, use_reentrant=False, preserve_rng_state=True, context_fn=lambda : (nullcontext(), _RepeatableBNContext(module)))
