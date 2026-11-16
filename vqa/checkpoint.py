"""Read our legacy NPU projector weights on CPU without a CANN dependency."""
import torch

def _legacy_npu_cpu(storage, offset, size, stride, requires_grad, hooks, storage_format):
    # torch_npu serializes the CPU copy's storage/shape/stride plus the original NPU format.
    # We rebuild exactly that CPU copy; no device code or arbitrary pickled code is executed.
    return torch._utils._rebuild_tensor_v2(storage,offset,size,stride,requires_grad,hooks)

def load_tensor_state(path):
    with torch.serialization.safe_globals([(_legacy_npu_cpu,'torch_npu.utils.storage._rebuild_npu_tensor'),
                                           (int,'torch_npu.npu._format.Format')]):
        return torch.load(path,map_location='cpu',weights_only=True)
