"""Scoped author's Ascend fused vision attention, weights/source unchanged.

OfficialUND's earlier deployment deliberately mapped all choices to eager V2.
This separately evaluated numerical candidate restores the author's provided
VisionAscendAttention.forward on the same qkv/proj modules. Not bit-exact.
"""
from contextlib import contextmanager
from types import MethodType
import sys

@contextmanager
def fused_vision(tower):
    states=[]
    try:
        for block in tower.blocks:
            attention=block.attn;module=sys.modules[type(attention).__module__]
            states.append((attention,'forward' in attention.__dict__,attention.__dict__.get('forward')))
            attention.forward=MethodType(module.VisionAscendAttention.forward,attention)
        yield
    finally:
        for attention,had,value in reversed(states):
            if had:attention.forward=value
            else:delattr(attention,'forward')
