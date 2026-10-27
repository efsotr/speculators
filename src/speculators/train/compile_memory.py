"""Release temporary compiler frame cycles at completed iteration boundaries."""
import gc
import json

import torch
from torch._dynamo.utils import counters

from speculators.train.distributed import get_rank


class CompileCycleCleanup:
    """Collect after new Dynamo graphs, including later shape recompilations.

    Dynamo tracing can leave unreachable Python frames holding real vocabulary
    logits until cyclic GC happens to run. Waiting until after backward/optimizer
    avoids keeping those tensors live into the next forward. This does not flush
    the CUDA allocator or change any reachable autograd tensors. Steady-state
    iterations with no new graph do not run a full Python collection.
    """

    def __init__(self, device_type, phase):
        self.enabled = device_type == 'cuda'
        self.phase = phase
        self.graphs = counters['stats']['unique_graphs']

    def collect_if_compiled(self):
        if not self.enabled:
            return
        graphs = counters['stats']['unique_graphs']
        if graphs == self.graphs:
            return
        new_graphs = graphs - self.graphs
        self.graphs = graphs
        before = torch.cuda.memory_allocated()
        collected = gc.collect()
        print('SPECULATORS_COMPILE_CLEANUP ' + json.dumps(dict(
            phase=self.phase, rank=get_rank(), new_graphs=new_graphs,
            collected_objects=collected,
            released_bytes=before - torch.cuda.memory_allocated(),
        )), flush=True)
