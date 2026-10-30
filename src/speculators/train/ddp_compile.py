"""Lifecycle cleanup for Python DDP reducer CUDA Graphs.

Transferred from the drafter bench used in SpecDecode runs 175 and 176.
Captured NCCL work retains communicator references until its graphs are reset.
"""

import torch


def release_cuda_graphs():
    """Reset captured graphs before ProcessGroupNCCL collective shutdown."""
    from torch._inductor.cudagraph_trees import get_container, reset_cudagraph_trees

    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    manager = get_container(torch.cuda.current_device()).tree_manager
    if manager is not None:
        queue = list(manager.get_roots())
        seen = set()
        while queue:
            node = queue.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            queue.extend(c for children in node.children.values() for c in children)
            if node.graph is not None:
                node.graph.reset()
        if manager.graph is not None:
            manager.graph.reset()
    reset_cudagraph_trees()
    torch._dynamo.reset()
