import torch
import torch.distributed as dist
from contextlib import contextmanager

dist.init_process_group("nccl", init_method="env://")
torch.cuda.set_device(dist.get_rank())

@contextmanager
def graph_capture(pool=None, stream=None, capture_error_mode: str = "global", dump_path=None):
    g = torch.cuda.CUDAGraph()
    if dump_path is not None:
        g.enable_debug_mode()
    with torch.cuda.graph(cuda_graph=g, pool=pool, stream=stream, capture_error_mode=capture_error_mode):
        yield g
    if dump_path is not None:
        g.debug_dump(dump_path)

# Placeholder input used for capture
inputs = torch.zeros((5,), device="cuda")

# Warmup before capture
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        dist.all_reduce(inputs, op=dist.ReduceOp.SUM)
torch.cuda.current_stream().wait_stream(s)

rank = dist.get_rank()
# Captures the graph
# To allow capture, automatically sets a side stream as the current stream in the context
with torch.cuda.nvtx.range("capture"):
    with graph_capture(dump_path=f"graph_{rank}.dot") as g:
        dist.all_reduce(inputs, op=dist.ReduceOp.SUM)

# Run the graph
g.replay()
torch.cuda.current_stream().synchronize()
print(inputs)
inputs += 1
g.replay()
torch.cuda.current_stream().synchronize()
print(inputs)