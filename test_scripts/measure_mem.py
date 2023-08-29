import subprocess
import psutil
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

from siren_pytorch import SirenNet

from copy import deepcopy
import tracemalloc
import gc
import os
from contextlib import contextmanager


net = SirenNet(
    dim_in=2,  # input dimension, ex. 2d coor
    dim_hidden=1024,  # hidden dimension
    dim_out=3,  # output dimension, ex. rgb value
    num_layers=5,  # number of layers
    final_activation=None,  # activation of final layer (nn.Identity() for direct output)
    w0_initial=30.0,  # different signals may require different omega_0 in the first layer - this is a hyperparameter
)

coor = torch.randn(64, 2, requires_grad=True)


def trace_gpu_v2(input_net, input_coor, device: str = "cuda:0"):
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated(device=device)
    torch.cuda.reset_max_memory_cached(device=device)
    torch.cuda.reset_peak_memory_stats(device=device)
    torch.cuda.reset_accumulated_memory_stats(device=device)

    net = deepcopy(input_net).to(device)
    coor = deepcopy(input_coor).to(device)
    net(coor)
    for i in range(3):
        torch.autograd.grad(
            out[..., i] / 256, coor, torch.ones_like(out[..., i]), create_graph=True
        )

    peak_mem_gpu = torch.cuda.max_memory_allocated(device=device)
    # print(f"Peak memory usage on GPU: {peak_mem_gpu / (1024 ** 3):.03f} GB")
    # MB
    print(f"Peak memory usage on GPU: {peak_mem_gpu / (1024 ** 2):.03f} MB")


def trace_gpu(func, device: str):
    torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=device)
    torch.cuda.reset_accumulated_memory_stats(device=device)

    # SATRT USER CODE
    func()
    # END USER CODE

    peak_mem_gpu = torch.cuda.max_memory_allocated(device=device)
    return peak_mem_gpu


def get_hwm(pid: int):
    gc.collect()
    pid = os.getpid()
    hwm = 0
    with open(f"/proc/{pid}/status", "r") as f:
        for line in f:
            if line.startswith("VmHWM"):
                hwm = int(line.split(":")[1].strip().split()[0]) * 1024
    return hwm


def get_rss(pid: int):
    gc.collect()
    pid = os.getpid()
    rss = 0
    with open(f"/proc/{pid}/status", "r") as f:
        for line in f:
            if line.startswith("VmRSS"):
                rss = int(line.split(":")[1].strip().split()[0]) * 1024
    return rss

dummy = []
def trace_cpu(func):
    gc.collect()
    pid = os.getpid()

    dummy.append(bytearray(1024 ** 3 // 2))

    hwm_start = get_hwm(pid)
    rss_start = get_rss(pid)

    # SATRT USER CODE
    func()
    # END USER CODE

    hwm_end = get_hwm(pid)
    rss_end = get_rss(pid)

    assert hwm_end > hwm_start
    total_memory = hwm_end - rss_start
    return total_memory


# torch.cuda.init()
# trace_gpu(net, coor, device="cuda:0")
# trace_cpu(net, coor)


def my_cpu_code(input_net, input_coor):
    net = deepcopy(input_net).to("cpu")
    coor = deepcopy(input_coor).to("cpu")
    out = net(coor)
    for i in range(3):
        g = torch.autograd.grad(
            out[..., i] / 256, coor, torch.ones_like(out[..., i]), create_graph=True
        )


def my_gpu_code(input_net, input_coor, device: str):
    net = deepcopy(input_net).to(device)
    coor = deepcopy(input_coor).to(device)
    out = net(coor)
    for i in range(3):
        g = torch.autograd.grad(
            out[..., i] / 256, coor, torch.ones_like(out[..., i]), create_graph=True
        )


N_WARMPUP = 10
N_TRIALS = 10

mem_used_cpu_list = []
mem_used_gpu_list = []


for i in range(N_WARMPUP + N_TRIALS):
    mem_used_cpu = trace_cpu(lambda: my_cpu_code(net, coor))
    if i >= N_WARMPUP:
        mem_used_cpu_list.append(mem_used_cpu)

for i in range(N_WARMPUP + N_TRIALS):
    mem_used_gpu = trace_gpu(lambda: my_gpu_code(net, coor, "cuda:0"), "cuda:0")
    if i >= N_WARMPUP:
        mem_used_gpu_list.append(mem_used_gpu)


mem_used_cpu_avg = sum(mem_used_cpu_list) / len(mem_used_cpu_list)
mem_used_gpu_avg = sum(mem_used_gpu_list) / len(mem_used_gpu_list)

print(f"Peak memory usage on CPU: {mem_used_cpu_avg / (1024 ** 2):.03f} MB")
print(f"Peak memory usage on GPU: {mem_used_gpu_avg / (1024 ** 2):.03f} MB")

# mem_used_cpu = trace_cpu(lambda: my_cpu_code(net, coor))
# print(f"Peak memory usage on CPU: {mem_used_cpu / (1024 ** 2):.03f} MB")

# mem_used_gpu = trace_gpu(lambda: my_gpu_code(net, coor, "cuda:0"), "cuda:0")
# print(f"Peak memory usage on GPU: {mem_used_gpu / (1024 ** 2):.03f} MB")
