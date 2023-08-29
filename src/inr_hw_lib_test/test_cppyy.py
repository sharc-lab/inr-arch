import itertools
from pathlib import Path
import os
import sys
import tempfile
import subprocess

from rich.pretty import pprint as pt
import cppyy
import numpy as np
from joblib import Parallel, delayed

from matplotlib import pyplot as plt



def vectorize(f, otypes=None):
    return np.vectorize(f, otypes=otypes)


# if <path> not in os.getenv('LD_LIBRARY_PATH'):
#     os.environ['LD_LIBRARY_PATH'] = ':'.join([os.getenv('LD_LIBRARY_PATH'),'<PATH>'])
#     try:
#         sys.stdout.flush()
#         os.execl(sys.executable,sys.executable, *sys.argv)
#     except OSError as e:
#         print(e)

# def add_path_to_ld_library_path(path: Path):
#     if str(path) not in os.getenv('LD_LIBRARY_PATH'):
#         os.environ['LD_LIBRARY_PATH'] = ':'.join([os.getenv('LD_LIBRARY_PATH'),str(path)])
#         try:
#             sys.stdout.flush()
#             os.execl(sys.executable,sys.executable, *sys.argv)
#         except OSError as e:
#             print(e)

AUTOPILOT_ROOT = Path("/tools/software/xilinx/Vitis_HLS/2022.1")
ASSEMBLE_SRC_ROOT = Path(".")

# add_path_to_ld_library_path(AUTOPILOT_ROOT / "lnx64/lib/csim/")
# add_path_to_ld_library_path(AUTOPILOT_ROOT / "lnx64/tools/fpo_v7_0/")

cppyy.cppdef("#define __SIM_FPO__")
cppyy.cppdef("#define __SIM_OPENCV__")
cppyy.cppdef("#define __SIM_FFT__")
cppyy.cppdef("#define __SIM_FIR__")
cppyy.cppdef("#define __SIM_DDS__")
cppyy.cppdef("#define __DSP48E1__")

cppyy.add_include_path(str(ASSEMBLE_SRC_ROOT))
cppyy.add_include_path(str(AUTOPILOT_ROOT / "include"))
cppyy.add_include_path(str(AUTOPILOT_ROOT / "include/etc"))
cppyy.add_include_path(str(AUTOPILOT_ROOT / "include/utils"))
cppyy.add_include_path(str(AUTOPILOT_ROOT / "lnx64/tools/fpo_v7_0"))
cppyy.add_include_path(str(AUTOPILOT_ROOT / "lnx64/tools/fft_v9_1"))
cppyy.add_include_path(str(AUTOPILOT_ROOT / "lnx64/tools/fir_v7_0"))
cppyy.add_include_path(str(AUTOPILOT_ROOT / "lnx64/tools/dds_v6_0"))
cppyy.add_include_path(str(AUTOPILOT_ROOT / "lnx64/lib/csim"))



# CFLAG += -L"${AUTOPILOT_ROOT}/lnx64/lib/csim" -lhlsmc++-CLANG39 -Wl,-rpath,"${AUTOPILOT_ROOT}/lnx64/lib/csim" -Wl,-rpath,"${AUTOPILOT_ROOT}/lnx64/tools/fpo_v7_0"
cppyy.add_library_path(str(AUTOPILOT_ROOT / "lnx64/lib/csim/"))
cppyy.add_library_path(str(AUTOPILOT_ROOT / "lnx64/tools/fpo_v7_0/"))


cppyy.load_library(str(AUTOPILOT_ROOT / "lnx64/tools/fpo_v7_0/libgmp.so.7"))
cppyy.load_library(str(AUTOPILOT_ROOT / "lnx64/tools/fpo_v7_0/libIp_floating_point_v7_0_bitacc_cmodel.so"))

with tempfile.TemporaryDirectory() as lib_link_dir:
    subprocess.run(["ld", str(AUTOPILOT_ROOT / "lnx64/lib/csim/libhlsmc++-CLANG39.so"), "-rpath", str(AUTOPILOT_ROOT / "lnx64/tools/fpo_v7_0/"), "-shared", "-o", str(lib_link_dir) + "/libhlsmc++-CLANG39.so"])
    cppyy.load_library(str(lib_link_dir) + "/libhlsmc++-CLANG39.so")


SCRIPT_DIR = Path(os.path.dirname(__file__))
LIB_DIR = Path(os.path.dirname(__file__)).parent / "inr_hw_lib"
LIB_H_FP = LIB_DIR / "inr_hw_lib.h"

cppyy.add_include_path("/tools/software/xilinx/Vitis_HLS/latest/include/")

cppyy.include(SCRIPT_DIR / "main.h")
cppyy.include(str(LIB_H_FP))



x = np.linspace(-5, 5, 1000).astype(np.float32)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
y_true = sigmoid(sigmoid(x))

W = [32]
I = range(5, 30)

combos = list(itertools.product(W, I))

cppyy.gbl.ap_fixed.__release_gil__ = True

def test_tanh(W, I, x, gbl):
    F_TYPE = gbl.ap_fixed[W, I]

    x_fixed = x.astype(F_TYPE)

    activation_sigmoid_v = vectorize(gbl.activation_sigmoid[F_TYPE], otypes=[F_TYPE])

    y_fixed = activation_sigmoid_v(activation_sigmoid_v(x_fixed))
    y_float = y_fixed.astype(float).astype(np.float32)
    return y_float

y_floats = Parallel(n_jobs=16, backend="multiprocessing", verbose=11)(delayed(test_tanh)(W, I, x, cppyy.gbl) for W, I in combos)
y_floats = np.array(y_floats)

y_error = np.abs(y_floats - y_true)

y_error_mae = np.mean(y_error, axis=1)

fig, ax = plt.subplots(1, 3, figsize=(18, 8))
ax[0].plot(x, y_true, label="true")
for idx, (w, i) in enumerate(combos):
    ax[0].plot(x, y_floats[idx], label=f"W={w}, I={i}")
ax[0].legend()
ax[0].set_xlabel("x")
ax[0].set_ylabel("sigmoid(x)")
ax[0].set_title("Function")

for idx, (w, i) in enumerate(combos):
    ax[1].plot(x, y_error[idx], label=f"W={w}, I={i}")
ax[1].legend()
ax[1].set_xlabel("x")
ax[1].set_ylabel("sigmoid(x) - hw_sigmoid(x)")
ax[1].set_title("Error")

# x-axis is I, y-axis is error sum, circle markers with line
ax[2].plot([i for w, i in combos], y_error_mae, marker="o", linestyle="-")
ax[2].set_title("MAE vs. I")
ax[2].set_xlabel("I")
ax[2].set_ylabel("MAE across interval [-5, 5]")

plt.suptitle(f"Sigmoid")
plt.tight_layout()

plt.savefig(SCRIPT_DIR / "sigmoid.png", dpi=300)
