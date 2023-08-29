from functools import cached_property
import os
import subprocess
import networkx as nx
from pathlib import Path
import pickle
import xml.etree.ElementTree as ET


from src.code_gen import code_gen_v2

SCRIPT_DIR = Path(__file__).parent
TEST_FILES_DIR = SCRIPT_DIR / "test_files"


FPX_Q_VALUES = []

class FPX():
    def __init__(self, W=32, I=16, Q="AP_TRN", O="AP_WRAP"):
        self.W = W
        self.I = I

        if I > 33:
            raise Exception("I must be <= 33")
        if W - I > 32:
            raise Exception("W-I must be <= 32")


SUPPORTED_FPGA_PARTS = [
    "xcu50-fsvh2104-2-e",
    "xcu280-fsvh2892-2L-e",
    "xczu9eg-ffvb1156-2-e"
    # "xilinx_u280_gen3x16_xdma_1_202211_1"
]


class Project:
    def __init__(
        self,
        name: str,
        G: nx.DiGraph,
        build_dir: Path,
        float_or_fixed: str = "fixed",
        fpx: FPX =  FPX(W=32, I=16, Q="AP_TRN", O="AP_WRAP"),
        dataflow: bool = False,
        clock_speed: float = 3.33,
        fpga_part: str = "xcu50-fsvh2104-2-e",
        n_jobs: int = 1,
        vitis_hls_bin: str = "vitis_hls",
        mm_p: int = 1,
        mm_version: str = "v2"
    ):
        self.name = name
        self.G = G
        self.build_dir = build_dir
        
        self.float_or_fixed = float_or_fixed
        if self.float_or_fixed not in ["float", "fixed"]:
            raise ValueError("float_or_fixed must be either 'float' or 'fixed'")
        self.fpx = fpx

        self.dataflow = dataflow

        self.clock_speed = clock_speed
        if self.clock_speed <= 0:
            raise ValueError("clock_speed must be > 0")

        self.fpga_part = fpga_part
        if self.fpga_part not in SUPPORTED_FPGA_PARTS:
            raise ValueError(f"user specified fpga_part {self.fpga_part} is not supported, supported fpga parts are {SUPPORTED_FPGA_PARTS}")

        self.n_jobs = n_jobs
        if self.n_jobs <= 0:
            raise ValueError("n_jobs must be > 0")
        
        self.vitis_hls_bin = vitis_hls_bin

        self.mm_p = mm_p
        if self.mm_p <= 1:
            raise ValueError("mm_p must be > 1")
        
        SUPPORTED_MM_VERSIONS = ["basic", "v2"]
        self.mm_version = mm_version
        if self.mm_version not in SUPPORTED_MM_VERSIONS:
            raise ValueError("mm_version must be one of {SUPPORTED_MM_VERSIONS}")

    @cached_property
    def model_dir(self):
        return self.build_dir / self.name

    def gen_hw_model(self, fifo_depths: dict[str, int] = {}):
        os.makedirs(self.build_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        code = code_gen_v2.gen_model_cpp(
            self.name,
            self.G,
            fifo_depths=fifo_depths,
            mm_p=self.mm_p,
        )
        code_fp = self.model_dir / "model.cpp"
        code_fp.write_text(code)

        model_top_declaration = code_gen_v2.gen_model_declearation(self.name, self.G)
        h = code_gen_v2.MODEL_H_TEMPLATE().render(
            float_or_fixed=self.float_or_fixed,
            fpx=self.fpx,
            model_top_declaration=model_top_declaration,
        )
        h_fp = self.model_dir / "model.h"
        h_fp.write_text(h)

        lib_fp = code_gen_v2.LIB_FP
        lib_new_fp = self.model_dir / "inr_hw_lib.h"
        lib_new_fp.write_text(lib_fp.read_text())

    def gen_testbench(self):
        os.makedirs(self.build_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        testbench_code = code_gen_v2.gen_testbench(self.name, self.G)
        testbench_fp = self.model_dir / "testbench.cpp"
        testbench_fp.write_text(testbench_code)

    def gen_testbench_data(self):
        os.makedirs(self.build_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        tb_data_dir = self.model_dir / "testbench_data"
        os.makedirs(tb_data_dir, exist_ok=True)

        # copy test_files/testbench_data/* to tb_data_dir
        for fp in (TEST_FILES_DIR / "testbench_data").iterdir():
            new_fp = tb_data_dir / fp.name
            new_fp.write_bytes(fp.read_bytes())

    def gen_makefile(self):
        os.makedirs(self.build_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        makefile_code = code_gen_v2.MAKEFILE_TEMPLATE().render({})
        makefile_fp = self.model_dir / "makefile"
        makefile_fp.write_text(makefile_code)
    
    def gen_vitis_hls_tcl_script(self):
        os.makedirs(self.build_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        vitis_hls_tcl_script_code = code_gen_v2.RUN_HLS_TEMPLATE().render({
            "clock_speed": self.clock_speed,
            "fpga_part": self.fpga_part,
        })
        vitis_hls_tcl_script_fp = self.model_dir / "run_hls.tcl"
        vitis_hls_tcl_script_fp.write_text(vitis_hls_tcl_script_code)


    def build_and_run_testbench(self):
        files_to_check = [
            self.model_dir / "testbench.cpp",
            self.model_dir / "model.h",
            self.model_dir / "model.cpp",
            self.model_dir / "inr_hw_lib.h",
            self.model_dir / "makefile",
        ]

        for fp in files_to_check:
            if not fp.exists():
                raise Exception(f"{fp} does not exist. Make sure you call the gen_<...> functions to generate the model and testbench source code.")

        proc_build = subprocess.run(["make", "result", "-B"], cwd=self.model_dir, capture_output=True)

        if proc_build.returncode != 0:
            print(f"return code: {proc_build.returncode}")
            print(proc_build.stdout.decode("utf-8"))
            print(proc_build.stderr.decode("utf-8"))
            raise Exception("Testbench build failed.")
        else:
            print(proc_build.stdout.decode("utf-8"))

        proc_run = subprocess.run(["./result"], cwd=self.model_dir, capture_output=True)
        if proc_run.returncode != 0:
            print(f"return code: {proc_run.returncode}")
            print(proc_run.stdout.decode("utf-8"))
            print(proc_run.stderr.decode("utf-8"))
            raise Exception("Testbench execution failed.")
        else:
            print(proc_run.stdout.decode("utf-8"))

    def gather_testbench_data(self):
        testbench_mae_fp = self.model_dir / "testbench_mae.txt"
        testbench_mae_text = testbench_mae_fp.read_text()
        testbench_mae = float(testbench_mae_text.split()[1])
        
        testbench_data = {
            "testbench_mae": testbench_mae,
        }

        return testbench_data

    def run_vitis_hls_synthesis(self, verbose=False):

        files_to_check = [
            self.model_dir / "run_hls.tcl",
            self.model_dir / "model.h",
            self.model_dir / "model.cpp",
            self.model_dir / "inr_hw_lib.h",
        ]

        for fp in files_to_check:
            if not fp.exists():
                raise Exception(f"{fp} does not exist. Make sure you call the gen_<...> functions to generate the model and testbench source code.")

        proj_tcl_file = str((self.model_dir / "run_hls.tcl").resolve())

        print("Launching HLS synthesis...")
        # proc = subprocess.run(["vitis_hls", proj_tcl_file], cwd=self.model_dir, capture_output=True)
        proc = subprocess.run([self.vitis_hls_bin, proj_tcl_file], cwd=self.model_dir, capture_output=True)

        if proc.returncode != 0:
            print(f"return code: {proc.returncode}")
            if verbose:
                print(proc.stdout.decode("utf-8"))
                print(proc.stderr.decode("utf-8"))
            raise Exception("Vitis HLS synthesis failed.")
        else:
            if verbose:
                print(proc.stdout.decode("utf-8"))

    def gather_synthesis_data(self):
        synth_report_fp = self.model_dir / f"model_top_vitis_hls_project" / "solution1" / "syn" / "report" / f"model_top_csynth.xml"
        if not synth_report_fp.exists():
            raise Exception(f"Can't find synthesis report file: {synth_report_fp}")
        
        with open(synth_report_fp, "r") as f:
            xml_content_top = f.read()
        
        top_root = ET.fromstring(xml_content_top)
        clock_period = float(top_root.find("UserAssignments").find("TargetClockPeriod").text)
        worst_case_runtime_cycles = float(top_root.find("PerformanceEstimates").find("SummaryOfOverallLatency").find("Worst-caseLatency").text)
        worst_case_runtime_ns = worst_case_runtime_cycles * clock_period
        worst_case_runtime = worst_case_runtime_ns / 1e9


        bram = int(top_root.find("AreaEstimates").find("Resources").find("BRAM_18K").text)
        dsp = int(top_root.find("AreaEstimates").find("Resources").find("DSP").text)
        ff = int(top_root.find("AreaEstimates").find("Resources").find("FF").text)
        lut = int(top_root.find("AreaEstimates").find("Resources").find("LUT").text)
        uram = int(top_root.find("AreaEstimates").find("Resources").find("URAM").text)

        synth_data = {
            "clock_period": clock_period,
            "worst_case_runtime_cycles": worst_case_runtime_cycles,
            "worst_case_runtime_ns": worst_case_runtime_ns,
            "worst_case_runtime": worst_case_runtime,
            "bram": bram,
            "dsp": dsp,
            "ff": ff,
            "lut": lut,
            "uram": uram,
        }

        return synth_data
