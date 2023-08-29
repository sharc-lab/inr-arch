AUTOPILOT_ROOT := /tools/software/xilinx/Vitis_HLS/2022.1
ASSEMBLE_SRC_ROOT := .

IFLAG += -I "${ASSEMBLE_SRC_ROOT}"

IFLAG += -I "${AUTOPILOT_ROOT}/include"
IFLAG += -I "${AUTOPILOT_ROOT}/include/etc"
IFLAG += -I "${AUTOPILOT_ROOT}/include/utils"

IFLAG += -I "${AUTOPILOT_ROOT}/lnx64/tools/fpo_v7_0"
IFLAG += -I "${AUTOPILOT_ROOT}/lnx64/tools/fft_v9_1"
IFLAG += -I "${AUTOPILOT_ROOT}/lnx64/tools/fir_v7_0"
IFLAG += -I "${AUTOPILOT_ROOT}/lnx64/tools/dds_v6_0"
IFLAG += -I "${AUTOPILOT_ROOT}/lnx64/lib/csim"

CFLAG += -L"${AUTOPILOT_ROOT}/lnx64/lib/csim" -lhlsmc++-CLANG39 -Wl,-rpath,"${AUTOPILOT_ROOT}/lnx64/lib/csim" -Wl,-rpath,"${AUTOPILOT_ROOT}/lnx64/tools/fpo_v7_0"
CFLAG += -Wl,--sysroot=/
# https://github.com/conda/conda/issues/7578
                                            
### C sim
IFLAG += -D__SIM_FPO__ -D__SIM_OPENCV__ -D__SIM_FFT__ -D__SIM_FIR__ -D__SIM_DDS__ -D__DSP48E1__


# IFLAG += -DDEBUG_FILE_PRINT=1
IFLAG += -g
CFLAG += -fPIC -fPIE -O3 -Wno-unused-result # -march=native
CFLAG += -lm
CFLAG += -std=c++14
# CFLAG += -fsanitize=address -fno-omit-frame-pointer

# CXX = g++
CXX = ${AUTOPILOT_ROOT}/lnx64/tools/clang-3.9/bin/clang++
# CXX = ${AUTOPILOT_ROOT}/lnx64/tools/gcc/bin/g++


./result: ./*.cpp ./*.h ../inr_hw_lib/inr_hw_lib_v2.h
	$(CXX) ${ASSEMBLE_SRC_ROOT}/*.cpp -o result $(CFLAG) $(IFLAG)

all: ./result
	result

run: ./result
	./result

clean:
	rm -f *.o result

.PHONY: all clean run