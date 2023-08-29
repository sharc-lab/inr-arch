open_project -reset proj

add_files kernel.cpp

set_top kernel_mm_v3

open_solution "solution1" -flow_target vitis
set_part xczu9eg-ffvb1156-2-e
create_clock -period 3.33 -name default

csynth_design