open_project -reset inr_synth_test_project

add_files top.h
add_files top.cc

# add_files -tb model_tb.cc
# add_files -tb tb_data

set_top top

open_solution "solution1" -flow_target vivado
set_part xcu50-fsvh2104-2-e
create_clock -period 5 -name default

# csim_design
csynth_design
# cosim_design -O -enable_dataflow_profiling -trace_level all