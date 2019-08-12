#################
#    HLS4ML
#################
array set opt {
  csim   1
}

foreach arg $::argv {
  foreach o [lsort [array names opt]] {
    regexp "$o +(\\w+)" $arg unused opt($o)
  }
}

open_project -reset hls_prj
set_top lenet5
add_files firmware/full_seq2seq.cpp -cflags "-I[file normalize ./nnet_utils] -std=c++0x"
add_files -tb tb_seq2seq.cpp -cflags "-I[file normalize ./nnet_utils] -std=c++0x"
# add_files -tb test_images

open_solution -reset "solution1"
catch {config_array_partition -maximum_size 4096}
set_part {xcku095-ffvb2104-1-c}
# set_part {xcku115-flvb2104-2-e}
create_clock -period 10 -name default

if {$opt(csim)} {
  puts "***** C SIMULATION *****"
  csim_design
}

exit