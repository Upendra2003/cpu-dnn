# Program the Zybo over JTAG (jumper JP5 on JTAG) and start the ARM application. Works with xsct of 2023.x/2024.x.
#   cd fpga_zybo && xsct vitis/run_app.tcl
set root [file normalize [file dirname [info script]]/..]
set bit  "$root/build/zybo_ibex_lod.bit"
set init [lindex [glob -nocomplain \
                                   "$root/build/vitis_ws/zybo_ibex_pf/hw/ps7_init.tcl" \
                                   "$root/build/vitis_ws/zybo_ibex_pf/*/*/ps7_init.tcl" \
                                   "$root/build/vitis_ws/zybo_ibex_pf/*/*/*/ps7_init.tcl"] 0]
set elf  [lindex [glob -nocomplain "$root/build/vitis_ws/ibex_host/Release/*.elf" \
                                   "$root/build/vitis_ws/ibex_host/build/*.elf"] 0]
foreach {what f} [list bitstream $bit ps7_init.tcl $init "ARM elf" $elf] {
  if {$f eq "" || ![file exists $f]} { error "missing $what (looked under build/)" }
}
connect
targets -set -nocase -filter {name =~ "APU*"}
rst -system
after 1000
targets -set -nocase -filter {name =~ "xc7z010*"}
fpga -file $bit
targets -set -nocase -filter {name =~ "APU*"}
source $init
ps7_init
ps7_post_config
targets -set -nocase -filter {name =~ "*A9*#0"}
dow $elf
con
puts "DONE: board programmed, ARM application running (LD0 lights while an Ibex program has finished)"
