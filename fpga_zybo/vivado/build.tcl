# Build the Zybo (XC7Z010) design: Zynq PS7 + Ibex (RV32IC, no multiplier, + LOD) with 192 KB block RAM.
#
#   cd fpga_zybo
#   vivado -mode batch -source vivado/build.tcl                     # LOD core: bitstream + XSA + reports
#   vivado -mode batch -source vivado/build.tcl -tclargs nolod      # stock Ibex (no LOD): reports only
#   vivado -mode batch -source vivado/build.tcl -tclargs lod 100    # optional: another PL clock in MHz
#
# Outputs (variant = lod | nolod):
#   build/vivado_<variant>/                 the Vivado project
#   build/zybo_ibex_<variant>.xsa           hardware platform for Vitis (bitstream included; lod only)
#   build/zybo_ibex_<variant>.bit
#   reports/<variant>/*.rpt                 utilization (hierarchical), timing, power (hierarchical)
#   reports/fpga_summary_<variant>.csv      the key numbers in one line (read by host/collect_reports.py)
#
# Needs the Digilent board files (board part digilentinc.com:zybo:part0:*) for the PS7 DDR/MIO preset;
# see README.md, step 1.

set variant [expr {[llength $argv] > 0 ? [lindex $argv 0] : "lod"}]
set fclk    [expr {[llength $argv] > 1 ? [lindex $argv 1] : 50}]
if {$variant ni {lod nolod}} { error "variant must be lod or nolod" }

set here   [file normalize [file dirname [info script]]]
set root   [file normalize "$here/.."]
set build  "$root/build"
set proj   "$build/vivado_$variant"
set repdir "$root/reports/$variant"
file mkdir $build $repdir

set part "xc7z010clg400-1"
create_project zybo_ibex_$variant $proj -part $part -force

# ---- board part (PS7 preset: DDR3 timing, MIO, UART1 on the USB port)
set boards [get_board_parts -quiet digilentinc.com:zybo:part0:*]
if {[llength $boards] == 0} {
  error "Digilent Zybo board files not found. Install them (README.md step 1), then re-run."
}
set board [lindex [lsort $boards] end]
set_property board_part $board [current_project]
puts "INFO: using board part $board, PL clock $fclk MHz, variant $variant"

# ---- RTL
set files {}
foreach f [split [string trim [read [open "$root/rtl/files.f"]]] "\n"] {
  set f [string trim $f]
  if {$f eq ""} continue
  # FPGA clock gate: BUFGCE instead of the generic latch-based gate
  if {[string match "*prim_generic_clock_gating*" $f]} { set f "prim_xilinx/prim_clock_gating.sv" }
  if {$variant eq "nolod"} {
    foreach base {ibex_alu.sv ibex_decoder.sv ibex_pkg.sv} {
      if {[file tail $f] eq $base} { set f "ibex_base_nolod/$base" }
    }
  }
  lappend files "$root/rtl/$f"
}
lappend files "$root/rtl/ibex_zybo_soc.sv"
add_files -norecurse $files
set_property file_type SystemVerilog [get_files -filter {NAME =~ *.sv}]
add_files -norecurse "$root/rtl/ibex_zybo_wrap.v"
set incdirs {}
foreach d [glob -type d "$root/rtl/ibex/*/rtl" "$root/rtl/ibex/lowrisc_dv_dv_fcov_macros_0"] { lappend incdirs $d }
set_property include_dirs $incdirs [current_fileset]
add_files -fileset constrs_1 -norecurse "$here/zybo.xdc"
update_compile_order -fileset sources_1

# ---- block design
create_bd_design "system"
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7 ps7]
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
  -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} $ps
set_property -dict [list \
  CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ $fclk \
  CONFIG.PCW_USE_M_AXI_GP0 {1} \
  CONFIG.PCW_USE_S_AXI_HP0 {1} \
  CONFIG.PCW_S_AXI_HP0_DATA_WIDTH {32} \
  CONFIG.PCW_UART1_PERIPHERAL_ENABLE {1} \
  CONFIG.PCW_UART1_UART1_IO {MIO 48 .. 49} \
  CONFIG.PCW_UART_PERIPHERAL_FREQMHZ {100} \
] $ps

set ibex [create_bd_cell -type module -reference ibex_zybo_wrap ibex]
set_property -dict [list CONFIG.HAS_LOD [expr {$variant eq "lod" ? 1 : 0}]] $ibex

# clock and reset
set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset rst_fclk0]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins rst_fclk0/slowest_sync_clk] [get_bd_pins ibex/clk_i]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins rst_fclk0/ext_reset_in]
connect_bd_net [get_bd_pins rst_fclk0/peripheral_aresetn] [get_bd_pins ibex/rst_ni]

# ARM -> Ibex control registers (AXI4-Lite through an interconnect)
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
  -config [list Clk_master {/ps7/FCLK_CLK0} Clk_slave {/ps7/FCLK_CLK0} Clk_xbar {/ps7/FCLK_CLK0} \
           Master {/ps7/M_AXI_GP0} Slave {/ibex/s_axil} intc_ip {New AXI Interconnect} master_apm {0}] \
  [get_bd_intf_pins ibex/s_axil]
# Ibex -> DDR (AXI4 into the high-performance port HP0)
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
  -config [list Clk_master {/ps7/FCLK_CLK0} Clk_slave {/ps7/FCLK_CLK0} Clk_xbar {/ps7/FCLK_CLK0} \
           Master {/ibex/m_axi} Slave {/ps7/S_AXI_HP0} intc_ip {New AXI Interconnect} master_apm {0}] \
  [get_bd_intf_pins ps7/S_AXI_HP0]

# done LED
create_bd_port -dir O done_led
connect_bd_net [get_bd_pins ibex/done_o] [get_bd_ports done_led]

# addresses: Ibex registers at 0x43C0_0000 (the ARM program uses this), DDR fully visible to the Ibex master
assign_bd_address
set reg_seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps7/Data] -filter {NAME =~ "*ibex*"}]
set_property offset 0x43C00000 $reg_seg
set_property range 4K $reg_seg
validate_bd_design
save_bd_design

set bd_file [get_files system.bd]
generate_target all $bd_file
set wrapper [make_wrapper -files $bd_file -top]
add_files -norecurse $wrapper
set_property top system_wrapper [current_fileset]
update_compile_order -fileset sources_1

# ---- synthesis and implementation
set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
launch_runs synth_1 -jobs 8
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} { error "synthesis failed" }
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { error "implementation failed" }

# ---- reports
open_run impl_1
report_utilization -hierarchical -hierarchical_depth 6 -file "$repdir/utilization_hier.rpt"
report_utilization -file "$repdir/utilization.rpt"
report_timing_summary -max_paths 20 -file "$repdir/timing_summary.rpt"
# Power: vectorless estimate (default toggle rates) at the default operating conditions. The relative power of
# the two cores (lod vs nolod) is what the energy model uses; see README.md for a simulation-based activity.
report_power -hier all -hierarchical_depth 6 -file "$repdir/power_hier.rpt"
report_power -file "$repdir/power.rpt"

set ibex_cell [get_cells -hier -filter {NAME =~ "*u_soc/u_ibex" && IS_PRIMITIVE == 0}]
set lut_ibex [llength [get_cells -hier -filter "NAME =~ [get_property NAME $ibex_cell]/* && PRIMITIVE_GROUP == LUT"]]
set ff_ibex  [llength [get_cells -hier -filter "NAME =~ [get_property NAME $ibex_cell]/* && PRIMITIVE_GROUP == FLOP_LATCH"]]
if {[catch {get_property SLACK [lindex [get_timing_paths -max_paths 1 -nworst 1 -setup] 0]} wns]} { set wns NA }
if {[catch {get_property TOTAL_POWER [get_runs impl_1]} ptot] || $ptot eq ""} { set ptot NA }
set fh [open "$root/reports/fpga_summary_$variant.csv" w]
puts $fh "variant,fclk_mhz,wns_ns,total_power_w,ibex_luts,ibex_ffs"
puts $fh "$variant,$fclk,$wns,$ptot,$lut_ibex,$ff_ibex"
close $fh
# per-module power (Ibex core, LOD's ALU, BRAM, ...) is parsed from power_hier.rpt by host/collect_reports.py

# ---- hardware platform for Vitis
if {$variant eq "lod"} {
  file copy -force [glob "$proj/zybo_ibex_$variant.runs/impl_1/*.bit"] "$build/zybo_ibex_$variant.bit"
  write_hw_platform -fixed -include_bit -force "$build/zybo_ibex_$variant.xsa"
}
puts "DONE: variant $variant, WNS $wns ns, total power $ptot W; reports in $repdir"
