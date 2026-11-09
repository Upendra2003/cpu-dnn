# Vitis 2023.x (classic): platform + bare-metal ARM application from build/zybo_ibex_lod.xsa.
#   cd fpga_zybo && xsct vitis/create_app_xsct.tcl
# Result: build/vitis_ws/ibex_host/Release/ibex_host.elf  (and the platform with ps7_init.tcl)
set root [file normalize [file dirname [info script]]/..]
set ws   "$root/build/vitis_ws"
set xsa  "$root/build/zybo_ibex_lod.xsa"
if {![file exists $xsa]} { error "missing $xsa: run vivado/build.tcl first" }
file mkdir $ws
setws $ws
platform create -name zybo_ibex_pf -hw $xsa -proc ps7_cortexa9_0 -os standalone -out $ws
# stdin/stdout on the USB-UART (PS UART1)
bsp config stdin ps7_uart_1
bsp config stdout ps7_uart_1
platform write
platform generate
app create -name ibex_host -platform zybo_ibex_pf -domain standalone_domain -template {Empty Application(C)}
importsources -name ibex_host -path "$root/vitis/src"
app config -name ibex_host build-config release
app config -name ibex_host -add compiler-misc {-O2}
app build -name ibex_host
puts "DONE: [glob $ws/ibex_host/Release/*.elf]"
