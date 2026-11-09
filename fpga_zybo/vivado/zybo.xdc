# Zybo (original, XC7Z010-1CLG400C): the Ibex "done" flag on LED LD0.
# Everything else (clock, reset, DDR, UART) comes from the Zynq processing system.
set_property -dict {PACKAGE_PIN M14 IOSTANDARD LVCMOS33} [get_ports done_led]
