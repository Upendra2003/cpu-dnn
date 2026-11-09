#!/usr/bin/env bash
# Zip this folder for the FPGA machine (run on the training server after host/make_programs.py).
#   fpga_zybo/make_bundle.sh      -> fpga_zybo_bundle.zip next to the fpga_zybo folder
set -euo pipefail
cd "$(dirname "$0")/.."
rm -f fpga_zybo_bundle.zip
zip -qr fpga_zybo_bundle.zip fpga_zybo \
  -x 'fpga_zybo/build/*' 'fpga_zybo/sim/obj_dir*' 'fpga_zybo/sim/trace.vcd' 'fpga_zybo/sim/build_tb.log' \
     'fpga_zybo/results/*' 'fpga_zybo/reports/*' '*/__pycache__/*' 'fpga_zybo/programs/.tmp*'
ls -la fpga_zybo_bundle.zip
