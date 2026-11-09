"""Vitis 2024.x (Unified IDE, Python API): platform + bare-metal ARM application from build/zybo_ibex_lod.xsa.

    cd fpga_zybo && vitis -s vitis/create_app_vitis.py

Result: build/vitis_ws/ibex_host/build/ibex_host.elf. The Vitis Python API changes between releases; if a call
below fails, the same steps in the IDE are: new platform component from the XSA (standalone, ps7_cortexa9_0,
stdin/stdout = ps7_uart_1), new empty application component on it, add vitis/src/main.c, build.
"""
import os
import pathlib

import vitis

root = pathlib.Path(__file__).resolve().parents[1]
ws = root / "build" / "vitis_ws"
xsa = root / "build" / "zybo_ibex_lod.xsa"
assert xsa.exists(), f"missing {xsa}: run vivado/build.tcl first"
ws.mkdir(parents=True, exist_ok=True)

client = vitis.create_client()
client.set_workspace(path=str(ws))
platform = client.create_platform_component(name="zybo_ibex_pf", hw_design=str(xsa), os="standalone",
                                            cpu="ps7_cortexa9_0")
domain = platform.get_domain(name="standalone_ps7_cortexa9_0")
for key in ("standalone_stdin", "standalone_stdout"):
    try:
        domain.set_config(option="os", param=key, value="ps7_uart_1")
    except Exception as e:                       # parameter names differ between releases
        print("note: could not set", key, e)
platform.build()
xpfm = next((ws / "zybo_ibex_pf").rglob("*.xpfm"))
app = client.create_app_component(name="ibex_host", platform=str(xpfm), domain="standalone_ps7_cortexa9_0",
                                  template="empty_application")
app.import_files(from_loc=str(root / "vitis" / "src"), files=["main.c"], dest_dir_in_cmp="src")
app.build()
print("DONE:", [str(p) for p in (ws / "ibex_host").rglob("*.elf")])
vitis.dispose()
