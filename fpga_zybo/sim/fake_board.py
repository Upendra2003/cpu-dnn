"""A stand-in for the Zybo: speaks the UART protocol of vitis/src/main.c on a pseudo-terminal and runs the Ibex
on the Verilator testbench. Lets host/run_board.py (the real board path, pyserial and all) be tested without a
board.

    python3 sim/fake_board.py &                     # prints the pty path, e.g. /dev/pts/7
    python3 host/run_board.py --port /dev/pts/7 --only toyadmos int8 dn1 --mem ddr --limit 2
"""
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import tty
import zlib

FZ = pathlib.Path(__file__).resolve().parents[1]
BRAM_ADDR, DDR_ADDR, BRAM_BYTES = 0x0010_0000, 0x8000_0000, 192 * 1024


def main():
    master, slave = os.openpty()
    tty.setraw(slave)
    print(os.ttyname(slave), flush=True)
    stage = {BRAM_ADDR: bytearray(BRAM_BYTES), DDR_ADDR: bytearray(16 << 20)}
    length = {BRAM_ADDR: 0, DDR_ADDR: 0}

    def get(n):
        b = b""
        while len(b) < n:
            b += os.read(master, n - len(b))
        return b

    def say(s):
        os.write(master, (s + "\r\n").encode())

    while True:
        c = get(1)
        if c == b"P":
            say(f"#PONG id=1be70103 bram={BRAM_BYTES} fclk_hz=50000000 baud=460800 cpu_hz=650000000")
        elif c == b"Z":
            length = {BRAM_ADDR: 0, DDR_ADDR: 0}
            say("#OK")
        elif c == b"W":
            addr, n = struct.unpack("<II", get(8))
            data = get(n)
            crc, = struct.unpack("<I", get(4))
            base = BRAM_ADDR if addr < DDR_ADDR else DDR_ADDR
            off = addr - base
            if off + n > len(stage[base]):
                say(f"#ERR range addr={addr:08x} len={n}")
            elif zlib.crc32(data) != crc:
                say("#ERR crc")
            else:
                stage[base][off:off + n] = data
                length[base] = max(length[base], off + n)
                say("#OK")
        elif c == b"R":
            timeout, = struct.unpack("<I", get(4))
            with tempfile.TemporaryDirectory() as d:
                files = []
                for base in (BRAM_ADDR, DDR_ADDR):
                    p = pathlib.Path(d) / f"{base:08x}.bin"
                    p.write_bytes(bytes(stage[base][:(length[base] + 3) // 4 * 4]))
                    files.append(str(p))
                out = subprocess.run([str(FZ / "sim/obj_dir/Vibex_zybo_soc"), "--raw", *files, "20"],
                                     capture_output=True, text=True).stdout.splitlines()
            for line in out:
                if not line.startswith("TB:"):
                    say(line)
            f = next(l for l in out if l.startswith("TB: PL cycles")).split()
            say(f"#DONE pl_cycles={f[3].rstrip(',')} arm_us=0 ddr_reads={f[6]} ddr_writes={f[8]} ddr_wait={f[10]}"
                f" err=0 timeout=0")


if __name__ == "__main__":
    sys.exit(main())
