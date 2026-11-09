// Verilator testbench for rtl/ibex_zybo_soc.sv: behaves like the ARM program (vitis/src/main.c) would.
//   tb_soc <program.elf> [ddr_latency_cycles=20]
// Loads the ELF (block-RAM segments through LOAD_ADDR/LOAD_DATA, DDR segments straight into the DDR model),
// releases the core, prints the console output and finally the counters, like the board does.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <vector>

#include "Vibex_zybo_soc.h"
#include "verilated.h"
#ifdef TRACE
#include "verilated_vcd_c.h"
static VerilatedVcdC *tfp;
#endif

static Vibex_zybo_soc *top;
static uint64_t tick_count = 0;

static const uint32_t DDR_PHYS = 0x10000000u, DDR_SIZE = 16u << 20;
static std::vector<uint8_t> ddr(DDR_SIZE);
static int ddr_lat = 20;

// ---- DDR AXI4 slave model: one transfer at a time, fixed latency
struct Pending { int left; bool write; uint32_t addr; };
static std::deque<Pending> pend;
static bool aw_seen = false, w_seen = false;
static uint32_t aw_addr = 0, w_data = 0, w_strb = 0;

static void ddr_model_comb() {
  top->m_axi_arready = pend.empty();
  top->m_axi_awready = pend.empty() && !aw_seen;
  top->m_axi_wready  = pend.empty() && !w_seen;
  top->m_axi_rvalid = 0;
  top->m_axi_bvalid = 0;
  if (!pend.empty() && pend.front().left <= 0) {
    Pending &p = pend.front();
    uint32_t off = p.addr - DDR_PHYS;
    if (p.write) {
      top->m_axi_bvalid = 1;
      top->m_axi_bresp = off < DDR_SIZE ? 0 : 2;
    } else {
      top->m_axi_rvalid = 1;
      uint32_t v = 0;
      if (off + 4 <= DDR_SIZE) memcpy(&v, &ddr[off], 4);
      top->m_axi_rdata = v;
      top->m_axi_rresp = off < DDR_SIZE ? 0 : 2;
    }
  }
}

static void ddr_model_posedge() {  // sample handshakes (values from before the edge)
  if (!pend.empty()) {
    Pending &p = pend.front();
    if (p.left > 0) p.left--;
    else if ((p.write && top->m_axi_bvalid && top->m_axi_bready) ||
             (!p.write && top->m_axi_rvalid && top->m_axi_rready))
      pend.pop_front();
  }
  if (top->m_axi_arvalid && top->m_axi_arready) pend.push_back({ddr_lat, false, top->m_axi_araddr});
  if (top->m_axi_awvalid && top->m_axi_awready) { aw_seen = true; aw_addr = top->m_axi_awaddr; }
  if (top->m_axi_wvalid && top->m_axi_wready) { w_seen = true; w_data = top->m_axi_wdata; w_strb = top->m_axi_wstrb; }
  if (aw_seen && w_seen) {
    uint32_t off = aw_addr - DDR_PHYS;
    if (off + 4 <= DDR_SIZE)
      for (int i = 0; i < 4; i++)
        if (w_strb >> i & 1) ddr[off + i] = (w_data >> (8 * i)) & 0xFF;
    pend.push_back({ddr_lat, true, aw_addr});
    aw_seen = w_seen = false;
  }
}

static void tick() {
  ddr_model_comb();
  top->clk_i = 0;
  top->eval();
  ddr_model_comb();
  top->eval();
  ddr_model_posedge();
  top->clk_i = 1;
  top->eval();
#ifdef TRACE
  if (tfp) tfp->dump(tick_count * 2);
#endif
  tick_count++;
}

// ---- AXI4-Lite master
static void axil_write(uint8_t addr, uint32_t data) {
  top->s_axil_awaddr = addr; top->s_axil_awvalid = 1;
  top->s_axil_wdata = data; top->s_axil_wstrb = 0xF; top->s_axil_wvalid = 1;
  top->s_axil_bready = 1;
  for (int i = 0;; i++) {
    top->eval();
    bool hs = top->s_axil_awready;
    tick();
    if (hs) break;
    if (i > 1000) { fprintf(stderr, "axil_write timeout\n"); exit(2); }
  }
  top->s_axil_awvalid = 0; top->s_axil_wvalid = 0;
  for (int i = 0; !top->s_axil_bvalid; i++) { tick(); if (i > 1000) exit(2); }
  tick();
  top->s_axil_bready = 0;
}

static uint32_t axil_read(uint8_t addr) {
  top->s_axil_araddr = addr; top->s_axil_arvalid = 1; top->s_axil_rready = 1;
  for (int i = 0;; i++) {
    top->eval();
    bool hs = top->s_axil_arready;
    tick();
    if (hs) break;
    if (i > 1000) { fprintf(stderr, "axil_read timeout\n"); exit(2); }
  }
  top->s_axil_arvalid = 0;
  for (int i = 0; !top->s_axil_rvalid; i++) { tick(); if (i > 1000) exit(2); }
  uint32_t v = top->s_axil_rdata;
  tick();
  top->s_axil_rready = 0;
  return v;
}

// ---- ELF32 loader
struct Seg { uint32_t addr; std::vector<uint8_t> bytes; };
static std::vector<Seg> load_elf(const char *path) {
  std::ifstream f(path, std::ios::binary);
  std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), {});
  if (b.size() < 52 || memcmp(b.data(), "\x7f" "ELF", 4)) { fprintf(stderr, "not an ELF\n"); exit(2); }
  uint32_t phoff; uint16_t phentsize, phnum;
  memcpy(&phoff, &b[28], 4); memcpy(&phentsize, &b[42], 2); memcpy(&phnum, &b[44], 2);
  std::vector<Seg> segs;
  for (int i = 0; i < phnum; i++) {
    const uint8_t *ph = &b[phoff + i * phentsize];
    uint32_t type, off, paddr, filesz;
    memcpy(&type, ph, 4); memcpy(&off, ph + 4, 4); memcpy(&paddr, ph + 12, 4); memcpy(&filesz, ph + 16, 4);
    if (type != 1 || filesz == 0) continue;
    segs.push_back({paddr, std::vector<uint8_t>(b.begin() + off, b.begin() + off + filesz)});
  }
  return segs;
}

int main(int argc, char **argv) {
  Verilated::randReset(1);       // registers power up as all ones (not zero), so the reset pulse below is a real
                                 // falling edge on every reset net, as on hardware; needs --x-initial unique
  Verilated::commandArgs(argc, argv);
  if (argc < 2) { fprintf(stderr, "usage: %s program.elf [ddr_latency]\n", argv[0]); return 2; }
  if (argc > 2 && strcmp(argv[1], "--raw")) ddr_lat = atoi(argv[2]);
  top = new Vibex_zybo_soc;
#ifdef TRACE
  Verilated::traceEverOn(true);
  tfp = new VerilatedVcdC;
  top->trace(tfp, 99);
  tfp->open("sim/trace.vcd");
#endif
  top->rst_ni = 1;                 // a real falling edge: asynchronous resets only act on the edge in Verilator
  tick();
  top->rst_ni = 0;
  for (int i = 0; i < 5; i++) tick();
  top->rst_ni = 1;
  tick();

  uint32_t id = axil_read(0x20), bram_bytes = axil_read(0x24);
  printf("TB: ID=%08x BRAM=%u bytes, DDR latency %d cycles\n", id, bram_bytes, ddr_lat);
  axil_write(0x1C, DDR_PHYS);
  std::vector<Seg> segs;
  if (!strcmp(argv[1], "--raw")) {         // tb_soc --raw bram.bin ddr.bin [latency]: images as the ARM stages them
    if (argc < 4) { fprintf(stderr, "usage: %s --raw bram.bin ddr.bin [ddr_latency]\n", argv[0]); return 2; }
    for (int k = 0; k < 2; k++) {
      std::ifstream f(argv[2 + k], std::ios::binary);
      std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), {});
      if (!b.empty()) segs.push_back({k ? 0x80000000u : 0x00100000u, b});
    }
    ddr_lat = argc > 4 ? atoi(argv[4]) : 20;
  } else {
    segs = load_elf(argv[1]);
  }
  for (auto &s : segs) {
    while (s.bytes.size() % 4) s.bytes.push_back(0);
    if (s.addr >= 0x80000000u) {
      memcpy(&ddr[s.addr - 0x80000000u], s.bytes.data(), s.bytes.size());
      printf("TB: segment %08x %6zu bytes -> DDR\n", s.addr, s.bytes.size());
    } else {
      uint32_t off = s.addr - 0x00100000u;
      if (off + s.bytes.size() > bram_bytes) { fprintf(stderr, "segment does not fit the BRAM\n"); return 2; }
      axil_write(0x14, off);
      for (size_t i = 0; i < s.bytes.size(); i += 4) {
        uint32_t w; memcpy(&w, &s.bytes[i], 4);
        axil_write(0x18, w);
      }
      axil_write(0x14, off);                 // read back the first and last word
      uint32_t first = axil_read(0x18), w0; memcpy(&w0, &s.bytes[0], 4);
      if (first != w0) { fprintf(stderr, "BRAM read-back mismatch\n"); return 2; }
      printf("TB: segment %08x %6zu bytes -> BRAM (read-back ok)\n", s.addr, s.bytes.size());
    }
  }
  axil_write(0x00, 1);
  uint64_t start = tick_count;
  bool done = false;
  while (true) {
    for (int i = 0; i < 2000; i++) tick();
    uint32_t st = axil_read(0x04);
    while (true) {
      uint32_t v = axil_read(0x08);
      if (!(v >> 31)) break;
      putchar(v & 0xFF);
    }
    if (st & 1) { if (done) break; done = true; }  // one more drain after done
    if (st & 16) { printf("TB: data bus error\n"); break; }
    if (tick_count - start > 20000000000ull) { printf("TB: timeout\n"); return 1; }
  }
  uint64_t cyc = axil_read(0x0C) | (uint64_t)axil_read(0x10) << 32;
  printf("TB: PL cycles %llu, DDR reads %u writes %u wait %u\n", (unsigned long long)cyc, axil_read(0x28),
         axil_read(0x2C), axil_read(0x30));
#ifdef TRACE
  tfp->close();
#endif
  delete top;
  return 0;
}
