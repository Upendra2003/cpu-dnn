// Ibex (RV32IC, no multiplier, + LOD custom instruction) for the Zybo (Zynq XC7Z010), driven by the ARM.
//
// The Ibex sees the same address map as the Verilator Simple System, so the very same programs run here:
//   0x0002_0000  console: write a byte to +0x0 (goes into a FIFO the ARM reads), write 1 to +0x8 = "done"
//   0x0003_0000  timer: reads 0, writes ignored (the programs do not use it)
//   0x0010_0000  block RAM, BramBytes (default 192 KB), 1-cycle like the Simple System's RAM;
//                instruction fetch always comes from here
//   0x8000_0000  DDR window, 16 MB, through an AXI4 master into the Zynq's S_AXI_HP0 port
//                (physical address = ddr_base register + offset); used by programs too large for the BRAM
//
// The ARM controls everything through an AXI4-Lite slave (registers below). While CTRL.run = 0 the Ibex is held
// in reset and the ARM can write and read the block RAM through LOAD_ADDR / LOAD_DATA.
//
// Registers (byte offsets on the AXI4-Lite slave):
//   0x00 CTRL       rw  bit0 run: 1 = release the Ibex from reset; writing 0 also clears done, the FIFO and the
//                       counters
//   0x04 STATUS     r   bit0 done, bit1 fifo empty, bit2 fifo full, bit3 run, bit4 data bus error seen
//   0x08 FIFO       r   bit31 = valid, bits7:0 = next console byte (the read pops it)
//   0x0C CYCLES_LO  r   PL clock cycles since run went to 1 (stops at done)
//   0x10 CYCLES_HI  r
//   0x14 LOAD_ADDR  rw  byte offset into the block RAM (word aligned)
//   0x18 LOAD_DATA  rw  write: store a word at LOAD_ADDR, LOAD_ADDR += 4; read: word at LOAD_ADDR, LOAD_ADDR += 4
//                       (only while run = 0)
//   0x1C DDR_BASE   rw  physical DDR address seen at Ibex address 0x8000_0000 (default 0x1000_0000)
//   0x20 ID         r   0x1BE7_0000 | HasLod << 8 | (BramBytes / 1024 / 64) — identifies the bitstream
//   0x24 BRAM_BYTES r   BramBytes
//   0x28 DDR_READS  r   number of Ibex data reads that went to DDR
//   0x2C DDR_WRITES r   number of Ibex data writes that went to DDR
//   0x30 DDR_WAIT_LO r  cycles the Ibex spent waiting on DDR accesses (request to response), 64 bit
//   0x34 DDR_WAIT_HI r

module ibex_zybo_soc #(
  parameter int unsigned BramBytes = 192 * 1024,
  parameter bit          HasLod    = 1'b1,        // only reported in ID; the LOD itself is in the RTL files used
  parameter int unsigned FifoDepth = 2048
) (
  input  logic        clk_i,
  input  logic        rst_ni,

  // AXI4-Lite slave (from the ARM's M_AXI_GP0)
  input  logic [7:0]  s_axil_awaddr,
  input  logic        s_axil_awvalid,
  output logic        s_axil_awready,
  input  logic [31:0] s_axil_wdata,
  input  logic [3:0]  s_axil_wstrb,
  input  logic        s_axil_wvalid,
  output logic        s_axil_wready,
  output logic [1:0]  s_axil_bresp,
  output logic        s_axil_bvalid,
  input  logic        s_axil_bready,
  input  logic [7:0]  s_axil_araddr,
  input  logic        s_axil_arvalid,
  output logic        s_axil_arready,
  output logic [31:0] s_axil_rdata,
  output logic [1:0]  s_axil_rresp,
  output logic        s_axil_rvalid,
  input  logic        s_axil_rready,

  // AXI4 master (to the ARM's S_AXI_HP0 -> DDR), one transfer at a time
  output logic [31:0] m_axi_awaddr,
  output logic [7:0]  m_axi_awlen,
  output logic [2:0]  m_axi_awsize,
  output logic [1:0]  m_axi_awburst,
  output logic [3:0]  m_axi_awcache,
  output logic [2:0]  m_axi_awprot,
  output logic        m_axi_awvalid,
  input  logic        m_axi_awready,
  output logic [31:0] m_axi_wdata,
  output logic [3:0]  m_axi_wstrb,
  output logic        m_axi_wlast,
  output logic        m_axi_wvalid,
  input  logic        m_axi_wready,
  input  logic [1:0]  m_axi_bresp,
  input  logic        m_axi_bvalid,
  output logic        m_axi_bready,
  output logic [31:0] m_axi_araddr,
  output logic [7:0]  m_axi_arlen,
  output logic [2:0]  m_axi_arsize,
  output logic [1:0]  m_axi_arburst,
  output logic [3:0]  m_axi_arcache,
  output logic [2:0]  m_axi_arprot,
  output logic        m_axi_arvalid,
  input  logic        m_axi_arready,
  input  logic [31:0] m_axi_rdata,
  input  logic [1:0]  m_axi_rresp,
  input  logic        m_axi_rvalid,
  output logic        m_axi_rready,

  output logic        done_o              // to an LED
);

  localparam int unsigned BramWords = BramBytes / 4;
  localparam int unsigned BramAw    = $clog2(BramWords);
  localparam logic [31:0] BramBase  = 32'h0010_0000;
  localparam logic [31:0] DdrBase   = 32'h8000_0000;
  localparam logic [31:0] DdrMask   = 32'h00FF_FFFF;     // 16 MB window

  // ------------------------------------------------------------------ control registers
  logic        run_q, done_q, err_q;
  logic [63:0] cycles_q;
  logic [31:0] load_addr_q, ddr_base_q;
  logic [31:0] ddr_reads_q, ddr_writes_q;
  logic [63:0] ddr_wait_q;

  logic core_rst_n;
  assign core_rst_n = rst_ni & run_q;
  assign done_o     = done_q;

  // ------------------------------------------------------------------ console FIFO (Ibex writes, ARM reads)
  localparam int unsigned FifoAw = $clog2(FifoDepth);
  logic [7:0]      fifo_mem [FifoDepth];
  logic [FifoAw:0] fifo_wp, fifo_rp;
  logic            fifo_empty, fifo_full, fifo_push, fifo_pop;
  logic [7:0]      fifo_wdata;
  assign fifo_empty = fifo_wp == fifo_rp;
  assign fifo_full  = (fifo_wp[FifoAw-1:0] == fifo_rp[FifoAw-1:0]) && (fifo_wp[FifoAw] != fifo_rp[FifoAw]);

  always_ff @(posedge clk_i) begin
    if (fifo_push) fifo_mem[fifo_wp[FifoAw-1:0]] <= fifo_wdata;
  end
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      fifo_wp <= '0;
      fifo_rp <= '0;
    end else if (!run_q) begin
      fifo_wp <= '0;
      fifo_rp <= '0;
    end else begin
      if (fifo_push) fifo_wp <= fifo_wp + 1'b1;
      if (fifo_pop)  fifo_rp <= fifo_rp + 1'b1;
    end
  end

  // ------------------------------------------------------------------ the core
  logic        instr_req, instr_rvalid, instr_err;
  logic [31:0] instr_addr, instr_rdata;
  logic        data_req, data_gnt, data_rvalid, data_we, data_err;
  logic [3:0]  data_be;
  logic [31:0] data_addr, data_wdata, data_rdata;

  ibex_top #(
    .RV32E           (1'b0),
    .RV32M           (ibex_pkg::RV32MNone),     // no multiplier: the whole point of the study
    .RV32B           (ibex_pkg::RV32BNone),
    .RV32ZC          (ibex_pkg::RV32Zca),       // same as the Verilator build (RV32IC)
    .RegFile         (ibex_pkg::RegFileFPGA),   // LUT-RAM register file; same timing as the FF one
    .BranchTargetALU (1'b0),
    .WritebackStage  (1'b0),
    .ICache          (1'b0),
    .BranchPredictor (1'b0),
    .DbgTriggerEn    (1'b0),
    .SecureIbex      (1'b0),
    .PMPEnable       (1'b0),
    .MHPMCounterNum  (0),
    .DmBaseAddr      (32'h0010_0000),
    .DmAddrMask      (32'h0000_0003),
    .DmHaltAddr      (32'h0010_0000),
    .DmExceptionAddr (32'h0010_0000)
  ) u_ibex (
    .clk_i                   (clk_i),
    .rst_ni                  (core_rst_n),
    .test_en_i               (1'b0),
    .scan_rst_ni             (1'b1),
    .ram_cfg_icache_tag_i    ('{default: prim_ram_1p_pkg::RAM_1P_CFG_REQ_DEFAULT}),
    .ram_cfg_icache_tag_o    (),
    .ram_cfg_icache_data_i   ('{default: prim_ram_1p_pkg::RAM_1P_CFG_REQ_DEFAULT}),
    .ram_cfg_icache_data_o   (),
    .hart_id_i               (32'b0),
    .boot_addr_i             (BramBase),
    .trvk_heap_base_addr_i   (32'h0),

    .instr_req_o             (instr_req),
    .instr_gnt_i             (instr_req),        // the BRAM accepts every fetch at once
    .instr_rvalid_i          (instr_rvalid),
    .instr_addr_o            (instr_addr),
    .instr_rdata_i           (instr_rdata),
    .instr_rdata_intg_i      (7'b0),
    .instr_err_i             (instr_err),

    .data_req_o              (data_req),
    .data_gnt_i              (data_gnt),
    .data_rvalid_i           (data_rvalid),
    .data_we_o               (data_we),
    .data_be_o               (data_be),
    .data_addr_o             (data_addr),
    .data_wdata_o            (data_wdata),
    .data_wdata_intg_o       (),
    .data_tag_o              (),
    .data_rdata_i            (data_rdata),
    .data_rdata_intg_i       (7'b0),
    .data_tag_i              (1'b0),
    .data_err_i              (data_err),

    .trvk_revbm_req_o        (),
    .trvk_revbm_gnt_i        (1'b0),
    .trvk_revbm_rvalid_i     (1'b0),
    .trvk_revbm_addr_o       (),
    .trvk_revbm_rdata_i      ('0),
    .trvk_revbm_rdata_intg_i ('0),
    .trvk_revbm_err_i        (1'b0),

    .irq_software_i          (1'b0),
    .irq_timer_i             (1'b0),
    .irq_external_i          (1'b0),
    .irq_fast_i              (15'b0),
    .irq_nm_i                (1'b0),

    .scramble_key_valid_i    ('0),
    .scramble_key_i          ('0),
    .scramble_nonce_i        ('0),
    .scramble_req_o          (),

    .debug_req_i             (1'b0),
    .crash_dump_o            (),
    .double_fault_seen_o     (),

    .fetch_enable_i          (ibex_pkg::IbexMuBiOn),
    .mcounteren_writable_i   (ibex_pkg::IbexMuBiOn),
    .alert_minor_o           (),
    .alert_major_internal_o  (),
    .alert_major_bus_o       (),
    .core_sleep_o            (),

    .lockstep_cmp_en_o       (),
    .data_req_shadow_o       (),
    .data_we_shadow_o        (),
    .data_be_shadow_o        (),
    .data_addr_shadow_o      (),
    .data_wdata_shadow_o     (),
    .data_wdata_intg_shadow_o(),
    .instr_req_shadow_o      (),
    .instr_addr_shadow_o     ()
  );

  // ------------------------------------------------------------------ block RAM (true dual port, byte writes)
  // Port A: Ibex data, or the ARM loader while the core is in reset. Port B: Ibex instruction fetch.
  (* ram_style = "block" *) logic [31:0] bram [BramWords];

  logic              a_en, a_we;
  logic [3:0]        a_be;
  logic [BramAw-1:0] a_idx, b_idx;
  logic [31:0]       a_wdata, a_rdata, b_rdata;

  always_ff @(posedge clk_i) begin
    if (a_en) begin
      for (int i = 0; i < 4; i++)
        if (a_we && a_be[i]) bram[a_idx][8*i +: 8] <= a_wdata[8*i +: 8];
      a_rdata <= bram[a_idx];
    end
  end
  always_ff @(posedge clk_i) begin
    if (instr_req) b_rdata <= bram[b_idx];
  end
  assign b_idx       = BramAw'((instr_addr - BramBase) >> 2);
  assign instr_rdata = b_rdata;
  always_ff @(posedge clk_i or negedge core_rst_n) begin
    if (!core_rst_n) begin
      instr_rvalid <= 1'b0;
      instr_err    <= 1'b0;
    end else begin
      instr_rvalid <= instr_req;
      instr_err    <= instr_req && ((instr_addr - BramBase) >= BramBytes);
    end
  end

  // ------------------------------------------------------------------ Ibex data bus decode
  typedef enum logic [2:0] {DevNone, DevBram, DevCons, DevTimer, DevDdr, DevErr} dev_e;
  dev_e sel, rsp_dev_q;
  logic rsp_pending_q;

  always_comb begin
    if ((data_addr - BramBase) < BramBytes)                  sel = DevBram;
    else if ((data_addr & ~32'h3FF) == 32'h0002_0000)        sel = DevCons;
    else if ((data_addr & ~32'h3FF) == 32'h0003_0000)        sel = DevTimer;
    else if ((data_addr & ~DdrMask) == DdrBase)              sel = DevDdr;
    else                                                     sel = DevErr;
  end

  // DDR master state
  typedef enum logic [2:0] {DIdle, DAddrW, DWaitB, DAddrR, DWaitR} ddr_state_e;
  ddr_state_e dst_q;
  logic [31:0] ddr_rdata_q;
  logic        ddr_done, ddr_err_q;
  logic        aw_done_q, w_done_q;

  logic cons_stall;
  assign cons_stall = (sel == DevCons) && data_we && (data_addr[3:0] == 4'h0) && fifo_full;

  // grant: BRAM/console/timer/error at once (console waits while its FIFO is full), DDR when idle
  always_comb begin
    data_gnt = 1'b0;
    if (data_req && (!rsp_pending_q || data_rvalid)) begin
      unique case (sel)
        DevDdr:  data_gnt = (dst_q == DIdle);
        DevCons: data_gnt = !cons_stall;
        default: data_gnt = 1'b1;
      endcase
    end
  end

  // only one data transaction is ever outstanding (Ibex's LSU waits for rvalid); remember who answers
  always_ff @(posedge clk_i or negedge core_rst_n) begin
    if (!core_rst_n) begin
      rsp_pending_q <= 1'b0;
      rsp_dev_q     <= DevNone;
    end else if (data_gnt) begin
      rsp_pending_q <= 1'b1;
      rsp_dev_q     <= sel;
    end else if (data_rvalid) begin
      rsp_pending_q <= 1'b0;
    end
  end

  // BRAM port A arbitration
  logic        load_wr, load_rd;          // from the AXI-Lite side, only while run = 0
  always_comb begin
    if (run_q) begin
      a_en    = data_gnt && (sel == DevBram);
      a_we    = data_we;
      a_be    = data_be;
      a_idx   = BramAw'((data_addr - BramBase) >> 2);
      a_wdata = data_wdata;
    end else begin
      a_en    = load_wr || load_rd;
      a_we    = load_wr;
      a_be    = 4'hF;
      a_idx   = BramAw'(load_addr_q >> 2);
      a_wdata = s_axil_wdata;
    end
  end

  // console and timer
  assign fifo_push  = data_gnt && (sel == DevCons) && data_we && (data_addr[3:0] == 4'h0);
  assign fifo_wdata = data_wdata[7:0];

  // responses
  logic ddr_rvalid;
  always_comb begin
    data_rvalid = 1'b0;
    data_rdata  = 32'h0;
    data_err    = 1'b0;
    if (rsp_pending_q) begin
      unique case (rsp_dev_q)
        DevBram:  begin data_rvalid = 1'b1; data_rdata = a_rdata; end
        DevCons:  begin data_rvalid = 1'b1; end
        DevTimer: begin data_rvalid = 1'b1; end
        DevDdr:   begin data_rvalid = ddr_done; data_rdata = ddr_rdata_q; data_err = ddr_err_q; end
        DevErr:   begin data_rvalid = 1'b1; data_err = 1'b1; end
        default:  ;
      endcase
    end
  end

  // ------------------------------------------------------------------ AXI4 master to DDR
  logic [31:0] ddr_addr_q, ddr_wdata_q;
  logic [3:0]  ddr_be_q;
  assign ddr_done = (dst_q == DIdle) && rsp_pending_q && (rsp_dev_q == DevDdr) && ddr_rvalid;

  always_ff @(posedge clk_i or negedge core_rst_n) begin
    if (!core_rst_n) begin
      dst_q       <= DIdle;
      ddr_addr_q  <= '0;
      ddr_wdata_q <= '0;
      ddr_be_q    <= '0;
      ddr_rdata_q <= '0;
      ddr_err_q   <= 1'b0;
      ddr_rvalid  <= 1'b0;
      aw_done_q   <= 1'b0;
      w_done_q    <= 1'b0;
    end else begin
      if (ddr_done) ddr_rvalid <= 1'b0;
      unique case (dst_q)
        DIdle: if (data_gnt && sel == DevDdr) begin
          ddr_addr_q  <= ddr_base_q + (data_addr & DdrMask & ~32'h3);
          ddr_wdata_q <= data_wdata;
          ddr_be_q    <= data_be;
          ddr_err_q   <= 1'b0;
          aw_done_q   <= 1'b0;
          w_done_q    <= 1'b0;
          dst_q       <= data_we ? DAddrW : DAddrR;
        end
        DAddrW: begin
          if (m_axi_awready) aw_done_q <= 1'b1;
          if (m_axi_wready)  w_done_q  <= 1'b1;
          if ((aw_done_q || m_axi_awready) && (w_done_q || m_axi_wready)) dst_q <= DWaitB;
        end
        DWaitB: if (m_axi_bvalid) begin
          ddr_err_q  <= m_axi_bresp[1];
          ddr_rvalid <= 1'b1;
          dst_q      <= DIdle;
        end
        DAddrR: if (m_axi_arready) dst_q <= DWaitR;
        DWaitR: if (m_axi_rvalid) begin
          ddr_rdata_q <= m_axi_rdata;
          ddr_err_q   <= m_axi_rresp[1];
          ddr_rvalid  <= 1'b1;
          dst_q       <= DIdle;
        end
        default: dst_q <= DIdle;
      endcase
    end
  end

  assign m_axi_awaddr  = ddr_addr_q;
  assign m_axi_awlen   = 8'd0;
  assign m_axi_awsize  = 3'd2;
  assign m_axi_awburst = 2'b01;
  assign m_axi_awcache = 4'b0011;
  assign m_axi_awprot  = 3'b000;
  assign m_axi_awvalid = (dst_q == DAddrW) && !aw_done_q;
  assign m_axi_wdata   = ddr_wdata_q;
  assign m_axi_wstrb   = ddr_be_q;
  assign m_axi_wlast   = 1'b1;
  assign m_axi_wvalid  = (dst_q == DAddrW) && !w_done_q;
  assign m_axi_bready  = (dst_q == DWaitB);
  assign m_axi_araddr  = ddr_addr_q;
  assign m_axi_arlen   = 8'd0;
  assign m_axi_arsize  = 3'd2;
  assign m_axi_arburst = 2'b01;
  assign m_axi_arcache = 4'b0011;
  assign m_axi_arprot  = 3'b000;
  assign m_axi_arvalid = (dst_q == DAddrR);
  assign m_axi_rready  = (dst_q == DWaitR);

  // ------------------------------------------------------------------ AXI4-Lite slave (ARM side)
  // Writes: accept AW and W together, answer with B. Reads: one cycle to fetch (BRAM is synchronous), then R.
  logic       aw_hs, ar_hs;
  logic [7:0] rd_addr_q;
  logic       rd_wait_q;

  assign s_axil_awready = s_axil_awvalid && s_axil_wvalid && !s_axil_bvalid;
  assign s_axil_wready  = s_axil_awready;
  assign aw_hs          = s_axil_awready;
  assign s_axil_bresp   = 2'b00;
  assign s_axil_arready = s_axil_arvalid && !s_axil_rvalid && !rd_wait_q;
  assign ar_hs          = s_axil_arready;
  assign s_axil_rresp   = 2'b00;

  assign load_wr  = aw_hs && (s_axil_awaddr[7:2] == 6'h06) && !run_q;
  assign load_rd  = ar_hs && (s_axil_araddr[7:2] == 6'h06) && !run_q;
  assign fifo_pop = ar_hs && (s_axil_araddr[7:2] == 6'h02) && !fifo_empty;

  logic [7:0] fifo_head_q;
  logic       fifo_head_valid_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      run_q             <= 1'b0;
      done_q            <= 1'b0;
      err_q             <= 1'b0;
      cycles_q          <= '0;
      load_addr_q       <= '0;
      ddr_base_q        <= 32'h1000_0000;
      ddr_reads_q       <= '0;
      ddr_writes_q      <= '0;
      ddr_wait_q        <= '0;
      s_axil_bvalid     <= 1'b0;
      s_axil_rvalid     <= 1'b0;
      s_axil_rdata      <= '0;
      rd_addr_q         <= '0;
      rd_wait_q         <= 1'b0;
      fifo_head_q       <= '0;
      fifo_head_valid_q <= 1'b0;
    end else begin
      // ---- core-side events
      if (run_q && !done_q) cycles_q <= cycles_q + 1'b1;
      if (data_gnt && sel == DevCons && data_we && data_addr[3:0] == 4'h8 && data_wdata[0]) done_q <= 1'b1;
      if (data_rvalid && data_err) err_q <= 1'b1;
      if (data_gnt && sel == DevDdr) begin
        if (data_we) ddr_writes_q <= ddr_writes_q + 1'b1;
        else         ddr_reads_q  <= ddr_reads_q + 1'b1;
      end
      if (rsp_pending_q && rsp_dev_q == DevDdr && !data_rvalid) ddr_wait_q <= ddr_wait_q + 1'b1;

      // ---- AXI-Lite writes
      if (s_axil_bvalid && s_axil_bready) s_axil_bvalid <= 1'b0;
      if (aw_hs) begin
        s_axil_bvalid <= 1'b1;
        unique case (s_axil_awaddr[7:2])
          6'h00: begin
            run_q <= s_axil_wdata[0];
            if (!s_axil_wdata[0]) begin
              done_q       <= 1'b0;
              err_q        <= 1'b0;
              cycles_q     <= '0;
              ddr_reads_q  <= '0;
              ddr_writes_q <= '0;
              ddr_wait_q   <= '0;
            end
          end
          6'h05: load_addr_q <= s_axil_wdata;
          6'h06: if (!run_q) load_addr_q <= load_addr_q + 32'd4;
          6'h07: ddr_base_q  <= s_axil_wdata;
          default: ;
        endcase
      end

      // ---- AXI-Lite reads
      if (s_axil_rvalid && s_axil_rready) s_axil_rvalid <= 1'b0;
      if (ar_hs) begin
        rd_addr_q <= s_axil_araddr;
        rd_wait_q <= 1'b1;
        fifo_head_valid_q <= !fifo_empty;
        fifo_head_q       <= fifo_mem[fifo_rp[FifoAw-1:0]];
        if (s_axil_araddr[7:2] == 6'h06 && !run_q) load_addr_q <= load_addr_q + 32'd4;
      end
      if (rd_wait_q) begin
        rd_wait_q     <= 1'b0;
        s_axil_rvalid <= 1'b1;
        unique case (rd_addr_q[7:2])
          6'h00: s_axil_rdata <= {31'b0, run_q};
          6'h01: s_axil_rdata <= {27'b0, err_q, run_q, fifo_full, fifo_empty, done_q};
          6'h02: s_axil_rdata <= {fifo_head_valid_q, 23'b0, fifo_head_q};
          6'h03: s_axil_rdata <= cycles_q[31:0];
          6'h04: s_axil_rdata <= cycles_q[63:32];
          6'h05: s_axil_rdata <= load_addr_q;
          6'h06: s_axil_rdata <= a_rdata;
          6'h07: s_axil_rdata <= ddr_base_q;
          6'h08: s_axil_rdata <= 32'h1BE7_0000 | (32'(HasLod) << 8) | 32'(BramBytes / 1024 / 64);
          6'h09: s_axil_rdata <= BramBytes;
          6'h0A: s_axil_rdata <= ddr_reads_q;
          6'h0B: s_axil_rdata <= ddr_writes_q;
          6'h0C: s_axil_rdata <= ddr_wait_q[31:0];
          6'h0D: s_axil_rdata <= ddr_wait_q[63:32];
          default: s_axil_rdata <= 32'hDEAD_BEEF;
        endcase
      end
    end
  end

endmodule
