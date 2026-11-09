// Verilog wrapper of ibex_zybo_soc for the Vivado block design (a module reference cannot be SystemVerilog).
// The X_INTERFACE attributes let Vivado see s_axil (AXI4-Lite slave) and m_axi (AXI4 master) as bus interfaces.

module ibex_zybo_wrap #(
  parameter BRAM_BYTES = 196608,
  parameter HAS_LOD    = 1
) (
  (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk_i CLK" *)
  (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axil:m_axi, ASSOCIATED_RESET rst_ni" *)
  input  wire        clk_i,
  (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rst_ni RST" *)
  (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
  input  wire        rst_ni,

  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil AWADDR" *)
  (* X_INTERFACE_PARAMETER = "PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 12" *)
  input  wire [11:0] s_axil_awaddr,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil AWVALID" *) input  wire        s_axil_awvalid,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil AWREADY" *) output wire        s_axil_awready,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil WDATA" *)   input  wire [31:0] s_axil_wdata,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil WSTRB" *)   input  wire [3:0]  s_axil_wstrb,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil WVALID" *)  input  wire        s_axil_wvalid,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil WREADY" *)  output wire        s_axil_wready,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil BRESP" *)   output wire [1:0]  s_axil_bresp,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil BVALID" *)  output wire        s_axil_bvalid,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil BREADY" *)  input  wire        s_axil_bready,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil ARADDR" *)  input  wire [11:0] s_axil_araddr,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil ARVALID" *) input  wire        s_axil_arvalid,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil ARREADY" *) output wire        s_axil_arready,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil RDATA" *)   output wire [31:0] s_axil_rdata,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil RRESP" *)   output wire [1:0]  s_axil_rresp,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil RVALID" *)  output wire        s_axil_rvalid,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axil RREADY" *)  input  wire        s_axil_rready,

  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi AWADDR" *)
  (* X_INTERFACE_PARAMETER = "PROTOCOL AXI4, DATA_WIDTH 32, ADDR_WIDTH 32, MAX_BURST_LENGTH 1, NUM_READ_OUTSTANDING 1, NUM_WRITE_OUTSTANDING 1, SUPPORTS_NARROW_BURST 0, HAS_LOCK 0, HAS_QOS 0, HAS_REGION 0, ID_WIDTH 0" *)
  output wire [31:0] m_axi_awaddr,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi AWLEN" *)   output wire [7:0]  m_axi_awlen,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi AWSIZE" *)  output wire [2:0]  m_axi_awsize,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi AWBURST" *) output wire [1:0]  m_axi_awburst,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi AWCACHE" *) output wire [3:0]  m_axi_awcache,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi AWPROT" *)  output wire [2:0]  m_axi_awprot,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi AWVALID" *) output wire        m_axi_awvalid,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi AWREADY" *) input  wire        m_axi_awready,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi WDATA" *)   output wire [31:0] m_axi_wdata,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi WSTRB" *)   output wire [3:0]  m_axi_wstrb,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi WLAST" *)   output wire        m_axi_wlast,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi WVALID" *)  output wire        m_axi_wvalid,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi WREADY" *)  input  wire        m_axi_wready,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi BRESP" *)   input  wire [1:0]  m_axi_bresp,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi BVALID" *)  input  wire        m_axi_bvalid,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi BREADY" *)  output wire        m_axi_bready,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi ARADDR" *)  output wire [31:0] m_axi_araddr,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi ARLEN" *)   output wire [7:0]  m_axi_arlen,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi ARSIZE" *)  output wire [2:0]  m_axi_arsize,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi ARBURST" *) output wire [1:0]  m_axi_arburst,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi ARCACHE" *) output wire [3:0]  m_axi_arcache,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi ARPROT" *)  output wire [2:0]  m_axi_arprot,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi ARVALID" *) output wire        m_axi_arvalid,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi ARREADY" *) input  wire        m_axi_arready,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi RDATA" *)   input  wire [31:0] m_axi_rdata,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi RRESP" *)   input  wire [1:0]  m_axi_rresp,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi RVALID" *)  input  wire        m_axi_rvalid,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi RREADY" *)  output wire        m_axi_rready,

  output wire        done_o
);

  ibex_zybo_soc #(
    .BramBytes (BRAM_BYTES),
    .HasLod    (HAS_LOD)
  ) u_soc (
    .clk_i          (clk_i),
    .rst_ni         (rst_ni),
    .s_axil_awaddr  (s_axil_awaddr[7:0]),
    .s_axil_awvalid (s_axil_awvalid),
    .s_axil_awready (s_axil_awready),
    .s_axil_wdata   (s_axil_wdata),
    .s_axil_wstrb   (s_axil_wstrb),
    .s_axil_wvalid  (s_axil_wvalid),
    .s_axil_wready  (s_axil_wready),
    .s_axil_bresp   (s_axil_bresp),
    .s_axil_bvalid  (s_axil_bvalid),
    .s_axil_bready  (s_axil_bready),
    .s_axil_araddr  (s_axil_araddr[7:0]),
    .s_axil_arvalid (s_axil_arvalid),
    .s_axil_arready (s_axil_arready),
    .s_axil_rdata   (s_axil_rdata),
    .s_axil_rresp   (s_axil_rresp),
    .s_axil_rvalid  (s_axil_rvalid),
    .s_axil_rready  (s_axil_rready),
    .m_axi_awaddr   (m_axi_awaddr),
    .m_axi_awlen    (m_axi_awlen),
    .m_axi_awsize   (m_axi_awsize),
    .m_axi_awburst  (m_axi_awburst),
    .m_axi_awcache  (m_axi_awcache),
    .m_axi_awprot   (m_axi_awprot),
    .m_axi_awvalid  (m_axi_awvalid),
    .m_axi_awready  (m_axi_awready),
    .m_axi_wdata    (m_axi_wdata),
    .m_axi_wstrb    (m_axi_wstrb),
    .m_axi_wlast    (m_axi_wlast),
    .m_axi_wvalid   (m_axi_wvalid),
    .m_axi_wready   (m_axi_wready),
    .m_axi_bresp    (m_axi_bresp),
    .m_axi_bvalid   (m_axi_bvalid),
    .m_axi_bready   (m_axi_bready),
    .m_axi_araddr   (m_axi_araddr),
    .m_axi_arlen    (m_axi_arlen),
    .m_axi_arsize   (m_axi_arsize),
    .m_axi_arburst  (m_axi_arburst),
    .m_axi_arcache  (m_axi_arcache),
    .m_axi_arprot   (m_axi_arprot),
    .m_axi_arvalid  (m_axi_arvalid),
    .m_axi_arready  (m_axi_arready),
    .m_axi_rdata    (m_axi_rdata),
    .m_axi_rresp    (m_axi_rresp),
    .m_axi_rvalid   (m_axi_rvalid),
    .m_axi_rready   (m_axi_rready),
    .done_o         (done_o)
  );

endmodule
