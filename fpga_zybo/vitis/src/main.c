/*
 * ARM (Cortex-A9, bare metal) side of the Zybo Ibex experiments.
 *
 * The PC (host/run_board.py) sends RISC-V programs over the USB-UART; this program keeps them in a staging copy,
 * loads them into the Ibex's memory (block RAM through the control registers, DDR window by memcpy), runs the
 * Ibex and forwards its console output (the "IMG,..." lines) back to the PC.
 *
 * UART protocol (PC -> board is binary, board -> PC is text lines; board lines that start with '#' are ours):
 *   'P'                               -> "#PONG id=<hex> bram=<bytes> fclk_hz=<Hz> baud=<baud>"
 *   'W' addr:u32 len:u32 data crc:u32 -> "#OK" | "#ERR ..."     store bytes at Ibex address addr in the staging copy
 *   'R' timeout_s:u32                 -> Ibex console lines, then
 *                                        "#DONE pl_cycles=.. arm_us=.. ddr_reads=.. ddr_writes=.. ddr_wait=.. err=.. timeout=.."
 *                                        (all decimal)
 *   'Z'                               -> "#OK"                     forget the staging copy (lengths to 0)
 * (all integers little-endian; crc = CRC-32 of data, the zlib polynomial)
 *
 * The staging copy is never touched by the Ibex, so the PC only has to send what changed between two programs
 * (in practice: the 5 test inputs of the next batch).
 */
#include <stdint.h>
#include <string.h>

#include "xil_cache.h"
#include "xil_io.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "xtime_l.h"
#include "xuartps_hw.h"

/* ---- addresses and sizes (must match vivado/build.tcl and rtl/ibex_zybo_soc.sv) */
#define IBEX_REGS        0x43C00000u
#define R_CTRL           0x00
#define R_STATUS         0x04
#define R_FIFO           0x08
#define R_CYC_LO         0x0C
#define R_CYC_HI         0x10
#define R_LOAD_ADDR      0x14
#define R_LOAD_DATA      0x18
#define R_DDR_BASE       0x1C
#define R_ID             0x20
#define R_BRAM_BYTES     0x24
#define R_DDR_READS      0x28
#define R_DDR_WRITES     0x2C
#define R_DDR_WAIT_LO    0x30
#define R_DDR_WAIT_HI    0x34

#define IBEX_BRAM_ADDR   0x00100000u          /* Ibex view */
#define IBEX_DDR_ADDR    0x80000000u          /* Ibex view of the DDR window */
#define DDR_WINDOW_PHYS  0x10000000u          /* where that window lives in the Zynq DDR (256 MB mark) */
#define DDR_WINDOW_SIZE  (16u << 20)
#define BRAM_MAX         (256u << 10)

#define UART_BASE        0xE0001000u          /* PS UART1 = the Zybo's USB-UART */
#ifndef BAUD
#define BAUD             460800u
#endif
#ifdef XPAR_XUARTPS_0_UART_CLK_FREQ_HZ
#define UART_REF_HZ      XPAR_XUARTPS_0_UART_CLK_FREQ_HZ
#elif defined(XPAR_XUARTPS_0_CLOCK_FREQ)
#define UART_REF_HZ      XPAR_XUARTPS_0_CLOCK_FREQ
#else
#define UART_REF_HZ      100000000u
#endif
#ifdef XPAR_PS7_CORTEXA9_0_CPU_CLK_FREQ_HZ
#define CPU_HZ           XPAR_PS7_CORTEXA9_0_CPU_CLK_FREQ_HZ
#else
#define CPU_HZ           650000000u
#endif

static inline void wr(uint32_t off, uint32_t v) { Xil_Out32(IBEX_REGS + off, v); }
static inline uint32_t rd(uint32_t off) { return Xil_In32(IBEX_REGS + off); }

/* ---- staging copies (in the ARM's own DDR, well below DDR_WINDOW_PHYS) */
static uint8_t stage_bram[BRAM_MAX] __attribute__((aligned(64)));
static uint8_t stage_ddr[DDR_WINDOW_SIZE] __attribute__((aligned(64)));
static uint32_t bram_len, ddr_len, bram_bytes;

/* ---- UART helpers */
static void uart_set_baud(uint32_t baud)
{
    /* baud = ref / (CD * (BDIV + 1)): pick the best CD for BDIV 4..254 */
    uint32_t best_cd = 0, best_bdiv = 0, best_err = 0xFFFFFFFFu;
    for (uint32_t bdiv = 4; bdiv < 255; bdiv++) {
        uint32_t cd = (UART_REF_HZ + baud * (bdiv + 1) / 2) / (baud * (bdiv + 1));
        if (cd < 1 || cd > 65535)
            continue;
        uint32_t real = UART_REF_HZ / (cd * (bdiv + 1));
        uint32_t err = real > baud ? real - baud : baud - real;
        if (err < best_err) {
            best_err = err;
            best_cd = cd;
            best_bdiv = bdiv;
        }
    }
    while (!(Xil_In32(UART_BASE + XUARTPS_SR_OFFSET) & XUARTPS_SR_TXEMPTY))
        ;
    uint32_t cr = Xil_In32(UART_BASE + XUARTPS_CR_OFFSET);
    Xil_Out32(UART_BASE + XUARTPS_CR_OFFSET, XUARTPS_CR_RX_DIS | XUARTPS_CR_TX_DIS);
    Xil_Out32(UART_BASE + XUARTPS_BAUDGEN_OFFSET, best_cd);
    Xil_Out32(UART_BASE + XUARTPS_BAUDDIV_OFFSET, best_bdiv);
    Xil_Out32(UART_BASE + XUARTPS_CR_OFFSET, XUARTPS_CR_RXRST | XUARTPS_CR_TXRST);
    Xil_Out32(UART_BASE + XUARTPS_CR_OFFSET, (cr & ~(XUARTPS_CR_RX_DIS | XUARTPS_CR_TX_DIS))
                                             | XUARTPS_CR_RX_EN | XUARTPS_CR_TX_EN);
}

static inline int uart_has_byte(void) { return XUartPs_IsReceiveData(UART_BASE); }
static inline uint8_t uart_get(void) { return XUartPs_RecvByte(UART_BASE); }
static inline void uart_put(uint8_t c) { XUartPs_SendByte(UART_BASE, c); }

static void put_str(const char *p)
{
    while (*p)
        uart_put((uint8_t)*p++);
}

static void put_u64(uint64_t v)                          /* decimal; xil_printf has no 64-bit format */
{
    char buf[21];
    int n = 0;
    do {
        buf[n++] = (char)('0' + v % 10);
        v /= 10;
    } while (v);
    while (n)
        uart_put((uint8_t)buf[--n]);
}

static uint32_t uart_get32(void)
{
    uint32_t v = 0;
    for (int i = 0; i < 4; i++)
        v |= (uint32_t)uart_get() << (8 * i);
    return v;
}

static uint32_t crc32_update(uint32_t crc, uint8_t b)
{
    crc ^= b;
    for (int k = 0; k < 8; k++)
        crc = (crc >> 1) ^ (0xEDB88320u & -(crc & 1u));
    return crc;
}

/* ---- commands */
static void cmd_write(void)
{
    uint32_t addr = uart_get32(), len = uart_get32();
    uint8_t *dst = 0;
    uint32_t *len_p = 0, off = 0, cap = 0;
    if (addr >= IBEX_BRAM_ADDR && addr < IBEX_BRAM_ADDR + bram_bytes) {
        dst = stage_bram; off = addr - IBEX_BRAM_ADDR; cap = bram_bytes; len_p = &bram_len;
    } else if (addr >= IBEX_DDR_ADDR && addr < IBEX_DDR_ADDR + DDR_WINDOW_SIZE) {
        dst = stage_ddr; off = addr - IBEX_DDR_ADDR; cap = DDR_WINDOW_SIZE; len_p = &ddr_len;
    }
    uint32_t crc = 0xFFFFFFFFu;
    int ok = dst && off + len <= cap;
    for (uint32_t i = 0; i < len; i++) {
        uint8_t b = uart_get();
        crc = crc32_update(crc, b);
        if (ok)
            dst[off + i] = b;
    }
    uint32_t want = uart_get32();
    if (!ok) {
        xil_printf("#ERR range addr=%08x len=%d\r\n", addr, len);
        return;
    }
    if ((crc ^ 0xFFFFFFFFu) != want) {
        xil_printf("#ERR crc\r\n");
        return;
    }
    if (off + len > *len_p)
        *len_p = off + len;
    xil_printf("#OK\r\n");
}

static void ibex_load(void)
{
    wr(R_CTRL, 0);                                     /* hold the core in reset, clear counters and FIFO */
    uint32_t words = (bram_len + 3) / 4;
    wr(R_LOAD_ADDR, 0);
    for (uint32_t i = 0; i < words; i++) {
        uint32_t w;
        memcpy(&w, &stage_bram[4 * i], 4);
        wr(R_LOAD_DATA, w);
    }
    if (ddr_len) {
        memcpy((void *)DDR_WINDOW_PHYS, stage_ddr, ddr_len);
        Xil_DCacheFlushRange(DDR_WINDOW_PHYS, ddr_len);   /* the Ibex reads DDR through HP0, past the caches */
    }
    wr(R_DDR_BASE, DDR_WINDOW_PHYS);
}

static void cmd_run(void)
{
    uint32_t timeout_s = uart_get32();
    ibex_load();
    XTime t0, t1, now;
    XTime_GetTime(&t0);
    wr(R_CTRL, 1);
    int done_seen = 0, timed_out = 0;
    while (1) {
        uint32_t st = rd(R_STATUS);
        uint32_t v;
        while ((v = rd(R_FIFO)) >> 31)
            uart_put((uint8_t)v);
        if (st & 1) {
            if (done_seen)
                break;                                    /* drained once more after done */
            done_seen = 1;
            XTime_GetTime(&t1);
            continue;
        }
        if (st & 16)
            break;                                        /* data bus error */
        XTime_GetTime(&now);
        if (timeout_s && (now - t0) / COUNTS_PER_SECOND > timeout_s) {
            timed_out = 1;
            t1 = now;
            break;
        }
    }
    uint64_t cyc = ((uint64_t)rd(R_CYC_HI) << 32) | rd(R_CYC_LO);
    uint64_t wait = ((uint64_t)rd(R_DDR_WAIT_HI) << 32) | rd(R_DDR_WAIT_LO);
    uint64_t us = (uint64_t)(t1 - t0) * 1000000u / COUNTS_PER_SECOND;
    uint32_t st = rd(R_STATUS);
    put_str("#DONE pl_cycles="); put_u64(cyc);
    put_str(" arm_us=");         put_u64(us);
    put_str(" ddr_reads=");      put_u64(rd(R_DDR_READS));
    put_str(" ddr_writes=");     put_u64(rd(R_DDR_WRITES));
    put_str(" ddr_wait=");       put_u64(wait);
    put_str(" err=");            put_u64((st >> 4) & 1);
    put_str(" timeout=");        put_u64(timed_out);
    put_str("\r\n");
    wr(R_CTRL, 0);
}

int main(void)
{
    Xil_DCacheEnable();
    bram_bytes = rd(R_BRAM_BYTES);
    if (bram_bytes == 0 || bram_bytes > BRAM_MAX)
        bram_bytes = 192u << 10;
    xil_printf("\r\n#ZYBO-IBEX id=%08x bram=%d, switching the UART to %d baud\r\n", rd(R_ID), bram_bytes, BAUD);
    if ((uintptr_t)(stage_ddr + DDR_WINDOW_SIZE) > DDR_WINDOW_PHYS)
        xil_printf("#WARNING the staging buffers overlap the Ibex DDR window: fix the ARM linker script\r\n");
    uart_set_baud(BAUD);
    wr(R_CTRL, 0);
    while (1) {
        uint8_t c = uart_get();
        switch (c) {
        case 'P': {
            uint32_t fclk = 0;
#ifdef XPAR_PS7_FPGA0_CLK_FREQ_HZ
            fclk = XPAR_PS7_FPGA0_CLK_FREQ_HZ;
#endif
            xil_printf("#PONG id=%08x bram=%d fclk_hz=%d baud=%d cpu_hz=%d\r\n", rd(R_ID), bram_bytes, fclk, BAUD,
                       CPU_HZ);
            break;
        }
        case 'W': cmd_write(); break;
        case 'R': cmd_run(); break;
        case 'Z': bram_len = ddr_len = 0; xil_printf("#OK\r\n"); break;
        default: break;                                   /* ignore noise (e.g. line ends from a terminal) */
        }
    }
    return 0;
}
