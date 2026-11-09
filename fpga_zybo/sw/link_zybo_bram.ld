/* fpga_zybo linker script (bram): hw/sw/link_1mb.ld with the Zybo memory map.
   Copyright lowRISC contributors (original simple_system link.ld).
   Licensed under the Apache License, Version 2.0, see LICENSE for details.
   SPDX-License-Identifier: Apache-2.0 */

OUTPUT_ARCH(riscv)

MEMORY
{
    /* Zybo block RAM: 192 KB at 0x100000 for code and data; its last 16 KB hold the stack */
    ram         : ORIGIN = 0x00100000, LENGTH = 0x2C000
    stack       : ORIGIN = 0x0012C000, LENGTH = 0x4000
}

/* Stack information variables */
_min_stack      = 0x2000;   /* 8K - minimum stack space to reserve */
_stack_len     = LENGTH(stack);
_stack_start   = ORIGIN(stack) + LENGTH(stack);

_entry_point = _vectors_start + 0x80;
ENTRY(_entry_point)

/* The tohost address is used by Spike for a magic "stop me now" message. This
   is set to equal SIM_CTRL_CTRL (see simple_system_regs.h), which has that
   effect in simple_system simulations. Note that it must be 8-byte aligned.

   We don't read data back from Spike, so fromhost is set to some dummy value:
   we place it just above the top of the stack.
 */
tohost   = 0x20008;
fromhost = _stack_start + 0x10;

SECTIONS
{
    .vectors :
    {
        . = ALIGN(4);
		_vectors_start = .;
        KEEP(*(.vectors))
		_vectors_end = .;
    } > ram

    .text : {
        . = ALIGN(4);
        *(.text)
        *(.text.*)
    }  > ram

    .rodata : {
        . = ALIGN(4);
        /* Small RO data before large RO data */
        *(.srodata)
        *(.srodata.*)
        *(.rodata);
        *(.rodata.*)
    } > ram

    .data : {
        . = ALIGN(4);
        /* Small data before large data */
        *(.sdata)
        *(.sdata.*)
        *(.data);
        *(.data.*)
    } > ram

    .bss :
    {
        . = ALIGN(4);
        _bss_start = .;
        /* Small BSS before large BSS */
        *(.sbss)
        *(.sbss.*)
        *(.bss)
        *(.bss.*)
        *(COMMON)
        _bss_end = .;
    } > ram

    /* ensure there is enough room for stack */
    .stack (NOLOAD): {
        . = ALIGN(4);
        . = . + _min_stack ;
        . = ALIGN(4);
        stack = . ;
        _stack = . ;
    } > stack
}
