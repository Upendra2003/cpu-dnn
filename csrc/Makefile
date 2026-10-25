CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -fPIC

libroba.so: roba.c roba.h
	$(CC) $(CFLAGS) -shared -o $@ roba.c

clean:
	rm -f libroba.so
