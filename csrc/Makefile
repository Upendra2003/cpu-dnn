CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -fPIC

libroba.so: roba.c roba_core_test.c roba.h roba_core.h
	$(CC) $(CFLAGS) -shared -o $@ roba.c roba_core_test.c

clean:
	rm -f libroba.so
