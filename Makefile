include Makefile.common

OUT = lib/libscan.so
OBJS = harris/scan.o sengupta/segscan.o common/scanref.o

all:
	cd common; make
	cd harris_sequential; make
	cd harris; make
	cd sengupta; make
	make $(OUT)

$(OUT): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OPENCL_LIB) $(OPENCL_INC) $(SHARED) -o $@ $^ -L../lib -lclwrapper

.PHONY: clean
clean:
	cd common; make clean
	cd harris_sequential; make clean
	cd harris; make clean
	cd sengupta; make clean
	rm -f $(OUT)
