include ../Makefile.common

all: sengupta

OBJ = segscan.o ../common/*.o

ifeq ($(EMBED_CL), 'true')
segscan.o: segscan.cpp segscan.cl.h
	$(CXX) $(CXXFLAGS) $(OPENCL_INC) $(INCLUDEDIR) -D EMBED_CL=$(EMBED_CL) -c $< -o $@
endif

sengupta: main.cpp $(OBJ)
	$(CXX) $(CXXFLAGS) $(OPENCL_LIB) $(OPENCL_INC) $(INCLUDEDIR) $(LIB) $^ -o $@

ifneq ($(UNITTEST_DIR), '')
test: segscan_unittest.cpp $(OBJ) $(UNITTEST_DIR)/libUnitTest++.a
	$(CXX) $(CXXFLAGS) $(OPENCL_LIB) $(OPENCL_INC) $(INCLUDEDIR) -I $(UNITTEST_DIR)/src $(LIB) -o $@ $^
endif

clean:
	rm -f test sengupta segscan.cl.h $(CLEAN)
