include ../Makefile.common

all: harris

OBJ = scan.o ../common/*.o

ifneq ($(EMBED_CL), '')
scan.o: scan.cpp scan.cl.h
	$(CXX) $(CXXFLAGS) $(OPENCL_INC) $(INCLUDEDIR) -D EMBED_CL=$(EMBED_CL) -c $< -o $@
endif

harris: main.cpp $(OBJ)
	$(CXX) $(CXXFLAGS) $(OPENCL_LIB) $(OPENCL_INC) $(INCLUDEDIR) $(LIB) $^ -o $@

ifneq ($(UNITTEST_DIR), '')
test: scan_unittest.cpp $(OBJ) $(UNITTEST_DIR)/libUnitTest++.a
	$(CXX) $(CXXFLAGS) $(OPENCL_LIB) $(OPENCL_INC) $(INCLUDEDIR) -I $(UNITTEST_DIR)/src $(LIB) -o $@ $^
endif

clean:
	rm -f test harris scan.cl.h $(CLEAN)
