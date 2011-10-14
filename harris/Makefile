include ../Makefile.common

all: harris

OBJ = scan.o ../common/*.o

harris: main.cpp $(OBJ)
	$(CXX) $(CXXFLAGS) $(OPENCL) $(INCLUDEDIR) $(LIB) $^ -o $@

ifneq ($(UNITTEST_DIR), '')
test: scan_unittest.cpp $(OBJ) $(UNITTEST_DIR)/libUnitTest++.a
	$(CXX) $(CXXFLAGS) $(OPENCL) $(INCLUDEDIR) -I $(UNITTEST_DIR)/src $(LIB) -o $@ $^
endif

clean:
	rm -f test harris $(CLEAN)
