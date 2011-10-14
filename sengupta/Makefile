include ../Makefile.common

all: sengupta

OBJ = segscan.o ../common/*.o

sengupta: main.cpp $(OBJ)
	$(CXX) $(CXXFLAGS) $(OPENCL) $(INCLUDEDIR) $(LIB) $^ -o $@

ifneq ($(UNITTEST_DIR), '')
test: segscan_unittest.cpp $(OBJ) $(UNITTEST_DIR)/libUnitTest++.a
	$(CXX) $(CXXFLAGS) $(OPENCL) $(INCLUDEDIR) -I $(UNITTEST_DIR)/src $(LIB) -o $@ $^
endif

clean:
	rm -f test sengupta $(CLEAN)
