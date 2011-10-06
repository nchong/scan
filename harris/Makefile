include ../Makefile.common

all: harris

harris: main.cpp
	$(CXX) $(CXXFLAGS) $(OPENCL) -I $(INCLUDEDIR) $^ -o $@

clean:
	rm -f harris $(CLEAN)
