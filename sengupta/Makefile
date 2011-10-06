include ../Makefile.common

all: sengupta

sengupta: main.cpp
	$(CXX) $(CXXFLAGS) $(OPENCL) -I $(INCLUDEDIR) $^ -o $@

clean:
	rm -f sengupta $(CLEAN)
