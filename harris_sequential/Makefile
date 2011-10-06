include ../Makefile.common

all: harris_sequential

harris_sequential: main.cpp
	$(CXX) $(CXXFLAGS) -I $(INCLUDEDIR) $^ -o $@

clean:
	rm -f harris_sequential $(CLEAN)
