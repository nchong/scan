include ../Makefile.common

all: harris_sequential

OBJ = ../common/*.o

harris_sequential: main.cpp $(OBJ)
	$(CXX) $(CXXFLAGS) $(INCLUDEDIR) $^ -o $@

clean:
	rm -f harris_sequential $(CLEAN)
