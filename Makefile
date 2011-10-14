all:
	cd clwrapper; make
	cd common; make
	cd harris_sequential; make
	cd harris; make
	cd sengupta; make

clean:
	cd common; make clean
	cd harris_sequential; make clean
	cd harris; make clean
	cd sengupta; make clean

veryclean:
	cd clwrapper; make clean
	cd common; make clean
	cd harris_sequential; make clean
	cd harris; make clean
	cd sengupta; make clean
