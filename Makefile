all:
	cd harris_sequential; make
	cd harris; make
	cd sengupta; make

clean:
	cd harris_sequential; make clean
	cd harris; make clean
	cd sengupta; make clean
