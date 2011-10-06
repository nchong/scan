all:
	cd harris_sequential; make all
	cd harris; make all
	cd sengupta; make all

clean:
	cd harris_sequential; make clean
	cd harris; make clean
	cd sengupta; make clean
