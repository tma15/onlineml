SUBDIRS = src/include src/sample

PREFIX = onlineml
python:
	cd swig && \
	swig -c++ -I../src/include -python $(PREFIX).swigcxx && \
	python setup.py build_ext --inplace && \
	cd -
