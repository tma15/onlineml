SUBDIRS = onlineml/common onlineml/learner onlineml/util onlineml/sample

PREFIX = onlineml
python:
	cd swig && \
	swig -c++ -I../src/include -python $(PREFIX).swigcxx && \
	python setup.py build_ext --inplace && \
	cd -


lib_LTLIBRARIES = libonlineml.la
libonlineml_la_SOURCES = \
			  onlineml/util/string_proc.h
