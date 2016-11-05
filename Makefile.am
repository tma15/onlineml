SUBDIRS = onlineml/common onlineml/learner onlineml/util onlineml/cmd swig

python:
	cd swig && \
	swig -c++ -I../ -python onlineml.swigcxx && \
	python setup.py build_ext --include-dirs=@prefix@/include --library-dirs=@prefix@/lib && \
	python setup.py install && \
	cd -

ACLOCAL_AMFLAGS = -I m4
