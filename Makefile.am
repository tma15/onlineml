SUBDIRS = onlineml/common onlineml/learner onlineml/util onlineml/cmd swig

python:
	cd swig && \
	swig -c++ -I../ -python onlineml.swigcxx && \
	python setup.py build_ext --inplace --include-dirs=@prefix@/include \
	--library-dirs=@prefix@/lib && \
	cd -

ACLOCAL_AMFLAGS = -I m4
