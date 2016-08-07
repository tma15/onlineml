
SUBDIRS = onlineml/common onlineml/learner onlineml/util onlineml/sample

PREFIX = onlineml
python:
	cd swig && \
	swig -c++ -I../ -python $(PREFIX).swigcxx && \
	python setup.py build_ext --inplace && \
	cd -



