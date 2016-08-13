
SUBDIRS = onlineml/common onlineml/learner onlineml/util onlineml/sample

# https://www.gnu.org/software/automake/manual/html_node/Libtool-Convenience-Libraries.html
lib_LTLIBRARIES = libonlineml.la
libonlineml_la_SOURCES =
libonlineml_la_LIBADD = \
			onlineml/util/libutil.la
			onlineml/common/libcommon.la
			onlineml/learner/liblearner.la

python:
	cd swig && \
	swig -c++ -I../ -python onlineml.swigcxx && \
	python setup.py build_ext --inplace --include-dirs=@prefix@/include \
	--library-dirs=@prefix@/lib && \
	cd -

ACLOCAL_AMFLAGS = -I m4
