# https://www.gnu.org/software/automake/manual/html_node/Libtool-Convenience-Libraries.html
lib_LTLIBRARIES = libonlineml.la
libonlineml_la_SOURCES =
libonlineml_la_LIBADD = \
			../util/libutil.la
			../common/libcommon.la
			../learner/liblearner.la

bin_PROGRAMS = train_onlineml_model test_onlineml_model
train_onlineml_model_SOURCES = train_online_model.cpp arg.hpp
train_onlineml_model_LDADD = libonlineml.la

test_onlineml_model_SOURCES = test_onlineml_model.cpp arg.hpp
test_onlineml_model_LDADD = libonlineml.la

if DEBUG
DBG = -DDEBUG
PG = -pg
else
DBG =
PG =
endif

AM_CPPFLAGS = -pthread -O9 $(DBG) $(PG)
AM_LDFLAGS = -lpthread $(PG)
