# https://www.gnu.org/software/automake/manual/html_node/Libtool-Convenience-Libraries.html
lib_LTLIBRARIES = libonlineml.la
libonlineml_la_SOURCES =
libonlineml_la_LIBADD = \
			../util/libutil.la
			../common/libcommon.la
			../learner/liblearner.la

bin_PROGRAMS = train_onlineml_model
train_onlineml_model_SOURCES = learn_online_model.cpp
train_onlineml_model_LDADD = libonlineml.la

