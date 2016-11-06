utildir=$(includedir)/onlineml/util
util_HEADERS = string_proc.hpp

noinst_LTLIBRARIES = libutil.la
libutil_la_SOURCES = string_proc.hpp string_proc.cpp

AM_CPPFLAGS = -pthread -O9 $(DBG) $(PG) -std=c++0x
AM_LDFLAGS = -lpthread $(PG)
