#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.63])
AC_INIT(onlineml, 0.2, takuyamakino15@gmail.com)
AM_INIT_AUTOMAKE
AC_CONFIG_HEADERS([config.h])

AC_CONFIG_FILES([Makefile
		 onlineml/common/Makefile
		 onlineml/cmd/Makefile
		 onlineml/learner/Makefile
		 onlineml/util/Makefile
		 swig/Makefile
		 ])


# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_LIBTOOL

# Checks for libraries.
AC_CONFIG_MACRO_DIR([m4])

# Checks for header files.

# Checks for typedefs, structures, and compiler characteristics.

# Checks for library functions.

# Checks for arguments

#AC_SUBST

#AC_ARG_WITH([gperf],
#        [AS_HELP_STRING([--with-gperf],
#        [profiler @<:@default=check@:>@])],
#        [],
#        [with_gperf=check])

#        LIBREADLINE=
#        AS_IF([test "x$with_gperf" != xno],
#        [AC_CHECK_LIB([readline], [main],
#        [AC_SUBST([LIBREADLINE], ["-lreadline -lncurses"])
#        AC_DEFINE([HAVE_LIBREADLINE], [1],
#        [Define if you have libreadline])
#        ],
#        [if test "x$with_readline" != xcheck; then
#        AC_MSG_FAILURE(
#        [--with-readline was given, but test for readline failed])
#        fi
#        ], -lncurses)]
#)


CXX="g++"
AC_OUTPUT
