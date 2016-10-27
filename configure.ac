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

AC_ARG_ENABLE([debug],
	      [  --enable-debug    Turn on debugging],
	      [case "${enableval}" in
	         yes) debug=true ;;
		 no)  debug=false ;;
		 *) AC_MSG_ERROR([bad value ${enableval} for --enable-debug]) ;;
	     esac],[debug=false])
AM_CONDITIONAL([DEBUG], [test x$debug = xtrue])


CXX="g++"
AC_OUTPUT
