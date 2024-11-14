AC_INIT([NGram],[1.3.16],[ngram@www.opengrm.org])
AM_INIT_AUTOMAKE([foreign nostdinc -Wall -Werror])
AM_PROG_AR

AC_PROG_CXX
# This library does not throw exceptions, so we do not generate exception
# handling code. However, users are free to re-enable exception handling.
CXX="$CXX -std=c++17 -fno-exceptions"

AC_DISABLE_STATIC
AC_PROG_LIBTOOL

AC_CONFIG_HEADERS([config.h])
AC_CONFIG_SRCDIR([src/include/ngram/ngram-model.h])
AC_CONFIG_FILES([
  Makefile
  src/Makefile
  src/include/Makefile
  src/lib/Makefile
  src/bin/Makefile
  src/test/Makefile
])
AC_CONFIG_MACRO_DIR([m4])
AC_LANG([C++])

AC_CHECK_HEADER([fst/fst.h], [],
 [AC_MSG_ERROR([fst/fst.h header not found])]
)

AC_CHECK_HEADER([fst/extensions/far/far.h], [],
 [AC_MSG_ERROR([fst/extensions/far/far.h header not found])]
)

AC_CHECK_LIB([dl], dlopen, [DL_LIBS=-ldl])
AC_SUBST([DL_LIBS])

AC_OUTPUT
