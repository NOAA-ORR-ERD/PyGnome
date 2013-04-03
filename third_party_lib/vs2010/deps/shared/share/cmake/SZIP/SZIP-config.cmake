#-----------------------------------------------------------------------------
# SZIP Config file for compiling against SZIP install directory
#-----------------------------------------------------------------------------

GET_FILENAME_COMPONENT (SELF_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
GET_FILENAME_COMPONENT(_IMPORT_PREFIX "${SELF_DIR}" PATH)
GET_FILENAME_COMPONENT(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
GET_FILENAME_COMPONENT(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)

GET_FILENAME_COMPONENT (SZIP_INCLUDE_DIRS "${_IMPORT_PREFIX}/include")

#-----------------------------------------------------------------------------
# Version Strings
#-----------------------------------------------------------------------------
SET (SZIP_VERSION_STRING 2.1)
SET (SZIP_VERSION_MAJOR  2)
SET (SZIP_VERSION_MINOR  1)

#-----------------------------------------------------------------------------
# Don't include targets if this file is being picked up by another
# project which has already build SZIP as a subproject
#-----------------------------------------------------------------------------
IF (NOT TARGET "SZIP")
  INCLUDE (${SELF_DIR}/SZIP-targets.cmake)
  SET (SZIP_LIBRARIES "szip")
ENDIF (NOT TARGET "SZIP")

#-----------------------------------------------------------------------------
# Unfinished
#-----------------------------------------------------------------------------

#
# To be continued ...
#
# XXX_INCLUDE_DIRS         The final set of include directories listed in one variable for use by client code.  This should not be a cache entry.
# XXX_LIBRARIES             The libraries to link against to use XXX. These should include full paths.  This should not be a cache entry.
# XXX_DEFINITIONS           Definitions to use when compiling code that uses XXX. This really shouldn't include options such as (-DHAS_JPEG)that a client source-code file uses to decide whether to #include <jpeg.h>
# XXX_EXECUTABLE            Where to find the XXX tool.
# XXX_YYY_EXECUTABLE        Where to find the YYY tool that comes with XXX.
# XXX_LIBRARY_DIRS         Optionally, the final set of library directories listed in one variable for use by client code.  This should not be a cache entry.
# XXX_ROOT_DIR              Where to find the base directory of XXX.
# XXX_VERSION_YY           Expect Version YY if true. Make sure at most one of these is ever true.
# XXX_WRAP_YY               If False, do not try to use the relevent CMake wrapping command.
# XXX_YY_FOUND              If False, optional YY part of XXX sytem is not available.
# XXX_FOUND                 Set to false, or undefined, if we haven't found, or don't want to use XXX.
# XXX_RUNTIME_LIBRARY_DIRS Optionally, the runtime library search path for use when running an executable linked to shared libraries.
#                          The list should be used by user code to create the PATH on windows or LD_LIBRARY_PATH on unix.
#                          This should not be a cache entry.
