#-----------------------------------------------------------------------------
# ZLIB Config file for compiling against ZLIB install directory
#-----------------------------------------------------------------------------

GET_FILENAME_COMPONENT (SELF_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
GET_FILENAME_COMPONENT(_IMPORT_PREFIX "${SELF_DIR}" PATH)
GET_FILENAME_COMPONENT(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
GET_FILENAME_COMPONENT(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)

GET_FILENAME_COMPONENT (ZLIB_INCLUDE_DIRS "${_IMPORT_PREFIX}/include")

#-----------------------------------------------------------------------------
# Version Strings
#-----------------------------------------------------------------------------
SET (ZLIB_VERSION_STRING 1.2)
SET (ZLIB_VERSION_MAJOR  1.2)
SET (ZLIB_VERSION_MINOR  5)

#-----------------------------------------------------------------------------
# Don't include targets if this file is being picked up by another
# project which has already build ZLIB as a subproject
#-----------------------------------------------------------------------------
IF (NOT TARGET "ZLIB")
  INCLUDE (${SELF_DIR}/ZLIB-targets.cmake)
  SET (ZLIB_LIBRARIES "zlib")
ENDIF (NOT TARGET "ZLIB")
