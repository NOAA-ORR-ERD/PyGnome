# to jcfr@github henryiii


cmake_minimum_required(VERSION 3.15...3.29)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(NetCDF REQUIRED MODULE)

find_package(
    Python
    COMPONENTS Interpreter Development.Module NumPy
    REQUIRED
)

find_program(CYTHON "cython")

# On Windows, we need to set these environmental variables so that the linker
# can properly find the .lib file associated with a dynamic library (.dll).
# And they must be set before any add_library() commands
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(BUILD_SHARED_LIBS TRUE)
endif()

set(EXT_NAMES
    cy_basic_types
    cy_mover
    cy_helpers
    cy_wind_mover
    cy_current_mover
    cy_cats_mover
    cy_component_mover
    cy_gridcurrent_mover
    cy_gridwind_mover
    cy_ice_mover
    cy_ice_wind_mover
    cy_currentcycle_mover
    cy_ossm_time
    cy_random_mover
    cy_random_mover_3d
    cy_rise_velocity_mover
    cy_land_check
    cy_grid_map
    cy_shio_time
    cy_grid
    cy_grid_rect
    cy_grid_curv
    cy_weatherers
)

set(LIBGNOME_CPP_FILES
    RectGridVeL_c.cpp
    MemUtils.cpp
    Mover_c.cpp
    Replacements.cpp
    ClassID_c.cpp
    Random_c.cpp
    TimeValuesIO.cpp
    GEOMETRY.cpp
    OSSMTimeValue_c.cpp
    TimeValue_c.cpp
    RectUtils.cpp
    WindMover_c.cpp
    CompFunctions.cpp
    StringFunctions.cpp
    OUTILS.cpp
    CATSMover_c.cpp
    CurrentMover_c.cpp
    ComponentMover_c.cpp
    ShioTimeValue_c.cpp
    ShioHeight.cpp
    TriGridVel_c.cpp
    DagTree.cpp
    DagTreeIO.cpp
    ShioCurrent1.cpp
    ShioCurrent2.cpp
    GridCurrentMover_c.cpp
    GridWindMover_c.cpp
    IceMover_c.cpp
    IceWindMover_c.cpp
    CurrentCycleMover_c.cpp
    TimeGridVel_c.cpp
    TimeGridWind_c.cpp
    MakeTriangles.cpp
    MakeDagTree.cpp
    GridMap_c.cpp
    GridMapUtils.cpp
    RandomVertical_c.cpp
    RiseVelocity_c.cpp
    Weatherers_c.cpp
)

list(TRANSFORM LIBGNOME_CPP_FILES PREPEND "lib_gnome/")

add_library(lib_gnome SHARED ${LIBGNOME_CPP_FILES})

target_link_libraries(lib_gnome PRIVATE NetCDF::NetCDF)

target_compile_definitions(lib_gnome PUBLIC pyGNOME=1)

target_include_directories(lib_gnome PUBLIC lib_gnome)

install(TARGETS lib_gnome LIBRARY DESTINATION gnome/cy_gnome)


set(CY_EXT_SOURCE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/gnome/cy_gnome
)
set(CY_EXT_BINARY_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/gnome/cy_gnome
)

# cython can't create the output .cpp file if the underlying directory
# doesn't exist.
if(NOT EXISTS ${CY_EXT_BINARY_DIR})
    message("Creating output directory '${CY_EXT_BINARY_DIR}'")
    file(MAKE_DIRECTORY "${CY_EXT_BINARY_DIR}")
endif()

foreach(ext IN LISTS EXT_NAMES)
    add_custom_command(
        OUTPUT "${CY_EXT_BINARY_DIR}/${ext}.cpp"
        DEPENDS "${CY_EXT_SOURCE_DIR}/${ext}.pyx"
        VERBATIM
        COMMAND "${CYTHON}" --cplus -3
                "${CY_EXT_SOURCE_DIR}/${ext}.pyx"
                --output-file
                "${CY_EXT_BINARY_DIR}/${ext}.cpp"
    )
    
    python_add_library(
        ${ext}
        MODULE "${CY_EXT_BINARY_DIR}/${ext}.cpp"
        WITH_SOABI
    )

    target_link_libraries(${ext} PRIVATE lib_gnome Python::NumPy)


    if(APPLE)
      set_target_properties(${ext} PROPERTIES INSTALL_RPATH "@loader_path/.")
    elseif(UNIX)
      set_target_properties(${ext} PROPERTIES INSTALL_RPATH "$ORIGIN/.")
    endif()

    install(TARGETS ${ext} DESTINATION gnome/cy_gnome)

endforeach()


#
# Extension: cy_point_in_polygon
#

set(GEO_SOURCE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/gnome/utilities/geometry
)
set(GEO_BINARY_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/gnome/utilities/geometry
)

# Now we make the c_point_in_polygon.c file into a library
# This will have to do until we figure out how to make a .pyx extension with
# additional .c files
# No need to make it dynamic since it
set(POINT_IN_POLY_C_FILES
    c_point_in_polygon.c
)

list(TRANSFORM POINT_IN_POLY_C_FILES PREPEND "gnome/utilities/geometry/")

add_library(point_in_polygon SHARED ${POINT_IN_POLY_C_FILES})

install(TARGETS point_in_polygon DESTINATION gnome/utilities/geometry)

# cython can't create the output .cpp file if the underlying directory
# doesn't exist.
if(NOT EXISTS ${GEO_BINARY_DIR})
    message("Creating output directory '${GEO_BINARY_DIR}'")
    file(MAKE_DIRECTORY "${GEO_BINARY_DIR}")
endif()

# now we make the cy_point_in_polygon extension
add_custom_command(
  OUTPUT "${GEO_BINARY_DIR}/cy_point_in_polygon.c"
  DEPENDS "${GEO_SOURCE_DIR}/cy_point_in_polygon.pyx"
  VERBATIM
  COMMAND "${CYTHON}" -3 -v
          "${GEO_SOURCE_DIR}/cy_point_in_polygon.pyx"
          --output-file
          "${GEO_BINARY_DIR}/cy_point_in_polygon.c"
)

python_add_library(
    cy_point_in_polygon
    MODULE "${GEO_BINARY_DIR}/cy_point_in_polygon.c"
    WITH_SOABI
)

target_link_libraries(cy_point_in_polygon PRIVATE point_in_polygon Python::NumPy)

if(APPLE)
    set_target_properties(cy_point_in_polygon PROPERTIES INSTALL_RPATH "@loader_path/.")
elseif(UNIX)
    set_target_properties(cy_point_in_polygon PROPERTIES INSTALL_RPATH "$ORIGIN/.")
endif()

install(TARGETS cy_point_in_polygon DESTINATION gnome/utilities/geometry)
