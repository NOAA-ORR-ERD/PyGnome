# to jcfr@github henryiii


cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(NetCDF REQUIRED MODULE)

find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRED)

find_program(CYTHON "cython")

set (EXT_NAMES cy_basic_types
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

set( CPP_FILES RectGridVeL_c.cpp
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

list(TRANSFORM CPP_FILES PREPEND "lib_gnome/")

add_library(gnome SHARED ${CPP_FILES})

target_link_libraries(gnome PRIVATE NetCDF::NetCDF)

target_compile_definitions(gnome PUBLIC pyGNOME=1)

target_include_directories(gnome PUBLIC lib_gnome)

foreach(ext IN LISTS EXT_NAMES)

    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/gnome/cy_gnome/${ext}.cpp"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gnome/cy_gnome/${ext}.pyx"
      VERBATIM
      COMMAND "${CYTHON}" --cplus -3
              "${CMAKE_CURRENT_SOURCE_DIR}/gnome/cy_gnome/${ext}.pyx"
              --output-file
              "${CMAKE_CURRENT_BINARY_DIR}/gnome/cy_gnome/${ext}.cpp")
    
    python_add_library(${ext} MODULE "${CMAKE_CURRENT_BINARY_DIR}/gnome/cy_gnome/${ext}.cpp"
                   WITH_SOABI)

    target_link_libraries(${ext} PRIVATE gnome Python::NumPy)


    #if(APPLE)
      set_target_properties(${ext} PROPERTIES INSTALL_RPATH "@loader_path/.")
    #elseif(UNIX)
    #  set_target_properties(${ext} PROPERTIES INSTALL_RPATH "$ORIGIN/.")
    #endif()

    install(TARGETS ${ext} DESTINATION gnome/cy_gnome)


endforeach()

install(TARGETS gnome DESTINATION gnome/cy_gnome)


