cmake_minimum_required(VERSION 3.12)
project(trac_ik_lib)

include(GNUInstallDirs)

# Find MuJoCo
find_library(MUJOCO_LIB mujoco
    PATHS
      $ENV{CONDA_PREFIX}/lib
    NO_DEFAULT_PATH
)

if(NOT MUJOCO_LIB)
    message(FATAL_ERROR "Could not find MuJoCo library.")
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
find_package(NLopt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(urdfdom REQUIRED)

add_library(trac_ik
  SHARED
  src/kdl_tl.cpp
  src/nlopt_ik.cpp
  src/trac_ik.cpp
  src/mjcf_parser.cpp
  src/kdl_parser.cpp)
target_link_libraries(trac_ik
  PUBLIC
  Eigen3::Eigen
  NLopt::nlopt
  spdlog::spdlog
  orocos-kdl
  urdfdom::urdf_parser
  urdfdom::urdfdom_model
  ${MUJOCO_LIB})
target_compile_features(trac_ik PRIVATE cxx_std_20)
target_include_directories(trac_ik
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

pybind11_add_module(trac_ik_py src/trac_ik_py.cpp)
target_link_libraries(trac_ik_py PRIVATE trac_ik)
target_compile_features(trac_ik_py PRIVATE cxx_std_20)
pybind11_extension(trac_ik_py)
pybind11_strip(trac_ik_py)

# Determine the python installation directory
set(PYTHON_SITE_PACKAGES ${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages)
message(STATUS "Python site-packages directory: " ${PYTHON_SITE_PACKAGES})

install(TARGETS trac_ik_py
  LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}
)

install(TARGETS trac_ik
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
