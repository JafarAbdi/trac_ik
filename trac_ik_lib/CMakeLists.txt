cmake_minimum_required(VERSION 3.12)
project(trac_ik_lib)

file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.42.0/CPM.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
  EXPECTED_HASH
    SHA256=2020b4fc42dba44817983e06342e682ecfc3d2f484a581f11cc5731fbe4dce8a)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)
cpmaddpackage(
  NAME
  eigen
  GIT_REPOSITORY
  https://gitlab.com/libeigen/eigen.git
  GIT_TAG
  3.4.0
  DOWNLOAD_ONLY
  ON)
if(eigen_ADDED)
  # Create the Eigen3::Eigen target
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  target_include_directories(Eigen3::Eigen INTERFACE ${eigen_SOURCE_DIR})

  # Also set the standard Eigen3 variables for compatibility
  set(EIGEN3_INCLUDE_DIR ${eigen_SOURCE_DIR})
  set(Eigen3_FOUND TRUE)
endif()
cpmaddpackage("gh:stevengj/nlopt@2.10.0")
cpmaddpackage("gh:gabime/spdlog@1.15.3")
cpmaddpackage(URI "gh:orocos/orocos_kinematics_dynamics@1.5.1" SOURCE_SUBDIR
              orocos_kdl)

cpmaddpackage("gh:leethomason/tinyxml2@10.0.0")
cpmaddpackage("gh:ros/urdfdom_headers#2.0.0")
cpmaddpackage("gh:ros/console_bridge#1.0.2")
if(tinyxml2_ADDED)
  set(TINYXML2_INCLUDE_DIR ${tinyxml2_SOURCE_DIR})
  set(TINYXML2_LIBRARY tinyxml2)
  set(TinyXML2_FOUND TRUE)
  set(TinyXML2_INCLUDE_DIRS ${tinyxml2_SOURCE_DIR})
  set(TinyXML2_LIBRARIES tinyxml2)
endif()
# https://github.com/cpm-cmake/CPM.cmake/issues/618
if(NOT URDFDOM_PATCHED)
  set(URDFDOM_PATCHED
      TRUE
      CACHE INTERNAL "Whether to apply patches to urdfdom")
  cpmaddpackage(URI "gh:ros/urdfdom#5.0.2" OPTIONS "BUILD_TESTING OFF" PATCHES
                "urdfdom.patch")
else()
  cpmaddpackage(URI "gh:ros/urdfdom#5.0.2" OPTIONS "BUILD_TESTING OFF")
endif()

get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
  message(STATUS ${_variableName}=${${_variableName}})
endforeach()

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

add_library(trac_ik SHARED src/kdl_tl.cpp src/nlopt_ik.cpp src/trac_ik.cpp
                           src/kdl_parser.cpp)
target_link_libraries(trac_ik PUBLIC Eigen3::Eigen nlopt spdlog::spdlog
                                     orocos-kdl urdf_parser urdfdom_model)
target_compile_features(trac_ik PRIVATE cxx_std_20)
target_include_directories(
  trac_ik PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                 $<INSTALL_INTERFACE:include>)
set_target_properties(
  trac_ik PROPERTIES VERSION ${${PROJECT_NAME}_VERSION}
                     SOVERSION ${CPACK_PACKAGE_VERSION_MAJOR})

pybind11_add_module(trac_ik_py src/trac_ik_py.cpp)
target_link_libraries(trac_ik_py PRIVATE trac_ik)
target_compile_features(trac_ik_py PRIVATE cxx_std_20)
pybind11_extension(trac_ik_py)
pybind11_strip(trac_ik_py)

install(TARGETS trac_ik_py LIBRARY DESTINATION ${PROJECT_NAME}
)# ${SKBUILD_PROJECT_NAME})

include(GNUInstallDirs)

install(
  TARGETS trac_ik
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
