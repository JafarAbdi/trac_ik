cmake_minimum_required(VERSION 3.12)
project(trac_ik_lib)

# Get version from version.hpp
file(READ pixi.toml version_hpp)
if(NOT version_hpp MATCHES "version = \"([0-9]+)\\.([0-9]+)\\.([0-9]+)\"")
  message(FATAL_ERROR "Cannot get TRAC_IK_VERSION from pixi.toml")
endif()

# Use math to skip leading zeros if any.
math(EXPR CPACK_PACKAGE_VERSION_MAJOR ${CMAKE_MATCH_1})
math(EXPR CPACK_PACKAGE_VERSION_MINOR ${CMAKE_MATCH_2})
math(EXPR CPACK_PACKAGE_VERSION_PATCH ${CMAKE_MATCH_3})
# cmake-lint: disable=C0103
set(${PROJECT_NAME}_VERSION
    "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}"
)
message(STATUS "${PROJECT_NAME} version: ${${PROJECT_NAME}_VERSION}")

include(GNUInstallDirs)

# Fix issue with pixi not being able to find python in github actions
# https://github.com/HumanoidTeam/prealpha_ros2/actions/runs/13999946388/job/39203937190
# --- Find Python interpreter ---
execute_process(
  COMMAND python3 -c
          "import sysconfig; print(sysconfig.get_config_var('BINDIR'))"
  OUTPUT_VARIABLE PYTHON_BIN_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(PYTHON_EXECUTABLE "${PYTHON_BIN_DIR}/python3")

# --- Find Python include directory ---
execute_process(
  COMMAND python3 -c "import sysconfig; print(sysconfig.get_path('include'))"
  OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(PYTHON_INCLUDE_DIRS "${PYTHON_INCLUDE_DIR}")

# --- Find Python libraries ---
execute_process(
  COMMAND python3 -c "import sysconfig; print(sysconfig.get_path('platlib'))"
  OUTPUT_VARIABLE PYTHON_LIB_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(PYTHON_LIBRARIES "${PYTHON_LIB_DIR}")

message(STATUS "Python executable: ${PYTHON_EXECUTABLE}")
message(STATUS "Python include dirs: ${PYTHON_INCLUDE_DIRS}")
message(STATUS "Python libraries: ${PYTHON_LIBRARIES}")

set(Python_EXECUTABLE
    "${PYTHON_EXECUTABLE}"
    CACHE PATH "Python executable"
)
set(Python_INCLUDE_DIRS
    "${PYTHON_INCLUDE_DIRS}"
    CACHE PATH "Python include directories"
)
set(Python_LIBRARIES
    "${PYTHON_LIBRARIES}"
    CACHE PATH "Python libraries"
)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
find_package(NLopt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(urdfdom REQUIRED)

add_library(
  trac_ik SHARED src/kdl_tl.cpp src/nlopt_ik.cpp src/trac_ik.cpp
                 src/kdl_parser.cpp
)
target_link_libraries(
  trac_ik PUBLIC Eigen3::Eigen NLopt::nlopt spdlog::spdlog orocos-kdl
                 urdfdom::urdf_parser urdfdom::urdfdom_model
)
target_compile_features(trac_ik PRIVATE cxx_std_20)
target_include_directories(
  trac_ik PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                 $<INSTALL_INTERFACE:include>
)
set_target_properties(
  trac_ik PROPERTIES VERSION ${${PROJECT_NAME}_VERSION}
                     SOVERSION ${CPACK_PACKAGE_VERSION_MAJOR}
)

pybind11_add_module(trac_ik_py src/trac_ik_py.cpp)
target_link_libraries(trac_ik_py PRIVATE trac_ik)
target_compile_features(trac_ik_py PRIVATE cxx_std_20)
pybind11_extension(trac_ik_py)
pybind11_strip(trac_ik_py)

# Determine the python installation directory
set(PYTHON_SITE_PACKAGES
    ${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages
)
message(STATUS "Python site-packages directory: " ${PYTHON_SITE_PACKAGES})

install(TARGETS trac_ik_py LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
# Generate the version, config and target files into the build directory.
write_basic_package_version_file(
  ${PROJECT_NAME}-config-version.cmake
  VERSION ${PROJECT_NAME}_VERSION
  COMPATIBILITY AnyNewerVersion
)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
  TARGETS trac_ik
  COMPONENT ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)

# Use a namespace because CMake provides better diagnostics for namespaced
# imported targets.
export(
  TARGETS trac_ik
  NAMESPACE ${PROJECT_NAME}::
  FILE ${PROJECT_NAME}-targets.cmake
)

# Install version, config and target files.
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  COMPONENT ${PROJECT_NAME}
)
install(
  EXPORT ${PROJECT_NAME}-targets
  NAMESPACE ${PROJECT_NAME}::
  COMPONENT ${PROJECT_NAME}
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
