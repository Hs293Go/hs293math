cmake_minimum_required(VERSION 3.28.0)
project(
  hs293go_math
  VERSION 0.1.0
  DESCRIPTION "Math library for Hs293Go"
  LANGUAGES CXX)

find_package(Eigen3 REQUIRED)

add_library(hs293go_math INTERFACE)
target_sources(
  hs293go_math
  INTERFACE FILE_SET
            HEADERS
            BASE_DIRS
            include
            FILES
            include/hs293go/math.hpp
            include/hs293go/rotation.hpp
            include/hs293go/frame_conversions.hpp
            include/hs293go/rigid_transform.hpp
            include/hs293go/eigen_utils.hpp
            include/hs293go/geo.hpp
            include/hs293go/nn.hpp)
target_compile_features(hs293go_math INTERFACE cxx_std_20)
target_link_libraries(hs293go_math INTERFACE Eigen3::Eigen)
set_target_properties(hs293go_math PROPERTIES EXPORT_NAME math)
add_library(hs293go::math ALIAS hs293go_math)

if(PROJECT_IS_TOP_LEVEL)
  include(CTest)
  if(BUILD_TESTING)
    add_subdirectory(tests)
  endif()
endif()

# Default to installing only when this is the top-level project, so consumers
# pulling it in via add_subdirectory()/FetchContent don't install it too.
option(HS293GO_MATH_INSTALL "Enable installation of hs293go_math"
       ${PROJECT_IS_TOP_LEVEL})
if(HS293GO_MATH_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  set(HS293GO_MATH_INSTALL_CMAKEDIR
      "${CMAKE_INSTALL_LIBDIR}/cmake/hs293go_math"
      CACHE STRING "Install location for hs293go_math CMake package files")

  install(
    TARGETS hs293go_math
    EXPORT hs293go_mathTargets
    FILE_SET HEADERS
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT dev)

  install(
    EXPORT hs293go_mathTargets
    FILE hs293go_mathTargets.cmake
    NAMESPACE hs293go::
    DESTINATION "${HS293GO_MATH_INSTALL_CMAKEDIR}"
    COMPONENT dev)

  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hs293go_mathConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/hs293go_mathConfig.cmake"
    INSTALL_DESTINATION "${HS293GO_MATH_INSTALL_CMAKEDIR}")

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/hs293go_mathConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion ARCH_INDEPENDENT)

  install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/hs293go_mathConfig.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/hs293go_mathConfigVersion.cmake"
    DESTINATION "${HS293GO_MATH_INSTALL_CMAKEDIR}"
    COMPONENT dev)
endif()
