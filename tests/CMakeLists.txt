include(FetchContent)
FetchContent_Declare(
  GTest
  URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
      SYSTEM)
FetchContent_MakeAvailable(GTest)

# Keep vendored gtest TUs out of compile_commands.json; clangd infers header
# flags from the nearest database entry, and a gtest entry lacks this project's
# include paths.
set_target_properties(gtest gtest_main gmock gmock_main
                      PROPERTIES EXPORT_COMPILE_COMMANDS OFF)

include(GoogleTest)

function(hs293go_add_test test_name)
  add_executable(${test_name} ${ARGN})
  target_link_libraries(${test_name} PRIVATE hs293go::math GTest::gtest_main
                                             GTest::gmock)
  gtest_discover_tests(${test_name})
endfunction()

hs293go_add_test(test_math src/test_math.cpp)
hs293go_add_test(test_frame_conversions src/test_frame_conversions.cpp)
hs293go_add_test(test_rotation src/test_rotation.cpp)
hs293go_add_test(test_geo src/test_geo.cpp)
hs293go_add_test(test_rigid_transform src/test_rigid_transform.cpp)
