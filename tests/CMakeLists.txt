include(FetchContent)
FetchContent_Declare(
  GTest
  URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
)
FetchContent_MakeAvailable(GTest)

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
