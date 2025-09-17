link_libraries(moptim gtest gtest_main)

add_executable(nist
 chwirut1.cc
#  kirby2.cc
#  nelson.cc
 )
gtest_discover_tests(nist)
