add_subdirectory(googletest)
add_executable(test_tensor ../onlineml/parameter_test.cc)
add_test(test_tensor ../onlineml/parameter_test.cc)

target_link_libraries(test_tensor onlineml gtest_main)
