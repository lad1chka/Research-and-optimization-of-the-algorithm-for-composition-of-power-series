include(GoogleTest)

function(pscomp_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE pscomp GTest::gtest GTest::gtest_main)
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/precision
    )
    gtest_discover_tests(${name})
endfunction()

pscomp_add_test(test_known_series         correctness/test_known_series.cpp)
pscomp_add_test(test_cross_algorithms_ntt correctness/test_cross_algorithms_ntt.cpp)
pscomp_add_test(test_random_ntt_large     correctness/test_random_ntt_large.cpp)
pscomp_add_test(test_fft_cross_algorithm  precision/test_fft_cross_algorithm.cpp)
