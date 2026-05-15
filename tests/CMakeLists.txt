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
pscomp_add_test(test_middle_product       correctness/test_middle_product.cpp)
pscomp_add_test(test_dual_extraction      correctness/test_dual_extraction.cpp)
pscomp_add_test(test_compose_kl_tellegen  correctness/test_compose_kl_tellegen.cpp)
pscomp_add_test(test_compose_kl_tellegen_xy correctness/test_compose_kl_tellegen_xy.cpp)
pscomp_add_test(test_bi_convolution       correctness/test_bi_convolution.cpp)
pscomp_add_test(test_fft_cross_algorithm  precision/test_fft_cross_algorithm.cpp)
