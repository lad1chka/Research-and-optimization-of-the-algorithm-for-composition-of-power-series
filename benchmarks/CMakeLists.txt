function(pscomp_add_bench name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE pscomp benchmark::benchmark)
endfunction()

pscomp_add_bench(bench_compose_time   bench_compose_time.cpp)
pscomp_add_bench(bench_compose_memory bench_compose_memory.cpp)
