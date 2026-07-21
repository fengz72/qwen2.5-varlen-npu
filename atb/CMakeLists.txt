cmake_minimum_required(VERSION 3.14)
project(acl_dynamic_shape_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ASCEND_HOME)
    if(DEFINED ENV{ASCEND_HOME})
        set(ASCEND_HOME $ENV{ASCEND_HOME})
    else()
        set(ASCEND_HOME /usr/local/Ascend/ascend-toolkit/latest)
    endif()
endif()

message(STATUS "ASCEND_HOME: ${ASCEND_HOME}")

set(ACL_INC_DIR "${ASCEND_HOME}/include")
set(ACL_LIB_DIR "${ASCEND_HOME}/lib64")

include_directories(${ACL_INC_DIR})
link_directories(${ACL_LIB_DIR})

add_executable(acl_infer main.cpp)

target_link_libraries(acl_infer
    ascendcl
    acl_op_compiler
    stdc++
)

add_executable(bench_throughput bench_throughput.cpp)

target_link_libraries(bench_throughput
    ascendcl
    stdc++
    pthread
)

add_executable(bench_latency bench_latency.cpp)

target_link_libraries(bench_latency
    ascendcl
    stdc++
    pthread
)

install(TARGETS acl_infer bench_throughput bench_latency DESTINATION bin)
