cmake_minimum_required(VERSION 3.10)
project(st CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out)

include_directories(${CMAKE_SOURCE_DIR}/.. ${CMAKE_SOURCE_DIR}/../include ${CMAKE_SOURCE_DIR}/stubs)

option(USE_BUGGY_IMPL "Use buggy implementation to test error detection" OFF)

set(CPP_SOURCES
    st_main.cpp
    st_worker.cpp
    stubs/shgemm_stub.cpp
    stubs/sbgemm_stub.cpp
    stubs/hgemm_stub.cpp
    stubs/bgemm_stub.cpp
    ../include/test_util.cpp
)

set(C_SOURCES
    ../openblas.c
)

if(USE_BUGGY_IMPL)
    message(STATUS "Using BUGGY implementation for testing")
    list(APPEND C_SOURCES ../unigemm_buggy.c)
else()
    message(STATUS "Using correct implementation")
    list(APPEND C_SOURCES ../unigemm.c)
endif()

set(SOURCES ${CPP_SOURCES} ${C_SOURCES})

add_executable(st ${SOURCES})

set_source_files_properties(${C_SOURCES} PROPERTIES LANGUAGE C)
target_compile_features(st PRIVATE c_std_99)

target_link_libraries(st PRIVATE pthread m)

if(APPLE)
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY /opt/homebrew/opt/libomp/lib/libomp.a)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_omp_INCLUDE_DIR "/opt/homebrew/opt/libomp/include")
    set(OpenMP_C_FOUND TRUE)
    set(OpenMP_CXX_FOUND TRUE)
endif()

find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(st PRIVATE ${OpenMP_omp_LIBRARY})
    set_source_files_properties(${C_SOURCES} PROPERTIES
        COMPILE_FLAGS "${OpenMP_C_FLAGS}"
    )
    message(STATUS "OpenMP support found")
endif()

install(TARGETS st DESTINATION bin)
