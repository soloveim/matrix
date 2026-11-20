cmake_minimum_required(VERSION 3.14)
project(MatrixLib VERSION 1.0.0 LANGUAGES CXX)

# ---------- Allgemeine Einstellungen ----------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Standard-Buildtyp: Release (wichtig für Performance, -O3)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MATRIX_BUILD_DEMO "Demo-Programm bauen" ON)
option(MATRIX_NATIVE_ARCH "Mit -march=native optimieren (nicht portabel)" OFF)

# ---------- Header-only Bibliothek (INTERFACE-Target) ----------
add_library(matrix INTERFACE)
add_library(Matrix::Matrix ALIAS matrix)

target_include_directories(matrix INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(matrix INTERFACE cxx_std_17)

# ---------- Demo / Benchmark ----------
if(MATRIX_BUILD_DEMO)
    add_executable(matrix_demo main.cpp)
    target_link_libraries(matrix_demo PRIVATE Matrix::Matrix)

    add_executable(jacobi_demo jacobi_demo.cpp)
    target_link_libraries(jacobi_demo PRIVATE Matrix::Matrix)

    if(MSVC)
        target_compile_options(matrix_demo PRIVATE /W4)
        target_compile_options(jacobi_demo PRIVATE /W4)
    else()
        target_compile_options(matrix_demo PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(jacobi_demo PRIVATE -Wall -Wextra -Wpedantic)
        if(MATRIX_NATIVE_ARCH)
            target_compile_options(matrix_demo PRIVATE -march=native)
            target_compile_options(jacobi_demo PRIVATE -march=native)
        endif()
    endif()
endif()

# ---------- Installation ----------
include(GNUInstallDirs)

install(FILES Matrix.hpp Jacobi.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS matrix EXPORT MatrixTargets)
install(EXPORT MatrixTargets
    FILE MatrixTargets.cmake
    NAMESPACE Matrix::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Matrix
)
if(MATRIX_BUILD_DEMO)
    install(TARGETS matrix_demo jacobi_demo RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
