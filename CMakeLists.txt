cmake_minimum_required(VERSION 3.17)
project(sfm_uncertainty)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
find_package(OpenMVG REQUIRED CONFIG HINTS /usr/local/lib/openMVG/cmake)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    /usr/local/include
    /usr/local/include/openMVG_dependencies
    /usr/include/libpng16
    src/zeisl/covEstimate/include
    src/zeisl/covEstimate/covEstimate/src
)

add_executable(cov_test
    src/zeisl/covEstimate/covEstimate/src/covEstimator.cpp
    src/zeisl/covEstimate/covEstimate/src/tets.cpp
    src/zeisl/covEstimate/covEstimate/src/fileIO.cpp
)

target_link_libraries(cov_test
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
    openMVG_features
    openMVG_image
    openMVG_sfm
    openMVG_matching
    openMVG_system
    openMVG_stlplus
    openMVG_numeric
    /usr/lib/x86_64-linux-gnu/libjpeg.so
    /usr/lib/x86_64-linux-gnu/libpng16.so
    /usr/lib/x86_64-linux-gnu/libtiff.so
)

add_executable(sfm_cov
    src/main.cpp
    src/zeisl/covEstimate/covEstimate/src/covEstimator.cpp
)

target_link_libraries(sfm_cov
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
    openMVG_features
    openMVG_image
    openMVG_sfm
    openMVG_matching
    openMVG_system
    openMVG_stlplus
    openMVG_numeric
    /usr/lib/x86_64-linux-gnu/libjpeg.so
    /usr/lib/x86_64-linux-gnu/libpng16.so
    /usr/lib/x86_64-linux-gnu/libtiff.so
)

add_executable(ba_ceres
    src/ba_ceres.cpp
)

target_link_libraries(ba_ceres
    Ceres::ceres
    openMVG_features
    openMVG_sfm
    openMVG_system
    openMVG_stlplus
    openMVG_numeric
)
