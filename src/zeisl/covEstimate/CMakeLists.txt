CMAKE_MINIMUM_REQUIRED(VERSION 3.17.0)

project(siftCov)
find_package(OpenCV REQUIRED)
find_package(openMVG REQUIRED)

set(SOURCES 
	src/tets.cpp
	src/fileIO.cpp
)

set(HEADERS 
	include/covArgEvaluator.h
	include/definitions.h
	include/fileIO.h
)

include_directories(${OPENMVG_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
link_directories(${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable(${PROJECT_NAME} tets.cpp)

target_link_libraries(siftCov ${OPENMVG_LIBRARIES})
target_link_libraries(siftCov ${OpenCV_LIBS}) 

