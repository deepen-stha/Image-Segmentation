set(CMAKE_CXX_STANDARD 17)
cmake_minimum_required(VERSION 3.10)
project(imagesegmentation VERSION 0.1.0)

include(CTest)
enable_testing()

set(OpenCV_DIR "C:/Users/admin/opencv/build") # path for installed opencv libraries

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(imagesegmentation main.cpp)

target_link_libraries( imagesegmentation ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)