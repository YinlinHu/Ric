CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
SET(PROJ_NAME RIC)
PROJECT (${PROJ_NAME})
# AUX_SOURCE_DIRECTORY(. SRC_LIST)
# SET(SRC_LIST testmain.cpp ./SLICSuperpixels/SLIC.cpp)
# MESSAGE(STATUS "source list: " ${SRC_LIST})

ADD_DEFINITIONS(-DWITH_SSE)

IF(MSVC)
INCLUDE_DIRECTORIES(
./include
C:/OpenCV/include
C:/OpenCV/include/opencv
C:/OpenCV/include/opencv2
)

LINK_DIRECTORIES(
C:/OpenCV/lib/x86/Debug
C:/OpenCV/lib/x86/Release
)
ELSE(MSVC)
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(
    ./include
    ${OpenCV_Include_Dirs}
    )
ENDIF(MSVC)

ADD_EXECUTABLE(${PROJ_NAME} main.cpp RIC.cpp SLIC.cpp)

IF(MSVC)
TARGET_LINK_LIBRARIES(${PROJ_NAME}
debug opencv_core310d.lib
debug opencv_highgui310d.lib
debug opencv_imgcodecs310d.lib
debug opencv_imgproc310d.lib
debug opencv_ximgproc310d.lib
debug opencv_optflow310d.lib
debug opencv_calib3d310d.lib
optimized opencv_core310.lib
optimized opencv_highgui310.lib
optimized opencv_imgcodecs310.lib
optimized opencv_imgproc310.lib
optimized opencv_ximgproc310.lib
optimized opencv_optflow310.lib
optimized opencv_calib3d310.lib)
ELSE(MSVC)
target_link_libraries(${PROJ_NAME} ${OpenCV_LIBS})
ENDIF(MSVC)
