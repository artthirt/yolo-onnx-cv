cmake_minimum_required(VERSION 3.20)

set(PROJECT_NAME yolo-onnx-cv)

project(${PROJECT_NAME} VERSION 1.0 LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)
#find_package(OpenCV REQUIRED PATHS D:/devs/opencv/build/x64/vc16/lib)

find_path(AVCODEC_INC NAMES libavcodec/avcodec.h REQUIRED PATHS 3rd/ffmpeg/include)
find_library(AVCODEC_LIB NAMES avcodec REQUIRED PATHS 3rd/ffmpeg/bin)
find_library(AVFORMAT_LIB NAMES avformat REQUIRED PATHS 3rd/ffmpeg/bin)
find_library(AVUTIL_LIB NAMES avutil REQUIRED PATHS 3rd/ffmpeg/bin)

message("${OpenCV_INCLUDE_DIRS}")
message("${OpenCV_LIBRARIES}")

set(SRC
    main.cpp
    AV/AVDecoderVideo.cpp
    AV/AVDecoderVideo.h
)

add_executable(${PROJECT_NAME} ${SRC})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    AV
    ${AVCODEC_INC}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBRARIES}
    ${AVCODEC_LIB}
    ${AVFORMAT_LIB}
    ${AVUTIL_LIB}
)
