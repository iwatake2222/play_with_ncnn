cmake_minimum_required(VERSION 2.8.10)

# Create executable file
set(PROJECT_NAME play_with_ncnn_mobilenet_v2)
project(${PROJECT_NAME})
add_executable(${PROJECT_NAME}
	Main.cpp
)

# For NCNN
set(NCNN_DIR ${PROJECT_SOURCE_DIR}/../ncnn)
add_subdirectory(${NCNN_DIR} ncnn)
target_link_libraries(${PROJECT_NAME} ncnn)
target_include_directories(${PROJECT_NAME} PUBLIC ${NCNN_DIR}/src)


# For OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
	target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
	target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
endif()

# Copy resouce
file(COPY ${PROJECT_SOURCE_DIR}/model DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/resource/)
file(COPY ${PROJECT_SOURCE_DIR}/../resource/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/resource/)
add_definitions(-DRESOURCE_DIR="${CMAKE_CURRENT_BINARY_DIR}/resource/")
