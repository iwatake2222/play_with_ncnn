cmake_minimum_required(VERSION 3.0)

set(ProjectName "main")

# Switch build target
set(ARCH_TYPE x64 CACHE STRING "Build target? [x64, armv7, aarch64]")
if(${ARCH_TYPE} STREQUAL armv7)
	message("[BUILD] armv7")
elseif(${ARCH_TYPE} STREQUAL aarch64)
	message("[BUILD] aarch64")
else()
	if(WIN32)
		message("[BUILD] windows (x64)")
		set(ARCH_TYPE x64_windows)
	else()
		message("[BUILD] linux (x64)")
		set(ARCH_TYPE x64_linux)
	endif()
endif()

# Compile option
if(NOT WIN32)
	set(CMAKE_CXX_FLAGS "-Wall -std=c++11")
	set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
	set(CMAKE_CXX_FLAGS_DEBUG "-g3")
	if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
		message(STATUS "No build type selected, default to Release")
		set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Debug)" FORCE)
	endif()
endif()

# Create project
project(${ProjectName})
add_executable(${ProjectName} Main.cpp)


# For ncnn
set(USE_PREBUILT_NCNN on CACHE BOOL "Use Prebuilt ncnn? [on/off]")
if(USE_PREBUILT_NCNN)
	if(MSVC_VERSION)
		if((MSVC_VERSION GREATER_EQUAL 1910) AND (MSVC_VERSION LESS 1920))
			target_link_libraries(${ProjectName}
				$<$<CONFIG:Debug>:${CMAKE_SOURCE_DIR}/../third_party/ncnn_prebuilt/x64_windows/lib/ncnnd.lib>
				$<$<CONFIG:RelWithDebInfo>:${CMAKE_SOURCE_DIR}/../third_party/ncnn_prebuilt/x64_windows/lib/ncnnRelWithDebInfo.lib>
				$<$<CONFIG:Release>:${CMAKE_SOURCE_DIR}/../third_party/ncnn_prebuilt/x64_windows/lib/ncnn.lib>
				$<$<CONFIG:MinSizeRel>:${CMAKE_SOURCE_DIR}/../third_party/ncnn_prebuilt/x64_windows/lib/ncnnMinSizeRel.lib>
			)
		else()
			message(FATAL_ERROR "unsupported MSVC version")
		endif()
	else()
		target_link_libraries(${ProjectName}
			# $<$<STREQUAL:${ARCH_TYPE},x64_windows>:${CMAKE_SOURCE_DIR}/../third_party/ncnn_prebuilt/x64_windows/lib/ncnn.lib>
			$<$<STREQUAL:${ARCH_TYPE},x64_linux>:${CMAKE_SOURCE_DIR}/../third_party/ncnn_prebuilt/x64_linux/lib/libncnn.a>
			$<$<STREQUAL:${ARCH_TYPE},armv7>:${CMAKE_SOURCE_DIR}/../third_party/ncnn_prebuilt/armv7/lib/libncnn.a>
			$<$<STREQUAL:${ARCH_TYPE},aarch64>:${CMAKE_SOURCE_DIR}/../third_party/ncnn_prebuilt/aarch64/lib/libncnn.a>
		)
	endif()
	target_include_directories(${ProjectName} PUBLIC
		$<$<STREQUAL:${ARCH_TYPE},x64_windows>:${CMAKE_SOURCE_DIR}/../third_party/ncnn_prebuilt/x64_windows/inc/>
		$<$<STREQUAL:${ARCH_TYPE},x64_linux>:${CMAKE_SOURCE_DIR}/../third_party/ncnn_prebuilt/x64_linux/inc/>
		$<$<STREQUAL:${ARCH_TYPE},armv7>:${CMAKE_SOURCE_DIR}/../third_party/ncnn_prebuilt/armv7/inc/>
		$<$<STREQUAL:${ARCH_TYPE},aarch64>:${CMAKE_SOURCE_DIR}/../third_party/ncnn_prebuilt/aarch64/inc/>
	)
	# target_include_directories(${ProjectName} PUBLIC ${CMAKE_SOURCE_DIR}/../third_party/ncnn/src)
else()
	set(NCNN_DIR ${CMAKE_SOURCE_DIR}/../third_party/ncnn/src)
	add_subdirectory(${NCNN_DIR} ncnn)
	target_link_libraries(${PROJECT_NAME} ncnn)
	target_include_directories(${PROJECT_NAME} PUBLIC ${NCNN_DIR}/src)
endif()

# For OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
	target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
	target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
endif()

# For OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Copy resouce
file(COPY ${PROJECT_SOURCE_DIR}/model DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/resource/)
file(COPY ${PROJECT_SOURCE_DIR}/../resource/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/resource/)
add_definitions(-DRESOURCE_DIR="${CMAKE_CURRENT_BINARY_DIR}/resource/")
