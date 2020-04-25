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
			message(FATAL_ERROR "[ncnn] unsupported MSVC version")
		endif()
	else()
		target_link_libraries(${ProjectName}
			# $<$<STREQUAL:${BUILD_SYSTEM},x64_windows>:${CMAKE_SOURCE_DIR}/../third_party/ncnn_prebuilt/x64_windows/lib/ncnn.lib>
			$<$<STREQUAL:${BUILD_SYSTEM},x64_linux>:${CMAKE_SOURCE_DIR}/../third_party/ncnn_prebuilt/x64_linux/lib/libncnn.a>
			$<$<STREQUAL:${BUILD_SYSTEM},armv7>:${CMAKE_SOURCE_DIR}/../third_party/ncnn_prebuilt/armv7/lib/libncnn.a>
			$<$<STREQUAL:${BUILD_SYSTEM},aarch64>:${CMAKE_SOURCE_DIR}/../third_party/ncnn_prebuilt/aarch64/lib/libncnn.a>
		)
	endif()
	target_include_directories(${ProjectName} PUBLIC
		$<$<STREQUAL:${BUILD_SYSTEM},x64_windows>:${CMAKE_SOURCE_DIR}/../third_party/ncnn_prebuilt/x64_windows/inc/>
		$<$<STREQUAL:${BUILD_SYSTEM},x64_linux>:${CMAKE_SOURCE_DIR}/../third_party/ncnn_prebuilt/x64_linux/inc/>
		$<$<STREQUAL:${BUILD_SYSTEM},armv7>:${CMAKE_SOURCE_DIR}/../third_party/ncnn_prebuilt/armv7/inc/>
		$<$<STREQUAL:${BUILD_SYSTEM},aarch64>:${CMAKE_SOURCE_DIR}/../third_party/ncnn_prebuilt/aarch64/inc/>
	)
	# target_include_directories(${ProjectName} PUBLIC ${CMAKE_SOURCE_DIR}/../third_party/ncnn/src)
else()
	set(NCNN_DIR ${CMAKE_SOURCE_DIR}/../third_party/ncnn/src)
	add_subdirectory(${NCNN_DIR} ncnn)
	target_link_libraries(${PROJECT_NAME} ncnn)
	target_include_directories(${PROJECT_NAME} PUBLIC ${NCNN_DIR}/src)
endif()

