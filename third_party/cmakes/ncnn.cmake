set(USE_PREBUILT_NCNN on CACHE BOOL "Use Prebuilt ncnn? [on/off]")
if(USE_PREBUILT_NCNN)
	if(DEFINED  ANDROID_ABI)
		set(NCNN_LIB ${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/android/${ANDROID_ABI}/libncnn.a)
		set(NCNN_INC ${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/android/include/ncnn)
	elseif(MSVC_VERSION)
		if((MSVC_VERSION GREATER_EQUAL 1910) AND (MSVC_VERSION LESS 1920))
			set(NCNN_LIB
				$<$<CONFIG:Debug>:${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/x64_windows/lib/ncnnd.lib>
				$<$<CONFIG:RelWithDebInfo>:${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/x64_windows/lib/ncnnRelWithDebInfo.lib>
				$<$<CONFIG:Release>:${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/x64_windows/lib/ncnn.lib>
				$<$<CONFIG:MinSizeRel>:${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/x64_windows/lib/ncnnMinSizeRel.lib>
			)
			set(NCNN_INC ${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/x64_windows/inc/)
		else()
			message(FATAL_ERROR "[ncnn] unsupported MSVC version")
		endif()
	else()
		set(NCNN_LIB
			# $<$<STREQUAL:${BUILD_SYSTEM},x64_windows>:${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/x64_windows/lib/ncnn.lib>
			$<$<STREQUAL:${BUILD_SYSTEM},x64_linux>:${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/x64_linux/lib/libncnn.a>
			$<$<STREQUAL:${BUILD_SYSTEM},armv7>:${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/armv7/lib/libncnn.a>
			$<$<STREQUAL:${BUILD_SYSTEM},aarch64>:${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/aarch64/lib/libncnn.a>
		)
		set(NCNN_INC
			$<$<STREQUAL:${BUILD_SYSTEM},x64_linux>:${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/x64_linux/inc/>
			$<$<STREQUAL:${BUILD_SYSTEM},armv7>:${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/armv7/inc/>
			$<$<STREQUAL:${BUILD_SYSTEM},aarch64>:${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/aarch64/inc/>
		)
	endif()
else()
	add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../ncnn/src ncnn)
	set(NCNN_LIB ncnn)
	set(NCNN_INC ${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/x64_linux/inc/)	# todo
endif()

