@echo off


call :convert mobilenetv2-1.0
exit /b

:convert
	set ORIGINAL_PARAM=%1.param
	set ORIGINAL_BIN=%1.bin
	set OPTIMIZED_PARAM=%1_opt.param
	set OPTIMIZED_BIN=%1_opt.bin
	set TABLE=%1_opt.table
	set QUANTIZED_PARAM=%1_int8.param
	set QUANTIZED_BIN=%1_int8.bin
	ncnnoptimize.exe %ORIGINAL_PARAM% %ORIGINAL_BIN% %OPTIMIZED_PARAM% %OPTIMIZED_BIN% 0
	ncnn2table.exe --param=%OPTIMIZED_PARAM% --bin=%OPTIMIZED_BIN% --images=images --output=%TABLE% --mean=104,117,123 --norm=0.017,0.017,0.017 --size=224,224 --thread=2
	ncnn2int8.exe %OPTIMIZED_PARAM% %OPTIMIZED_BIN% %QUANTIZED_PARAM% %QUANTIZED_BIN% %TABLE%

	mkdir temp
	REM move %ORIGINAL_PARAM% temp/.
	REM move %ORIGINAL_BIN% temp/.
	move %OPTIMIZED_PARAM% temp/.
	move %OPTIMIZED_BIN% temp/.
	move %TABLE% temp/.
	move %QUANTIZED_PARAM% temp/.
	move %QUANTIZED_BIN% temp/.
	exit /b

