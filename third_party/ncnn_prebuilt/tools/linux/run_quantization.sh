#!/bin/bash


function convert () {
	ORIGINAL_PARAM=$1.param
	ORIGINAL_BIN=$1.bin
	OPTIMIZED_PARAM=$1_opt.param
	OPTIMIZED_BIN=$1_opt.bin
	TABLE=$1_opt.table
	QUANTIZED_PARAM=$1_int8.param
	QUANTIZED_BIN=$1_int8.bin
	./ncnnoptimize $ORIGINAL_PARAM $ORIGINAL_BIN $OPTIMIZED_PARAM $OPTIMIZED_BIN 0
	./ncnn2table --param=$OPTIMIZED_PARAM --bin=$OPTIMIZED_BIN --images=images --output=$TABLE --mean=104,117,123 --norm=0.017,0.017,0.017 --size=224,224 --thread=2
	./ncnn2int8 $OPTIMIZED_PARAM $OPTIMIZED_BIN $QUANTIZED_PARAM $QUANTIZED_BIN $TABLE

	mkdir temp
	# mv $ORIGINAL_PARAM temp/.
	# mv $ORIGINAL_BIN temp/.
	mv $OPTIMIZED_PARAM temp/.
	mv $OPTIMIZED_BIN temp/.
	mv $TABLE temp/.
	mv $QUANTIZED_PARAM temp/.
	mv $QUANTIZED_BIN temp/.
}

convert mobilenetv2-1.0
