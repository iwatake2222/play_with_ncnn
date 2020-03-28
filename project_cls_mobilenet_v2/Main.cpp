#include <stdio.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "net.h"

#define MODEL_PARAM_NAME RESOURCE_DIR"/model/mobilenetv2-1.0.param"
#define MODEL_BIN_NAME   RESOURCE_DIR"/model/mobilenetv2-1.0.bin"
//#define MODEL_PARAM_NAME RESOURCE_DIR"/model/mobilenetv2-1.0_nobn.param"
//#define MODEL_BIN_NAME   RESOURCE_DIR"/model/mobilenetv2-1.0_nobn.bin"
//#define MODEL_PARAM_NAME RESOURCE_DIR"/model/mobilenetv2-1.0_int8.param"
//#define MODEL_BIN_NAME   RESOURCE_DIR"/model/mobilenetv2-1.0_int8.bin"
#define MODEL_WIDTH 224
#define MODEL_HEIGHT 224
#define MODEL_CHANNEL ncnn::Mat::PIXEL_BGR
#define NCNN_CPU_NUM 4

#define LOOP_NUM_TO_MEASURE_INFERENCE_TIME 100

int main()
{
	/*** Load ncnn model (probably, need this only once) ***/
	ncnn::Net net;
	net.load_param(MODEL_PARAM_NAME);
	net.load_model(MODEL_BIN_NAME);

	/*** Read image using OpenCV ***/
	cv::Mat image = cv::imread(RESOURCE_DIR"parrot.jpg");
	cv::imshow("Display", image);

	/*** PreProcess for ncnn ***/
	ncnn::Mat ncnnMat = ncnn::Mat::from_pixels_resize(image.data, MODEL_CHANNEL, image.cols, image.rows, MODEL_WIDTH, MODEL_HEIGHT);
	float mean[3] = { 128.f, 128.f, 128.f };
	float norm[3] = { 1 / 128.f, 1 / 128.f, 1 / 128.f };
	ncnnMat.substract_mean_normalize(mean, norm);

	/* Repeat inference */
	/* the following loop is just to measure inference time, you don't need this for the real product */
	std::vector<float> results;
	auto t0 = std::chrono::system_clock::now();
	for (int i = 0; i < LOOP_NUM_TO_MEASURE_INFERENCE_TIME; i++) {
		/*** Prepare inference ***/
		ncnn::Extractor ex = net.create_extractor();
		ex.set_light_mode(true);
		ex.set_num_threads(NCNN_CPU_NUM);
		ex.input("data", ncnnMat);

		/*** Run inference ***/
		ncnn::Mat ncnnOut;
		ex.extract("mobilenetv20_output_flatten0_reshape0", ncnnOut);

		/*** Retrieve results ***/
		int outputNum = ncnnOut.w;
		// std::vector<float> results(outputNum);
		results.resize(outputNum);
		for (int i = 0; i < outputNum; i++) {
			results[i] = ((float*)ncnnOut.data)[i];
		}
	}
	auto t1 = std::chrono::system_clock::now();

	/*** PostProcess ***/
	int maxIndex = std::max_element(results.begin(), results.end()) - results.begin();
	float maxScore = *std::max_element(results.begin(), results.end());
	printf("Result = %d (%.3f)\n", maxIndex, maxScore);
	
	double inferenceTime = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
	printf("Inference time: %.2lf [msec]\n", inferenceTime / LOOP_NUM_TO_MEASURE_INFERENCE_TIME);

	cv::waitKey(0);
	return 0;
}
