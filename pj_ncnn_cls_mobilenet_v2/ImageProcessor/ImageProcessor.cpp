/*** Include ***/
/* for general */
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
#include <chrono>


/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for ncnn */
#include "net.h"

#include "ImageProcessor.h"

/*** Macro ***/
#if defined(ANDROID) || defined(__ANDROID__)
#include <android/log.h>
#define TAG "MyApp_NDK"
#define PRINT(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#else
#define PRINT(...) printf(__VA_ARGS__)
#endif

#define CHECK(x)                              \
  if (!(x)) {                                                \
	PRINT("Error at %s:%d\n", __FILE__, __LINE__); \
	exit(1);                                                 \
  }

/* Settings */
#define MODEL_WIDTH 224
#define MODEL_HEIGHT 224
#define MODEL_CHANNEL ncnn::Mat::PIXEL_BGR
#define NCNN_CPU_NUM 4

/*** Global variables ***/
static ncnn::Net *s_net;
static std::vector<std::string> s_labels;

/*** Functions ***/
static void readLabel(const char* filename, std::vector<std::string> & labels)
{
	std::ifstream ifs(filename);
	if (ifs.fail()) {
		PRINT("failed to read %s\n", filename);
		return;
	}
	std::string str;
	while (getline(ifs, str)) {
		labels.push_back(str);
	}
}


int ImageProcessor_initialize(const char *modelFilename, INPUT_PARAM *inputParam)
{
	/*** Load ncnn model ***/
	s_net = new ncnn::Net();
	CHECK(s_net != NULL);
	CHECK(s_net->load_param((std::string(modelFilename) + ".param").c_str()) == 0);
	CHECK(s_net->load_model((std::string(modelFilename) + ".bin").c_str()) == 0);

	/* read label */
	readLabel(inputParam->labelFilename, s_labels);

	return 0;
}


int ImageProcessor_process(cv::Mat *mat, OUTPUT_PARAM *outputParam)
{
	/*** PreProcess for ncnn ***/
	ncnn::Mat ncnnMat = ncnn::Mat::from_pixels_resize(mat->data, MODEL_CHANNEL, mat->cols, mat->rows, MODEL_WIDTH, MODEL_HEIGHT);
	float mean[3] = { 128.f, 128.f, 128.f };
	float norm[3] = { 1 / 128.f, 1 / 128.f, 1 / 128.f };
	ncnnMat.substract_mean_normalize(mean, norm);

	/*** Inference ***/
	ncnn::Extractor ex = s_net->create_extractor();
	ex.set_light_mode(true);
	ex.set_num_threads(NCNN_CPU_NUM);
	CHECK(ex.input("data", ncnnMat) == 0);

	/*** Run inference ***/
	ncnn::Mat ncnnOut;
	CHECK(ex.extract("mobilenetv20_output_flatten0_reshape0", ncnnOut) == 0);

	/*** PostProcess ***/
	/* Retrieve results */
	std::vector<float> results;
	int outputNum = ncnnOut.w;
	results.resize(outputNum);
	for (int i = 0; i < outputNum; i++) {
		results[i] = ((float*)ncnnOut.data)[i];
	}
	
	/* Find the max score */
	int maxIndex = (int)(std::max_element(results.begin(), results.end()) - results.begin());
	float maxScore = *std::max_element(results.begin(), results.end());
	PRINT("Result = %s (%d) (%.3f)\n", s_labels[maxIndex].c_str(), maxIndex, maxScore);

	/* Draw the result */
	std::string resultStr;
	resultStr = "Result:" + s_labels[maxIndex] + " (score = " + std::to_string(maxScore) + ")";
	cv::putText(*mat, resultStr, cv::Point(100, 100), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 0), 3);
	cv::putText(*mat, resultStr, cv::Point(100, 100), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1);

	/* Contain the results */
	outputParam->classId = maxIndex;
	snprintf(outputParam->label, sizeof(outputParam->label), s_labels[maxIndex].c_str());
	outputParam->score = maxScore;
	
	return 0;
}


int ImageProcessor_finalize(void)
{
	delete s_net;
	return 0;
}
