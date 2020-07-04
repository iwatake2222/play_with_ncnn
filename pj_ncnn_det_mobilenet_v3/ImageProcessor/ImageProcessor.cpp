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
#define MODEL_WIDTH 300
#define MODEL_HEIGHT 300
#define MODEL_CHANNEL ncnn::Mat::PIXEL_BGR
#define NCNN_CPU_NUM 4

/* Types */
typedef struct {
	double x;
	double y;
	double w;
	double h;
	int classId;
	std::string classIdName;
	double score;
} BBox;


/*** Global variables ***/
static ncnn::Net *s_net = NULL;
static std::vector<std::string> s_labels;

/*** Functions ***/
static void getBBox(std::vector<BBox> &bboxList, const ncnn::Mat &ncnnOut, const double threshold, const int imageWidth = 0, const int imageHeight = 0)
{
	for (int i = 0; i < ncnnOut.h; i++) {
		const float* values = ncnnOut.row(i);
		BBox bbox;
		bbox.classId = (int)values[0];
		bbox.score = values[1];
		if (bbox.score < threshold) continue;
		bbox.x = std::max<float>(values[2], 0.0f);
		bbox.y = std::max<float>(values[3], 0.0f);
		bbox.w = std::min<float>(values[4], 1.0f) - values[2];
		bbox.h = std::min<float>(values[5], 1.0f) - values[3];
		if (imageWidth != 0) {
			bbox.x *= imageWidth;
			bbox.y *= imageHeight;
			bbox.w *= imageWidth;
			bbox.h *= imageHeight;
		}
		bboxList.push_back(bbox);
	}
}

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
	/* Load ncnn model */
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
	float mean[3] = { 123.675f, 116.28f, 103.53f };
	float norm[3] = { 1.0f, 1.0f, 1.0f };
	ncnnMat.substract_mean_normalize(mean, norm);

	/*** Inference ***/
	ncnn::Extractor ex = s_net->create_extractor();
	ex.set_light_mode(true);
	ex.set_num_threads(NCNN_CPU_NUM);
	CHECK(ex.input("input", ncnnMat) == 0);

	/*** Run inference ***/
	ncnn::Mat ncnnOut;
	CHECK(ex.extract("detection_out", ncnnOut) == 0);

	/*** PostProcess ***/
	/* Retrieve results */
	std::vector<BBox> bboxList;
	getBBox(bboxList, ncnnOut, 0.2, mat->cols, mat->rows);
	
	/* Display bbox */
	for (int i = 0; i < (int)bboxList.size(); i++) {
		const BBox bbox = bboxList[i];
		cv::rectangle(*mat, cv::Rect((int)bbox.x, (int)bbox.y, (int)bbox.w, (int)bbox.h), cv::Scalar(255, 255, 0), 3);
		cv::putText(*mat, s_labels[bbox.classId], cv::Point((int)bbox.x, (int)bbox.y + 10), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 0), 3);
		cv::putText(*mat, s_labels[bbox.classId], cv::Point((int)bbox.x, (int)bbox.y + 10), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1);
	}

	/* Contain the results */
	outputParam->resultNum = (int)bboxList.size();
	if (outputParam->resultNum > NUM_MAX_RESULT) outputParam->resultNum = NUM_MAX_RESULT;
	for (int i = 0; i < outputParam->resultNum; i++) {
		const BBox bbox = bboxList[i];
		outputParam->RESULTS[i].classId = bbox.classId;
		snprintf(outputParam->RESULTS[i].label, sizeof(outputParam->RESULTS[i].label), "%s", s_labels[bbox.classId].c_str());
		outputParam->RESULTS[i].score = bbox.score;
		outputParam->RESULTS[i].x = (int)bbox.x;
		outputParam->RESULTS[i].y = (int)bbox.y;
		outputParam->RESULTS[i].width = (int)bbox.w;
		outputParam->RESULTS[i].height = (int)bbox.h;
	}
	
	return 0;
}


int ImageProcessor_finalize(void)
{
	delete s_net;
	return 0;
}
