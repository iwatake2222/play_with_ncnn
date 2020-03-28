#include <stdio.h>
#include <fstream> 
#include <vector>
#include <algorithm>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "net.h"

#define LABEL_NAME       RESOURCE_DIR"/model/label_PASCAL_VOC2012.txt"
#define MODEL_PARAM_NAME RESOURCE_DIR"/model/mobilenetv3_ssdlite_voc.param"
#define MODEL_BIN_NAME   RESOURCE_DIR"/model/mobilenetv3_ssdlite_voc.bin"
//#define MODEL_PARAM_NAME RESOURCE_DIR"/model/mobilenetv2-1.0_nobn.param"
//#define MODEL_BIN_NAME   RESOURCE_DIR"/model/mobilenetv2-1.0_nobn.bin"
//#define MODEL_PARAM_NAME RESOURCE_DIR"/model/mobilenetv2-1.0_int8.param"
//#define MODEL_BIN_NAME   RESOURCE_DIR"/model/mobilenetv2-1.0_int8.bin"
#define MODEL_WIDTH 300
#define MODEL_HEIGHT 300
#define MODEL_CHANNEL ncnn::Mat::PIXEL_BGR2RGB
#define NCNN_CPU_NUM 4

#define LOOP_NUM_TO_MEASURE_INFERENCE_TIME 10

typedef struct {
	double x;
	double y;
	double w;
	double h;
	int classId;
	std::string classIdName;
	double score;
} BBox;
static void getBBox(std::vector<BBox> &bboxList, const ncnn::Mat &ncnnOut, const double threshold, const int imageWidth = 0, const int imageHeight = 0)
{
	for (int i = 0; i < ncnnOut.h; i++) {
		const float* values = ncnnOut.row(i);
		BBox bbox;
		bbox.classId = values[0];
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
		printf("failed to read %s\n", filename);
		return;
	}
	std::string str;
	while (getline(ifs, str)) {
		labels.push_back(str);
	}
}

int main()
{
	/*** Initialize ***/
	/* read label */
	std::vector<std::string> labels;
	readLabel(LABEL_NAME, labels);

	/* Load ncnn model (probably, need this only once) */
	ncnn::Net net;
	net.load_param(MODEL_PARAM_NAME);
	net.load_model(MODEL_BIN_NAME);

	/*** Process for each image ***/
	/* Read image using OpenCV */
	cv::Mat image = cv::imread(RESOURCE_DIR"cat.jpg");
	//cv::imshow("Display", image);

	/* PreProcess for ncnn */
	ncnn::Mat ncnnMat = ncnn::Mat::from_pixels_resize(image.data, MODEL_CHANNEL, image.cols, image.rows, MODEL_WIDTH, MODEL_HEIGHT);
	float mean[3] = { 123.675f, 116.28f, 103.53f };
	float norm[3] = { 1.0f, 1.0f, 1.0f };
	ncnnMat.substract_mean_normalize(mean, norm);

	/* Prepare inference */
	ncnn::Extractor ex = net.create_extractor();
	ex.set_light_mode(true);
	ex.set_num_threads(NCNN_CPU_NUM);
	ex.input("input", ncnnMat);

	/* Run inference */
	ncnn::Mat ncnnOut;
	ex.extract("detection_out", ncnnOut);

	/* Retrieve results */
	std::vector<BBox> bboxList;
	getBBox(bboxList, ncnnOut, 0.2, image.cols, image.rows);

	/* Display bbox */
	for (int i = 0; i < bboxList.size(); i++) {
		const BBox bbox = bboxList[i];
		cv::rectangle(image, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), cv::Scalar(255, 255, 0));
		cv::putText(image, labels[bbox.classId], cv::Point(bbox.x, bbox.y + 10), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 0), 3);
		cv::putText(image, labels[bbox.classId], cv::Point(bbox.x, bbox.y + 10), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1);
	}
	cv::imshow("test", image); cv::waitKey(1);



	/*** (optional) measure inference time ***/
	auto t0 = std::chrono::system_clock::now();
	for (int i = 0; i < LOOP_NUM_TO_MEASURE_INFERENCE_TIME; i++) {
		/*** Prepare inference ***/
		ncnn::Extractor ex = net.create_extractor();
		ex.set_light_mode(true);
		ex.set_num_threads(NCNN_CPU_NUM);
		ex.input("input", ncnnMat);

		/*** Run inference ***/
		ncnn::Mat ncnnOut;
		ex.extract("detection_out", ncnnOut);
	}
	auto t1 = std::chrono::system_clock::now();
	double inferenceTime = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
	printf("Inference time: %.2lf [msec]\n", inferenceTime / LOOP_NUM_TO_MEASURE_INFERENCE_TIME);

	cv::waitKey(0);
	return 0;
}
