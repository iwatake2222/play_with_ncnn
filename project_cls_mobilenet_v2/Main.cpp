/*** Include ***/
/* for general */
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <chrono>

/* for OpenCV */
#include <opencv2/opencv.hpp>

#include "ImageProcessor.h"

/*** Macro ***/
/* Model parameters */
#define MODEL_NAME   RESOURCE_DIR"/model/mobilenetv2-1.0"
#define LABEL_NAME   RESOURCE_DIR"/model/imagenet_labels.txt"
#define IMAGE_NAME   RESOURCE_DIR"/parrot.jpg"

/* Settings */
#define LOOP_NUM_FOR_TIME_MEASUREMENT 10

int main()
{
	/*** Initialize ***/
	/* Initialize image processor library */
	INPUT_PARAM inputParam;
	strcpy_s(inputParam.labelFilename, sizeof(inputParam.labelFilename), LABEL_NAME);
	ImageProcessor_initialize(MODEL_NAME, &inputParam);

	/* Read an input image */
	cv::Mat originalImage = cv::imread(IMAGE_NAME);

	/* Call image processor library */
	OUTPUT_PARAM outputParam;
	ImageProcessor_process(&originalImage, &outputParam);

	cv::imshow("originalImage", originalImage);
	cv::waitKey(1);

	/*** (Optional) Measure inference time ***/
	const auto& t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < LOOP_NUM_FOR_TIME_MEASUREMENT; i++) {
		ImageProcessor_process(&originalImage, &outputParam);
	}
	const auto& t1 = std::chrono::steady_clock::now();
	std::chrono::duration<double> timeSpan = t1 - t0;
	printf("Inference time = %f [msec]\n", timeSpan.count() * 1000.0 / LOOP_NUM_FOR_TIME_MEASUREMENT);

	/* Fianlize image processor library */
	ImageProcessor_finalize();

	cv::waitKey(-1);

	return 0;
}
