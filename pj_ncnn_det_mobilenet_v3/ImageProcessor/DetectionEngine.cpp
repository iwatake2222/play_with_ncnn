/*** Include ***/
/* for general */
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <fstream>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "CommonHelper.h"
#include "InferenceHelper.h"
#include "DetectionEngine.h"

/*** Macro ***/
#define TAG "DetectionEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define MODEL_NAME   "mobilenetv3_ssdlite_voc"
#define LABEL_NAME   "label_PASCAL_VOC2012.txt"


/*** Function ***/
int32_t DetectionEngine::initialize(const std::string& workDir, const int32_t numThreads)
{
	/* Set model information */
	std::string modelFilename = workDir + "/model/" + MODEL_NAME;
	std::string labelFilename = workDir + "/model/" + LABEL_NAME;

	/* Set input tensor info */
	m_inputTensorList.clear();
	InputTensorInfo inputTensorInfo;
	inputTensorInfo.name = "input";
	inputTensorInfo.tensorType = TensorInfo::TENSOR_TYPE_FP32;
	inputTensorInfo.tensorDims.batch = 1;
	inputTensorInfo.tensorDims.width = 300;
	inputTensorInfo.tensorDims.height = 300;
	inputTensorInfo.tensorDims.channel = 3;
	inputTensorInfo.data = nullptr;
	inputTensorInfo.dataType = InputTensorInfo::DATA_TYPE_IMAGE;
	inputTensorInfo.imageInfo.width = -1;
	inputTensorInfo.imageInfo.height = -1;
	inputTensorInfo.imageInfo.channel = -1;
	inputTensorInfo.imageInfo.cropX = -1;
	inputTensorInfo.imageInfo.cropY = -1;
	inputTensorInfo.imageInfo.cropWidth = -1;
	inputTensorInfo.imageInfo.cropHeight = -1;
	inputTensorInfo.imageInfo.isBGR = true;
	inputTensorInfo.imageInfo.swapColor = false;
	inputTensorInfo.normalize.mean[0] = 0.485f;   /* https://github.com/Tencent/ncnn/blob/master/examples/mobilenetv3ssdlite.cpp */
	inputTensorInfo.normalize.mean[1] = 0.456f;
	inputTensorInfo.normalize.mean[2] = 0.406f;
	inputTensorInfo.normalize.norm[0] = 1 / 255.0f;
	inputTensorInfo.normalize.norm[1] = 1 / 255.0f;
	inputTensorInfo.normalize.norm[2] = 1 / 255.0f;
#if 0
	/* Convert to speeden up normalization:  ((src / 255) - mean) / norm  = src * 1 / (255 * norm) - (mean / norm) */
	for (int32_t i = 0; i < 3; i++) {
		inputTensorInfo.normalize.mean[i] /= inputTensorInfo.normalize.norm[i];
		inputTensorInfo.normalize.norm[i] *= 255.0f;
		inputTensorInfo.normalize.norm[i] = 1.0f / inputTensorInfo.normalize.norm[i];
	}
#endif
#if 1
	/* Convert to speeden up normalization:  ((src / 255) - mean) / norm = (src  - (mean * 255))  * (1 / (255 * norm)) */
	for (int32_t i = 0; i < 3; i++) {
		inputTensorInfo.normalize.mean[i] *= 255.0f;
		inputTensorInfo.normalize.norm[i] *= 255.0f;
		inputTensorInfo.normalize.norm[i] = 1.0f / inputTensorInfo.normalize.norm[i];
	}
#endif
	m_inputTensorList.push_back(inputTensorInfo);

	/* Set output tensor info */
	m_outputTensorList.clear();
	OutputTensorInfo outputTensorInfo;
	outputTensorInfo.name = "detection_out";
	outputTensorInfo.tensorType = TensorInfo::TENSOR_TYPE_FP32;
	m_outputTensorList.push_back(outputTensorInfo);

	/* Create and Initialize Inference Helper */
	//m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::OPEN_CV));
	//m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::TENSOR_RT));
	m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::NCNN));

	if (!m_inferenceHelper) {
		return RET_ERR;
	}
	if (m_inferenceHelper->setNumThread(numThreads) != InferenceHelper::RET_OK) {
		m_inferenceHelper.reset();
		return RET_ERR;
	}
	if (m_inferenceHelper->initialize(modelFilename, m_inputTensorList, m_outputTensorList) != InferenceHelper::RET_OK) {
		m_inferenceHelper.reset();
		return RET_ERR;
	}

	/* read label */
	if (readLabel(labelFilename, m_labelList) != RET_OK) {
		return RET_ERR;
	}


	return RET_OK;
}

int32_t DetectionEngine::finalize()
{
	if (!m_inferenceHelper) {
		PRINT_E("Inference helper is not created\n");
		return RET_ERR;
	}
	m_inferenceHelper->finalize();
	return RET_OK;
}


int32_t DetectionEngine::invoke(const cv::Mat& originalMat, RESULT& result)
{
	if (!m_inferenceHelper) {
		PRINT_E("Inference helper is not created\n");
		return RET_ERR;
	}
	/*** PreProcess ***/
	const auto& tPreProcess0 = std::chrono::steady_clock::now();
	InputTensorInfo& inputTensorInfo = m_inputTensorList[0];
	/* do resize and color conversion here because some inference engine doesn't support these operations */
	cv::Mat imgSrc;
	cv::resize(originalMat, imgSrc, cv::Size(inputTensorInfo.tensorDims.width, inputTensorInfo.tensorDims.height));
	if (inputTensorInfo.imageInfo.channel == 3 && inputTensorInfo.imageInfo.swapColor) {
		cv::cvtColor(imgSrc, imgSrc, cv::COLOR_BGR2RGB);
	}
	inputTensorInfo.data = imgSrc.data;
	inputTensorInfo.dataType = InputTensorInfo::DATA_TYPE_IMAGE;
	inputTensorInfo.imageInfo.width = imgSrc.cols;
	inputTensorInfo.imageInfo.height = imgSrc.rows;
	inputTensorInfo.imageInfo.channel = imgSrc.channels();
	inputTensorInfo.imageInfo.cropX = 0;
	inputTensorInfo.imageInfo.cropY = 0;
	inputTensorInfo.imageInfo.cropWidth = imgSrc.cols;
	inputTensorInfo.imageInfo.cropHeight = imgSrc.rows;
	inputTensorInfo.imageInfo.isBGR = true;
	inputTensorInfo.imageInfo.swapColor = true;
	if (m_inferenceHelper->preProcess(m_inputTensorList) != InferenceHelper::RET_OK) {
		return RET_ERR;
	}
	const auto& tPreProcess1 = std::chrono::steady_clock::now();

	/*** Inference ***/
	const auto& tInference0 = std::chrono::steady_clock::now();
	if (m_inferenceHelper->invoke(m_outputTensorList) != InferenceHelper::RET_OK) {
		return RET_ERR;
	}
	const auto& tInference1 = std::chrono::steady_clock::now();

	/*** PostProcess ***/
	const auto& tPostProcess0 = std::chrono::steady_clock::now();
	/* Retrieve result */
	std::vector<OBJECT> objectList;
	getObject(m_outputTensorList[0], objectList, 0.2, originalMat.cols, originalMat.rows);
	const auto& tPostProcess1 = std::chrono::steady_clock::now();

	/* Return the results */
	result.objectList = objectList;
	result.timePreProcess = static_cast<std::chrono::duration<double_t>>(tPreProcess1 - tPreProcess0).count() * 1000.0;
	result.timeInference = static_cast<std::chrono::duration<double_t>>(tInference1 - tInference0).count() * 1000.0;
	result.timePostProcess = static_cast<std::chrono::duration<double_t>>(tPostProcess1 - tPostProcess0).count() * 1000.0;;

	return RET_OK;
}


int32_t DetectionEngine::readLabel(const std::string& filename, std::vector<std::string>& labelList)
{
	std::ifstream ifs(filename);
	if (ifs.fail()) {
		PRINT_E("Failed to read %s\n", filename.c_str());
		return RET_ERR;
	}
	labelList.clear();
	std::string str;
	while (getline(ifs, str)) {
		labelList.push_back(str);
	}
	return RET_OK;
}


int32_t DetectionEngine::getObject(const OutputTensorInfo& rawOutput, std::vector<OBJECT>& objectList, double_t threshold, int32_t width, int32_t height)
{
	const float_t* p = static_cast<const float_t*>(rawOutput.data);
	for (int32_t i = 0; i < rawOutput.tensorDims.height; i++) {
		const float_t* values = p + (rawOutput.tensorDims.width * i);
		OBJECT object;
		object.classId = (int32_t)values[0];
		object.label = m_labelList[object.classId];
		object.score = values[1];
		if (object.score < threshold) continue;
		object.x = std::max<float_t>(values[2], 0.0f);
		object.y = std::max<float_t>(values[3], 0.0f);
		object.width = std::min<float_t>(values[4], 1.0f) - values[2];
		object.height = std::min<float_t>(values[5], 1.0f) - values[3];
		if (width > 0) {
			object.x *= width;
			object.y *= width;
			object.width *= height;
			object.height *= height;
		}
		objectList.push_back(object);
	}

	return RET_OK;
}