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
//#define MODEL_NAME   "nanodet.onnx"
//#define MODEL_NAME   "nanodet"
#define MODEL_NAME   "nanodet_m.param"
#define LABEL_NAME   "coco_label.txt"
#define NUM_CLASS 80
#define REG_MAX 7

/*** Function ***/
int32_t DetectionEngine::initialize(const std::string& workDir, const int32_t numThreads)
{
	/* Set model information */
	std::string modelFilename = workDir + "/model/" + MODEL_NAME;
	std::string labelFilename = workDir + "/model/" + LABEL_NAME;

	/* Set input tensor info */
	m_inputTensorList.clear();
	InputTensorInfo inputTensorInfo;
	inputTensorInfo.name = "input.1";
	inputTensorInfo.tensorType = TensorInfo::TENSOR_TYPE_FP32;
	inputTensorInfo.tensorDims.batch = 1;
	inputTensorInfo.tensorDims.width = 320;
	inputTensorInfo.tensorDims.height = 320;
	inputTensorInfo.tensorDims.channel = 3;
	inputTensorInfo.dataType = InputTensorInfo::DATA_TYPE_IMAGE;
	inputTensorInfo.normalize.mean[0] = 0.408f;   /* https://github.com/RangiLyu/nanodet/blob/main/demo_android_ncnn/app/src/main/cpp/NanoDet.cpp */
	inputTensorInfo.normalize.mean[1] = 0.447f;
	inputTensorInfo.normalize.mean[2] = 0.470f;
	inputTensorInfo.normalize.norm[0] = 0.289f;
	inputTensorInfo.normalize.norm[1] = 0.274f;
	inputTensorInfo.normalize.norm[2] = 0.278f;
	m_inputTensorList.push_back(inputTensorInfo);

	/* Set output tensor info */
	m_outputTensorList.clear();
	OutputTensorInfo outputTensorInfo;
	outputTensorInfo.tensorType = TensorInfo::TENSOR_TYPE_FP32;
	outputTensorInfo.name = "792";
	m_outputTensorList.push_back(outputTensorInfo);
	outputTensorInfo.name = "795";
	m_outputTensorList.push_back(outputTensorInfo);
	outputTensorInfo.name = "814";
	m_outputTensorList.push_back(outputTensorInfo);
	outputTensorInfo.name = "817";
	m_outputTensorList.push_back(outputTensorInfo);
	outputTensorInfo.name = "836";
	m_outputTensorList.push_back(outputTensorInfo);
	outputTensorInfo.name = "839";
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

	/* Check if input tensor info is set */
	for (const auto& inputTensorInfo : m_inputTensorList) {
		if ((inputTensorInfo.tensorDims.width <= 0) || (inputTensorInfo.tensorDims.height <= 0) || inputTensorInfo.tensorType == TensorInfo::TENSOR_TYPE_NONE) {
			PRINT_E("Invalid tensor size\n");
			m_inferenceHelper.reset();
			return RET_ERR;
		}
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
#ifndef CV_COLOR_IS_RGB
	cv::cvtColor(imgSrc, imgSrc, cv::COLOR_BGR2RGB);
#endif
	inputTensorInfo.data = imgSrc.data;
	inputTensorInfo.dataType = InputTensorInfo::DATA_TYPE_IMAGE;
	inputTensorInfo.imageInfo.width = imgSrc.cols;
	inputTensorInfo.imageInfo.height = imgSrc.rows;
	inputTensorInfo.imageInfo.channel = imgSrc.channels();
	inputTensorInfo.imageInfo.cropX = 0;
	inputTensorInfo.imageInfo.cropY = 0;
	inputTensorInfo.imageInfo.cropWidth = imgSrc.cols;
	inputTensorInfo.imageInfo.cropHeight = imgSrc.rows;
	inputTensorInfo.imageInfo.isBGR = false;
	inputTensorInfo.imageInfo.swapColor = false;
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
	decodeInfer(objectList, m_outputTensorList[0], m_outputTensorList[1], 0.4, 8, inputTensorInfo.tensorDims.width, inputTensorInfo.tensorDims.height);
	decodeInfer(objectList, m_outputTensorList[2], m_outputTensorList[3], 0.4, 16, inputTensorInfo.tensorDims.width, inputTensorInfo.tensorDims.height);
	decodeInfer(objectList, m_outputTensorList[4], m_outputTensorList[5], 0.4, 32, inputTensorInfo.tensorDims.width, inputTensorInfo.tensorDims.height);

	/* NMS */
	std::vector<OBJECT> objectListNms;
	nms(objectList, objectListNms, false);

	/* Convert coordinate (model size to image size) */
	for (auto& object : objectListNms) {
		object.x = (object.x * originalMat.cols) / inputTensorInfo.tensorDims.width;
		object.width = (object.width * originalMat.cols) / inputTensorInfo.tensorDims.width;
		object.y = (object.y * originalMat.rows) / inputTensorInfo.tensorDims.height;
		object.height = (object.height * originalMat.rows) / inputTensorInfo.tensorDims.height;
	}
	const auto& tPostProcess1 = std::chrono::steady_clock::now();

	/* Return the results */
	result.objectList = objectListNms;
	result.timePreProcess = static_cast<std::chrono::duration<double>>(tPreProcess1 - tPreProcess0).count() * 1000.0;
	result.timeInference = static_cast<std::chrono::duration<double>>(tInference1 - tInference0).count() * 1000.0;
	result.timePostProcess = static_cast<std::chrono::duration<double>>(tPostProcess1 - tPostProcess0).count() * 1000.0;;

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

/* Original code: https://github.com/RangiLyu/nanodet/blob/main/demo_ncnn/nanodet.cpp */
int32_t DetectionEngine::decodeInfer(std::vector<OBJECT>& objectList, const OutputTensorInfo& clsPred, const OutputTensorInfo& disPred, double threshold, int32_t stride, int32_t modelWidth, int32_t modelHeight)
{
	int32_t feature_w = modelWidth / stride;
	int32_t feature_h = modelHeight / stride;

	for (int32_t idx = 0; idx < feature_h * feature_w; idx++) {
		
		const float* score = static_cast<const float*>(clsPred.data);
		int32_t row = idx / feature_h;
		int32_t col = idx % feature_w;
		float scoreMax = 0;
		int32_t classIdMax = 0;
		for (int32_t label = 0; label < NUM_CLASS; label++) {
			//float currentScore = score[clsPred.tensorDims.width * label + idx];	/* memo: In ONNX model, H = label, W = pos(idx) */
			float currentScore = score[clsPred.tensorDims.width * idx + label];
			if (currentScore > scoreMax) {
				scoreMax = currentScore;
				classIdMax = label;
			}
		}
		if (scoreMax > threshold) {
			OBJECT object;
			disPred2Bbox(object, disPred, idx, col, row, stride);
			object.x = (std::max)(object.x, 0.f);
			object.y = (std::max)(object.y, 0.f);
			object.width = (std::min)(object.width, modelWidth - object.x);
			object.height = (std::min)(object.height, modelHeight - object.y);
			object.classId = classIdMax;
			object.label = m_labelList[object.classId];
			object.score = scoreMax;
			objectList.push_back(object);
		}
	}
	return RET_OK;
}

inline float fast_exp(float x)
{
	union {
		uint32_t i;
		float f;
	} v{};
	v.i = static_cast<int32_t>((1 << 23) * (1.4426950409 * x + 126.93490512f));
	return v.f;
}

inline float sigmoid(float x)
{
	return 1.0f / (1.0f + fast_exp(-x));
}

template<typename _Tp>
int32_t activation_function_softmax(const _Tp* src, _Tp* dst, int32_t length)
{
	const _Tp alpha = *std::max_element(src, src + length);
	_Tp denominator{ 0 };

	for (int32_t i = 0; i < length; ++i) {
		dst[i] = fast_exp(src[i] - alpha);
		denominator += dst[i];
	}

	for (int32_t i = 0; i < length; ++i) {
		dst[i] /= denominator;
	}

	return 0;
}

void DetectionEngine::disPred2Bbox(OBJECT& object, const OutputTensorInfo& disPred, int32_t idx, int32_t x, int32_t y, int32_t stride)
{
	float ct_x = (x + 0.5f) * stride;
	float ct_y = (y + 0.5f) * stride;
	std::vector<float> dis_pred;
	dis_pred.resize(4);


	for (int32_t i = 0; i < 4; i++) {
		float dis = 0;
		float dis_after_sm[REG_MAX + 1];
		//activation_function_softmax(static_cast<float*>(disPred.data) + disPred.tensorDims.width * (i * (REG_MAX + 1)) + idx, dis_after_sm, REG_MAX + 1);		/* memo: In ONNX model, H = label, W = pos(idx) */
		activation_function_softmax(static_cast<float*>(disPred.data) + disPred.tensorDims.width * idx + (i * (REG_MAX + 1)), dis_after_sm, REG_MAX + 1);
		for (int32_t j = 0; j < REG_MAX + 1; j++) {
			dis += j * dis_after_sm[j];
		}
		dis *= stride;
		dis_pred[i] = dis;
	}

	object.x = (std::max)(ct_x - dis_pred[0], 0.0f);
	object.y = (std::max)(ct_y - dis_pred[1], 0.0f);
	object.width = ct_x + dis_pred[2] - object.x;
	object.height = ct_y + dis_pred[3] - object.y;

	return;
}

float DetectionEngine::calculateIoU(const OBJECT& det0, const OBJECT& det1)
{
	float interx0 = std::max(det0.x, det1.x);
	float intery0 = std::max(det0.y, det1.y);
	float interx1 = std::min(det0.x + det0.width, det1.x + det1.width);
	float intery1 = std::min(det0.y + det0.height, det1.y + det1.height);

	float area0 = det0.width * det0.height;
	float area1 = det1.width * det1.height;
	float areaInter = (interx1 - interx0) * (intery1 - intery0);
	float areaSum = area0 + area1 - areaInter;

	return areaInter / areaSum;
}

void DetectionEngine::nms(std::vector<OBJECT> &objectList, std::vector<OBJECT> &objectListNMS, bool useWeight)
{
	std::sort(objectList.begin(), objectList.end(), [](OBJECT const& lhs, OBJECT const& rhs) {
		if (lhs.width * lhs.height > rhs.width * rhs.height) return true;
		// if (lhs.score > rhs.score) return true;
		return false;
	});

	std::unique_ptr<bool[]> isMerged(new bool[objectList.size()]);
	for (int32_t i = 0; i < objectList.size(); i++) isMerged[i] = false;
	for (int32_t indexHighScore = 0; indexHighScore < objectList.size(); indexHighScore++) {
		std::vector<OBJECT> candidates;
		if (isMerged[indexHighScore]) continue;
		candidates.push_back(objectList[indexHighScore]);
		for (int32_t indexLowScore = indexHighScore + 1; indexLowScore < objectList.size(); indexLowScore++) {
			if (isMerged[indexLowScore]) continue;
			if (objectList[indexHighScore].classId != objectList[indexLowScore].classId) continue;
			if (calculateIoU(objectList[indexHighScore], objectList[indexLowScore]) > 0.5) {
				candidates.push_back(objectList[indexLowScore]);
				isMerged[indexLowScore] = true;
			}
		}

		/* weight by score */
		if (useWeight) {
			if (candidates.size() < 2) continue;	// do not use detected object if the number of bbox is small
			OBJECT mergedBox = candidates[0];
			mergedBox.score = 0;
			mergedBox.x = 0;
			mergedBox.y = 0;
			mergedBox.width = 0;
			mergedBox.height = 0;
			float sumScore = 0;
			for (auto candidate : candidates) {
				sumScore += candidate.score;
				mergedBox.score += candidate.score;
				mergedBox.x += candidate.x * candidate.score;
				mergedBox.y += candidate.y * candidate.score;
				mergedBox.width += candidate.width * candidate.score;
				mergedBox.height += candidate.height * candidate.score;

			}
			mergedBox.score /= candidates.size();
			mergedBox.x /= sumScore;
			mergedBox.y /= sumScore;
			mergedBox.width /= sumScore;
			mergedBox.height /= sumScore;
			objectListNMS.push_back(mergedBox);
		} else {
			objectListNMS.push_back(candidates[0]);
		}

	}
}

