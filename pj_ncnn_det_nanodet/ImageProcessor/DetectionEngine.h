#ifndef DETECTION_ENGINE_
#define DETECTION_ENGINE_

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <memory>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "InferenceHelper.h"


class DetectionEngine {
public:
	enum {
		RET_OK = 0,
		RET_ERR = -1,
	};

	typedef struct {
		int32_t     classId;
		std::string label;
		float_t  score;
		float_t  x;
		float_t  y;
		float_t  width;
		float_t  height;
	} OBJECT;

	typedef struct RESULT_ {
		std::vector<OBJECT> objectList;
		double_t            timePreProcess;		// [msec]
		double_t            timeInference;		// [msec]
		double_t            timePostProcess;	// [msec]
		RESULT_() : timePreProcess(0), timeInference(0), timePostProcess(0)
		{}
	} RESULT;

public:
	DetectionEngine() {}
	~DetectionEngine() {}
	int32_t initialize(const std::string& workDir, const int32_t numThreads);
	int32_t finalize(void);
	int32_t invoke(const cv::Mat& originalMat, RESULT& result);

private:
	int32_t readLabel(const std::string& filename, std::vector<std::string>& labelList);
	int32_t decodeInfer(std::vector<OBJECT>& objectList, const OutputTensorInfo& clsPred, const OutputTensorInfo& disPred, double_t threshold, int32_t stride, int32_t modelWidth, int32_t modelHeight);
	void disPred2Bbox(OBJECT& object, const OutputTensorInfo& disPred, int32_t idx, int32_t x, int32_t y, int32_t stride);
	void nms(std::vector<OBJECT>& objectList, std::vector<OBJECT>& objectListNMS, bool useWeight);
	float calculateIoU(const OBJECT& det0, const OBJECT& det1);

private:
	std::unique_ptr<InferenceHelper> m_inferenceHelper;
	std::vector<InputTensorInfo> m_inputTensorList;
	std::vector<OutputTensorInfo> m_outputTensorList;
	std::vector<std::string> m_labelList;
};

#endif
