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
#include "common_helper.h"
#include "inference_helper.h"
#include "detection_engine.h"

/*** Macro ***/
#define TAG "DetectionEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
//#define MODEL_NAME   "nanodet.onnx"
//#define MODEL_NAME   "nanodet"
#define MODEL_NAME   "nanodet_m.param"
#define LABEL_NAME   "coco_label.txt"
#define INPUT_NAME   "input.1"
#define INPUT_DIMS    { 1, 3, 320, 320 }
#define OUTPUT_0_NAME  "792"
#define OUTPUT_1_NAME  "795"
#define OUTPUT_2_NAME  "814"
#define OUTPUT_3_NAME  "817"
#define OUTPUT_4_NAME  "836"
#define OUTPUT_5_NAME  "839"
#define TENSORTYPE    TensorInfo::kTensorTypeFp32
#define IS_NCHW       true

#define NUM_CLASS 80
#define REG_MAX 7

/*** Function ***/
int32_t DetectionEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
	/* Set model information */
	std::string model_filename = work_dir + "/model/" + MODEL_NAME;
	std::string label_filename = work_dir + "/model/" + LABEL_NAME;

	/* Set input tensor info */
	input_tensor_info_list_.clear();
	InputTensorInfo input_tensor_info(INPUT_NAME, TENSORTYPE, IS_NCHW);
	input_tensor_info.tensor_dims = INPUT_DIMS;
	input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
	input_tensor_info.normalize.mean[0] = 0.408f;   /* https://github.com/RangiLyu/nanodet/blob/main/demo_android_ncnn/app/src/main/cpp/NanoDet.cpp */
	input_tensor_info.normalize.mean[1] = 0.447f;
	input_tensor_info.normalize.mean[2] = 0.470f;
	input_tensor_info.normalize.norm[0] = 0.289f;
	input_tensor_info.normalize.norm[1] = 0.274f;
	input_tensor_info.normalize.norm[2] = 0.278f;
	input_tensor_info_list_.push_back(input_tensor_info);

	/* Set output tensor info */
	output_tensor_info_list_.clear();
	output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_0_NAME, TENSORTYPE));
	output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_1_NAME, TENSORTYPE));
	output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_2_NAME, TENSORTYPE));
	output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_3_NAME, TENSORTYPE));
	output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_4_NAME, TENSORTYPE));
	output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_5_NAME, TENSORTYPE));

	/* Create and Initialize Inference Helper */
	inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kNcnn));

	if (!inference_helper_) {
		return kRetErr;
	}
	if (inference_helper_->SetNumThreads(num_threads) != InferenceHelper::kRetOk) {
		inference_helper_.reset();
		return kRetErr;
	}
	if (inference_helper_->Initialize(model_filename, input_tensor_info_list_, output_tensor_info_list_) != InferenceHelper::kRetOk) {
		inference_helper_.reset();
		return kRetErr;
	}

	/* read label */
	if (ReadLabel(label_filename, label_list_) != kRetOk) {
		return kRetErr;
	}


	return kRetOk;
}

int32_t DetectionEngine::Finalize()
{
	if (!inference_helper_) {
		PRINT_E("Inference helper is not created\n");
		return kRetErr;
	}
	inference_helper_->Finalize();
	return kRetOk;
}


int32_t DetectionEngine::Process(const cv::Mat& original_mat, Result& result)
{
	if (!inference_helper_) {
		PRINT_E("Inference helper is not created\n");
		return kRetErr;
	}
	/*** PreProcess ***/
	const auto& t_pre_process0 = std::chrono::steady_clock::now();
	InputTensorInfo& input_tensor_info = input_tensor_info_list_[0];
	/* do resize and color conversion here because some inference engine doesn't support these operations */
	cv::Mat img_src;
	cv::resize(original_mat, img_src, cv::Size(input_tensor_info.GetWidth(), input_tensor_info.GetHeight()));
#ifndef CV_COLOR_IS_RGB
	cv::cvtColor(img_src, img_src, cv::COLOR_BGR2RGB);
#endif
	input_tensor_info.data = img_src.data;
	input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
	input_tensor_info.image_info.width = img_src.cols;
	input_tensor_info.image_info.height = img_src.rows;
	input_tensor_info.image_info.channel = img_src.channels();
	input_tensor_info.image_info.crop_x = 0;
	input_tensor_info.image_info.crop_y = 0;
	input_tensor_info.image_info.crop_width = img_src.cols;
	input_tensor_info.image_info.crop_height = img_src.rows;
	input_tensor_info.image_info.is_bgr = false;
	input_tensor_info.image_info.swap_color = false;
	if (inference_helper_->PreProcess(input_tensor_info_list_) != InferenceHelper::kRetOk) {
		return kRetErr;
	}
	const auto& t_pre_process1 = std::chrono::steady_clock::now();

	/*** Inference ***/
	const auto& t_inference0 = std::chrono::steady_clock::now();
	if (inference_helper_->Process(output_tensor_info_list_) != InferenceHelper::kRetOk) {
		return kRetErr;
	}
	const auto& t_inference1 = std::chrono::steady_clock::now();

	/*** PostProcess ***/
	const auto& t_post_process0 = std::chrono::steady_clock::now();
	/* Retrieve result */
	std::vector<Object> object_list;
	DecodeInfer(object_list, output_tensor_info_list_[0], output_tensor_info_list_[1], 0.4, 8, input_tensor_info.GetWidth(), input_tensor_info.GetHeight());
	DecodeInfer(object_list, output_tensor_info_list_[2], output_tensor_info_list_[3], 0.4, 16, input_tensor_info.GetWidth(), input_tensor_info.GetHeight());
	DecodeInfer(object_list, output_tensor_info_list_[4], output_tensor_info_list_[5], 0.4, 32, input_tensor_info.GetWidth(), input_tensor_info.GetHeight());

	/* NMS */
	std::vector<Object> object_list_nms;
	Nms(object_list, object_list_nms, false);

	/* Convert coordinate (model size to image size) */
	for (auto& object : object_list_nms) {
		object.x = (object.x * original_mat.cols) / input_tensor_info.GetWidth();
		object.width = (object.width * original_mat.cols) / input_tensor_info.GetWidth();
		object.y = (object.y * original_mat.rows) / input_tensor_info.GetHeight();
		object.height = (object.height * original_mat.rows) / input_tensor_info.GetHeight();
	}
	const auto& t_post_process1 = std::chrono::steady_clock::now();

	/* Return the results */
	result.object_list = object_list_nms;
	result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
	result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
	result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;;

	return kRetOk;
}


int32_t DetectionEngine::ReadLabel(const std::string& filename, std::vector<std::string>& label_list)
{
	std::ifstream ifs(filename);
	if (ifs.fail()) {
		PRINT_E("Failed to read %s\n", filename.c_str());
		return kRetErr;
	}
	label_list.clear();
	std::string str;
	while (getline(ifs, str)) {
		label_list.push_back(str);
	}
	return kRetOk;
}

/* Original code: https://github.com/RangiLyu/nanodet/blob/main/demo_ncnn/nanodet.cpp */
int32_t DetectionEngine::DecodeInfer(std::vector<Object>& object_list, const OutputTensorInfo& cls_pred, const OutputTensorInfo& dis_pred, double threshold, int32_t stride, int32_t model_width, int32_t model_height)
{
	int32_t feature_w = model_width / stride;
	int32_t feature_h = model_height / stride;

	for (int32_t idx = 0; idx < feature_h * feature_w; idx++) {
		
		const float* score = static_cast<const float*>(cls_pred.data);
		int32_t row = idx / feature_h;
		int32_t col = idx % feature_w;
		float scoreMax = 0;
		int32_t classIdMax = 0;
		for (int32_t label = 0; label < NUM_CLASS; label++) {
			//float currentScore = score[cls_pred.tensor_dims.width * label + idx];	/* memo: In ONNX model, H = label, W = pos(idx) */
			float currentScore = score[cls_pred.tensor_dims[3] * idx + label];
			if (currentScore > scoreMax) {
				scoreMax = currentScore;
				classIdMax = label;
			}
		}
		if (scoreMax > threshold) {
			Object object;
			DisPred2Bbox(object, dis_pred, idx, col, row, stride);
			object.x = (std::max)(object.x, 0.f);
			object.y = (std::max)(object.y, 0.f);
			object.width = (std::min)(object.width, model_width - object.x);
			object.height = (std::min)(object.height, model_height - object.y);
			object.class_id = classIdMax;
			object.label = label_list_[object.class_id];
			object.score = scoreMax;
			object_list.push_back(object);
		}
	}
	return kRetOk;
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
int32_t Activation_function_softmax(const _Tp* src, _Tp* dst, int32_t length)
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

void DetectionEngine::DisPred2Bbox(Object& object, const OutputTensorInfo& dis_pred_raw, int32_t idx, int32_t x, int32_t y, int32_t stride)
{
	float ct_x = (x + 0.5f) * stride;
	float ct_y = (y + 0.5f) * stride;
	std::vector<float> dis_pred;
	dis_pred.resize(4);


	for (int32_t i = 0; i < 4; i++) {
		float dis = 0;
		float dis_after_sm[REG_MAX + 1];
		//activation_function_softmax(static_cast<float*>(dis_pred.data) + dis_pred.tensor_dims.width * (i * (REG_MAX + 1)) + idx, dis_after_sm, REG_MAX + 1);		/* memo: In ONNX model, H = label, W = pos(idx) */
		Activation_function_softmax(static_cast<float*>(dis_pred_raw.data) + dis_pred_raw.tensor_dims[3] * idx + (i * (REG_MAX + 1)), dis_after_sm, REG_MAX + 1);
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

float DetectionEngine::CalculateIoU(const Object& det0, const Object& det1)
{
	float interx0 = std::max(det0.x, det1.x);
	float intery0 = std::max(det0.y, det1.y);
	float interx1 = std::min(det0.x + det0.width, det1.x + det1.width);
	float intery1 = std::min(det0.y + det0.height, det1.y + det1.height);

	float area0 = det0.width * det0.height;
	float area1 = det1.width * det1.height;
	float area_inter = (interx1 - interx0) * (intery1 - intery0);
	float area_sum = area0 + area1 - area_inter;

	return area_inter / area_sum;
}

void DetectionEngine::Nms(std::vector<Object> &object_list, std::vector<Object> &object_list_nms, bool use_weight)
{
	std::sort(object_list.begin(), object_list.end(), [](Object const& lhs, Object const& rhs) {
		if (lhs.width * lhs.height > rhs.width * rhs.height) return true;
		// if (lhs.score > rhs.score) return true;
		return false;
	});

	std::unique_ptr<bool[]> is_merged(new bool[object_list.size()]);
	for (int32_t i = 0; i < object_list.size(); i++) is_merged[i] = false;
	for (int32_t index_high_score = 0; index_high_score < object_list.size(); index_high_score++) {
		std::vector<Object> candidates;
		if (is_merged[index_high_score]) continue;
		candidates.push_back(object_list[index_high_score]);
		for (int32_t index_low_score = index_high_score + 1; index_low_score < object_list.size(); index_low_score++) {
			if (is_merged[index_low_score]) continue;
			if (object_list[index_high_score].class_id != object_list[index_low_score].class_id) continue;
			if (CalculateIoU(object_list[index_high_score], object_list[index_low_score]) > 0.5) {
				candidates.push_back(object_list[index_low_score]);
				is_merged[index_low_score] = true;
			}
		}

		/* weight by score */
		if (use_weight) {
			if (candidates.size() < 2) continue;	// do not use detected object if the number of bbox is small
			Object merged_box = candidates[0];
			merged_box.score = 0;
			merged_box.x = 0;
			merged_box.y = 0;
			merged_box.width = 0;
			merged_box.height = 0;
			float sum_score = 0;
			for (auto candidate : candidates) {
				sum_score += candidate.score;
				merged_box.score += candidate.score;
				merged_box.x += candidate.x * candidate.score;
				merged_box.y += candidate.y * candidate.score;
				merged_box.width += candidate.width * candidate.score;
				merged_box.height += candidate.height * candidate.score;

			}
			merged_box.score /= candidates.size();
			merged_box.x /= sum_score;
			merged_box.y /= sum_score;
			merged_box.width /= sum_score;
			merged_box.height /= sum_score;
			object_list_nms.push_back(merged_box);
		} else {
			object_list_nms.push_back(candidates[0]);
		}

	}
}

