#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/resize_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe {

	template <typename Dtype>
	__global__ void NNInterporationForward(const int nthreads,
		const Dtype* bottom_data, Dtype* top_data, const int num, const int channels,
		const int bot_h, const int bot_w, const int bot_hw, const int bot_chw, const int top_height_, const int top_width_) {

		Dtype bot_hDtop_h = (Dtype)bot_h / top_height_;
		Dtype bot_wDtop_w = (Dtype)bot_w / top_width_;

		CUDA_KERNEL_LOOP(index, nthreads) {
			const int pw = index % top_width_;
			const int ph = (index / top_width_) % top_height_;
			const int c = (index / top_width_ / top_height_) % channels;
			const int n = index / top_width_ / top_height_ / channels;
			
			
			int h_ = __max(0, __min(bot_h - 1, int(bot_hDtop_h*(ph + 1) - 0.5)));
			int w_ = __max(0, __min(bot_w - 1, int(bot_wDtop_w*(pw + 1) - 0.5)));
			int bot_index = n*bot_chw + c*bot_hw + h_*bot_w + w_;

			top_data[index] = bottom_data[bot_index];
		}
	}

	template <typename Dtype>
	__global__ void LinearInterporationForward(const int nthreads,
		const Dtype* bottom_data, Dtype* top_data, const int num, const int channels,
		const int bot_h, const int bot_w, const int bot_hw, const int bot_chw, const int top_height_, const int top_width_) {

		Dtype bot_hDtop_h = (Dtype)bot_h / top_height_;
		Dtype bot_wDtop_w = (Dtype)bot_w / top_width_;

		CUDA_KERNEL_LOOP(index, nthreads) {
			const int pw = index % top_width_;
			const int ph = (index / top_width_) % top_height_;
			const int c = (index / top_width_ / top_height_) % channels;
			const int n = index / top_width_ / top_height_ / channels;

			Dtype h_ = __max(0, __min(bot_h - 1, bot_hDtop_h*(ph + 1) - 1.5));
			Dtype w_ = __max(0, __min(bot_w - 1, bot_wDtop_w*(pw + 1) - 1.5));
			int top = __max(0, __min(bot_h - 1, int(h_)));
			int bot = __max(0, __min(bot_h - 1, top + 1));
			int left = __max(0, __min(bot_w - 1, int(w_)));
			int right = __max(0, __min(bot_w - 1, left + 1));

			Dtype hr = __max(0, __min(1, h_ - top));
			Dtype wr = __max(0, __min(1, w_ - left));

			int bot_index_top_left = n*bot_chw + c*bot_hw + top*bot_w + left;
			int bot_index_top_right = n*bot_chw + c*bot_hw + top*bot_w + right;
			int bot_index_bot_left = n*bot_chw + c*bot_hw + bot*bot_w + left;
			int bot_index_bot_right = n*bot_chw + c*bot_hw + bot*bot_w + right;

			top_data[index] = bottom_data[bot_index_top_left] * (1 - hr)*(1 - wr)
				+ bottom_data[bot_index_top_right] * (1 - hr)*(wr)
				+bottom_data[bot_index_bot_left] * (hr)*(1 - wr)
				+ bottom_data[bot_index_bot_right] * (hr)*(wr);
		}
	}



	template <typename Dtype>
	void ResizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		const int count = top[0]->count();

		switch (this->layer_param_.resize_param().interpolation())
		{
		case ResizeParameter_InterpolationMethod_NN:
		{
			int bot_chw = bottom[0]->count() / bottom[0]->num();
			int bot_hw = bot_chw / bottom[0]->channels();
			int bot_w = bottom[0]->width();
			int bot_h = bottom[0]->height();

			NNInterporationForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
				count, bottom_data, top_data, bottom[0]->num(), bottom[0]->channels(), bot_h, bot_w, bot_hw, bot_chw,
				top_height_, top_width_);

			break;
		}
		case ResizeParameter_InterpolationMethod_LINEAR:
		{
			int bot_chw = bottom[0]->count() / bottom[0]->num();
			int bot_hw = bot_chw / bottom[0]->channels();
			int bot_w = bottom[0]->width();
			int bot_h = bottom[0]->height();

			LinearInterporationForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
				count, bottom_data, top_data, bottom[0]->num(), bottom[0]->channels(), bot_h, bot_w, bot_hw, bot_chw,
				top_height_, top_width_);

			break;
		}
		}
		CUDA_POST_KERNEL_CHECK;
	}


	template <typename Dtype>
	__global__ void NNInterporationBackward(const int nthreads, 
		Dtype* bottom_diff, const Dtype* top_diff, const int num, const int channels,
		const int bot_h, const int bot_w, const int bot_hw, const int bot_chw, const int top_height_, const int top_width_) {

		Dtype bot_hDtop_h = (Dtype)bot_h / top_height_;
		Dtype bot_wDtop_w = (Dtype)bot_w / top_width_;

		CUDA_KERNEL_LOOP(index, nthreads) {
			const int pw = index % top_width_;
			const int ph = (index / top_width_) % top_height_;
			const int c = (index / top_width_ / top_height_) % channels;
			const int n = index / top_width_ / top_height_ / channels;

			int h_ = __max(0, __min(bot_h - 1, int(bot_hDtop_h*(ph + 1) - 0.5)));
			int w_ = __max(0, __min(bot_w - 1, int(bot_wDtop_w*(pw + 1) - 0.5)));
			int bot_index = n*bot_chw + c*bot_hw + h_*bot_w + w_;

			caffe_gpu_atomic_add(top_diff[index], bottom_diff + bot_index);
		}
	}

	template <typename Dtype>
	__global__ void LinearInterporationBackward(const int nthreads,
		Dtype* bottom_diff, const Dtype* top_diff, const int num, const int channels,
		const int bot_h, const int bot_w, const int bot_hw, const int bot_chw, const int top_height_, const int top_width_) {

		Dtype bot_hDtop_h = (Dtype)bot_h / top_height_;
		Dtype bot_wDtop_w = (Dtype)bot_w / top_width_;

		CUDA_KERNEL_LOOP(index, nthreads) {
			const int pw = index % top_width_;
			const int ph = (index / top_width_) % top_height_;
			const int c = (index / top_width_ / top_height_) % channels;
			const int n = index / top_width_ / top_height_ / channels;

			Dtype h_ = __max(0, __min(bot_h - 1, bot_hDtop_h*(ph + 1) - 1.5));
			Dtype w_ = __max(0, __min(bot_w - 1, bot_wDtop_w*(pw + 1) - 1.5));
			int top = __max(0, __min(bot_h - 1, int(h_)));
			int bot = __max(0, __min(bot_h - 1, top + 1));
			int left = __max(0, __min(bot_w - 1, int(w_)));
			int right = __max(0, __min(bot_w - 1, left + 1));

			Dtype hr = __max(0, __min(1, h_ - top));
			Dtype wr = __max(0, __min(1, w_ - left));

			int bot_index_top_left = n*bot_chw + c*bot_hw + top*bot_w + left;
			int bot_index_top_right = n*bot_chw + c*bot_hw + top*bot_w + right;
			int bot_index_bot_left = n*bot_chw + c*bot_hw + bot*bot_w + left;
			int bot_index_bot_right = n*bot_chw + c*bot_hw + bot*bot_w + right;

			caffe_gpu_atomic_add(top_diff[index] * (1 - hr)*(1 - wr), bottom_diff + bot_index_top_left);
			caffe_gpu_atomic_add(top_diff[index] * (1 - hr)*(wr), bottom_diff + bot_index_top_right);
			caffe_gpu_atomic_add(top_diff[index] * (hr)*(1 - wr), bottom_diff + bot_index_bot_left);
			caffe_gpu_atomic_add(top_diff[index] * (hr)*(wr), bottom_diff + bot_index_bot_right);
		}
	}

	template <typename Dtype>
	void ResizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (!propagate_down[0]) {
			return;
		}
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const int count = top[0]->count();

		caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);

		switch (this->layer_param_.resize_param().interpolation())
		{
		case ResizeParameter_InterpolationMethod_NN:
		{
			int bot_chw = bottom[0]->count() / bottom[0]->num();
			int bot_hw = bot_chw / bottom[0]->channels();
			int bot_w = bottom[0]->width();
			int bot_h = bottom[0]->height();

			NNInterporationBackward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
				count, bottom_diff, top_diff, bottom[0]->num(), bottom[0]->channels(), bot_h, bot_w, bot_hw, bot_chw,
				top_height_, top_width_);

			break;
		}
		case ResizeParameter_InterpolationMethod_LINEAR:
		{
			int bot_chw = bottom[0]->count() / bottom[0]->num();
			int bot_hw = bot_chw / bottom[0]->channels();
			int bot_w = bottom[0]->width();
			int bot_h = bottom[0]->height();

			LinearInterporationBackward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
				count, bottom_diff, top_diff, bottom[0]->num(), bottom[0]->channels(), bot_h, bot_w, bot_hw, bot_chw,
				top_height_, top_width_);

			break;
		}
		}
		CUDA_POST_KERNEL_CHECK;
	}


	INSTANTIATE_LAYER_GPU_FUNCS(ResizeLayer);


}  // namespace caffe
