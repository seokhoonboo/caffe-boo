#include <algorithm>
#include <vector>

#include "caffe/layers/mean_squared_relative_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe {

	template <typename Dtype>
	__global__ void MSR_IGNORE(const int n, const Dtype* bottom, Dtype* diff
		, Dtype min, Dtype max) {

		CUDA_KERNEL_LOOP(index, n) {
			if (min <= bottom[index] && bottom[index] <= max)
			{
				diff[index] = 0;
			}
		}
	}


	template <typename Dtype>
	void MeanSquaredRelativeLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int count = bottom[0]->count();
		const Dtype* bt0 = bottom[0]->gpu_data();
		const Dtype* bt1 = bottom[1]->gpu_data();

		caffe_copy(count, bottom[0]->gpu_data(), bottoms_.mutable_gpu_data());
		caffe_copy(count, bottom[1]->gpu_data(), bottoms_.mutable_gpu_diff());
		caffe_gpu_sub(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), diff_.mutable_gpu_data());

		if (ignore_min <= ignore_max)
		{
			MSR_IGNORE<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			count, bt1, diff_.mutable_gpu_data(), ignore_min, ignore_max);	
		}

		caffe_gpu_div(count, diff_.gpu_data(), bottom[1]->gpu_data(), diff_.mutable_gpu_data());

		Dtype dot;
		caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
		Dtype loss = dot / bottom[0]->num();
		
		top[0]->mutable_cpu_data()[0] = loss;
		CUDA_POST_KERNEL_CHECK;
	}

	template <typename Dtype>
	void MeanSquaredRelativeLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int count = bottom[0]->count();

		Dtype alpha = top[0]->mutable_cpu_diff()[0] * 2 / (bottom[0]->num());
		
		if (propagate_down[0]) {
			caffe_gpu_div(count, diff_.gpu_data(), bottoms_.gpu_diff(), bottom[0]->mutable_gpu_diff());
			caffe_gpu_scal(count, alpha, bottom[0]->mutable_gpu_diff());
		}
		if (propagate_down[1]) {
			if (!propagate_down[0])
			{
				caffe_gpu_div(count, diff_.gpu_data(), bottoms_.gpu_diff(), bottom[1]->mutable_gpu_diff());
				caffe_gpu_scal(count, -alpha, bottom[1]->mutable_gpu_diff());
			}
			else
				caffe_gpu_scale(count, Dtype(-1), bottom[0]->mutable_gpu_diff(), bottom[1]->mutable_gpu_diff());

			caffe_gpu_div(count, bottom[1]->mutable_gpu_diff(), bottoms_.gpu_diff(), bottom[1]->mutable_gpu_diff());
			caffe_gpu_mul(count, bottom[1]->mutable_gpu_diff(), bottoms_.gpu_data(), bottom[1]->mutable_gpu_diff());
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(MeanSquaredRelativeLossLayer);

}  // namespace caffe
