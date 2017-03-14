#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe {

	template <typename Dtype>
	__global__ void EUC_IGNORE(const int n, const Dtype* bottom, Dtype* diff
		, Dtype min, Dtype max) {

		CUDA_KERNEL_LOOP(index, n) {
			if (min <= bottom[index] && bottom[index] <= max)
			{
				diff[index] = 0;
			}
		}
	}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	int count = bottom[0]->count();
	const Dtype* bt0 = bottom[0]->gpu_data();
	const Dtype* bt1 = bottom[1]->gpu_data();

	caffe_gpu_sub(
		count,
		bt0,
		bt1,
		diff_.mutable_gpu_data());

	if (ignore_min <= ignore_max)
	{
		EUC_IGNORE<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
		count, bt1, diff_.mutable_gpu_data(), ignore_min, ignore_max);
	}

	Dtype dot;
	caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
	Dtype loss = dot / bottom[0]->num() / Dtype(2);

	top[0]->mutable_cpu_data()[0] = loss;
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	for (int i = 0; i < 2; ++i) {
		if (propagate_down[i]) {
			const Dtype sign = (i == 0) ? 1 : -1;
			Dtype alpha = sign * top[0]->cpu_diff()[0] / (bottom[i]->num());
			caffe_gpu_axpby(
				bottom[i]->count(),              // count
				alpha,                              // alpha
				diff_.gpu_data(),                   // a
				Dtype(0),                           // beta
				bottom[i]->mutable_gpu_diff());  // b
		}
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossLayer);

}  // namespace caffe
