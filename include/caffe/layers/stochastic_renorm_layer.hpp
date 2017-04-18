#ifndef CAFFE_STOCHASTICRENORM_LAYER_HPP_
#define CAFFE_STOCHASTICRENORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
	template <typename Dtype>
	class StochasticReNormLayer : public Layer<Dtype> {
	public:
		explicit StochasticReNormLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "StochasticReNorm"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		Blob<Dtype> mean_, variance_, temp_, x_norm_, r_, d_;
		bool use_global_stats_;
		Dtype moving_average_fraction_;
		int channels_;
		Dtype eps_;

		Dtype r_max_increase_step_, d_max_increase_step_, mean_rng_, std_rng_;
		int iter_size_,step_to_init_;


		Blob<Dtype> batch_sum_multiplier_;
		Blob<Dtype> num_by_chans_;
		Blob<Dtype> spatial_sum_multiplier_;
	};

}  // namespace caffe

#endif  // CAFFE_STOCHASTICRENORM_LAYER_HPP_
