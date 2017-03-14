#ifndef CAFFE_MEAN_SQUARED_RELATIVE_LOSS_LAYER_HPP_
#define CAFFE_MEAN_SQUARED_RELATIVE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {


	template <typename Dtype>
	class MeanSquaredRelativeLossLayer : public LossLayer<Dtype> {
	public:
		explicit MeanSquaredRelativeLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param), diff_() {}

		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "MeanSquaredRelativeLoss"; }
		
		virtual inline bool AllowForceBackward(const int bottom_index) const {
			return true;
		}

	protected:
		/// @copydoc EuclideanLossLayer
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);


		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		Blob<Dtype> diff_, bottoms_;
		float ignore_min;
		float ignore_max;
	};

}  // namespace caffe

#endif  // CAFFE_MEAN_SQUARED_RELATIVE_LOSS_LAYER_HPP_
