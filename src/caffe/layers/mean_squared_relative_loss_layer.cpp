#include <vector>

#include "caffe/layers/mean_squared_relative_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void MeanSquaredRelativeLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		if (this->layer_param_.loss_weight_size() == 0) {
			this->layer_param_.add_loss_weight(Dtype(1));
		}

		ignore_min = this->layer_param_.mean_squared_relative_loss_param().ignore_value_min();
		ignore_max = this->layer_param_.mean_squared_relative_loss_param().ignore_value_max();
	}

	template <typename Dtype>
	void MeanSquaredRelativeLossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::Reshape(bottom, top);
		CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
			<< "Inputs must have the same dimension.";
		diff_.ReshapeLike(*bottom[0]);
		bottoms_.ReshapeLike(*bottom[0]);
	}

	template <typename Dtype>
	void MeanSquaredRelativeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int count = bottom[0]->count();
		const Dtype* bt0 = bottom[0]->cpu_data();
		const Dtype* bt1 = bottom[1]->cpu_data();

		caffe_copy(count, bottom[0]->cpu_data(), bottoms_.mutable_cpu_data());
		caffe_copy(count, bottom[1]->cpu_data(), bottoms_.mutable_cpu_diff());
		caffe_sub(count,bottom[0]->cpu_data(),bottom[1]->cpu_data(),diff_.mutable_cpu_data());

		if (ignore_min <= ignore_max)
		{
			for (int i = 0; i < count; i++)
			{
				if (ignore_min <= bt1[i] && bt1[i] <= ignore_max)
				{
					diff_.mutable_cpu_data()[i] = 0;
				}
			}
		}

		caffe_div(count, diff_.cpu_data(),bottom[1]->cpu_data(),diff_.mutable_cpu_data());
		
		Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
		Dtype loss = dot / bottom[0]->num();

		top[0]->mutable_cpu_data()[0] = loss;
	}

	template <typename Dtype>
	void MeanSquaredRelativeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int count = bottom[0]->count();

		Dtype alpha = top[0]->mutable_cpu_diff()[0] * 2 / (bottom[0]->num());
		
		if (propagate_down[0]) {
			caffe_div(count, diff_.cpu_data(), bottoms_.cpu_diff(), bottom[0]->mutable_cpu_diff());
			caffe_scal(count, alpha, bottom[0]->mutable_cpu_diff());
		}
		if (propagate_down[1]) {
			if (!propagate_down[0])
			{
				caffe_div(count, diff_.cpu_data(), bottoms_.cpu_diff(), bottom[1]->mutable_cpu_diff());
				caffe_scal(count, -alpha, bottom[1]->mutable_cpu_diff());
			}
			else
				caffe_cpu_scale(count, Dtype(-1),bottom[0]->cpu_diff(), bottom[1]->mutable_cpu_diff());
			
			caffe_div(count, bottom[1]->mutable_cpu_diff(), bottoms_.cpu_diff(), bottom[1]->mutable_cpu_diff());
			caffe_mul(count, bottom[1]->mutable_cpu_diff(), bottoms_.cpu_data(), bottom[1]->mutable_cpu_diff());
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(MeanSquaredRelativeLossLayer);
#endif

	INSTANTIATE_CLASS(MeanSquaredRelativeLossLayer);
	REGISTER_LAYER_CLASS(MeanSquaredRelativeLoss);

}  // namespace caffe
