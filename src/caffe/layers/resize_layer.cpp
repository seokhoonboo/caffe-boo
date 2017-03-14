#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/resize_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void ResizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		ResizeParameter resize_param = this->layer_param_.resize_param();
		top_height_ = resize_param.height();
		top_width_ = resize_param.width();
	}

	template <typename Dtype>
	void ResizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
			<< "corresponding to (num, channels, height, width)";
	
		top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
			top_height_, top_width_);
	}

	// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
	// case?
	template <typename Dtype>
	void ResizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();

		switch (this->layer_param_.resize_param().interpolation())
		{
		case ResizeParameter_InterpolationMethod_NN:
		{
			int top_index = 0;
			int bot_chw = bottom[0]->count() / bottom[0]->num();
			int bot_hw = bot_chw / bottom[0]->channels();
			int bot_w = bottom[0]->width();
			int bot_h = bottom[0]->height();
			Dtype bot_hDtop_h = (Dtype)bot_h / top_height_;
			Dtype bot_wDtop_w = (Dtype)bot_w / top_width_;
			for (int n = 0; n < bottom[0]->num(); ++n)
			{
				for (int c = 0; c < bottom[0]->channels(); ++c)
				{
					for (int ph = 0; ph < top_height_; ++ph)
					{
						for (int pw = 0; pw < top_width_; ++pw)
						{
							int h_ = __max(0, __min(bot_h - 1, int(bot_hDtop_h*(ph + 1) - 0.5)));
							int w_ = __max(0, __min(bot_w - 1, int(bot_wDtop_w*(pw + 1) - 0.5)));
							int bot_index = n*bot_chw + c*bot_hw + h_*bot_w + w_;

							top_data[top_index++] = bottom_data[bot_index];
						}
					}
				}
			}
			break;
		}
		case ResizeParameter_InterpolationMethod_LINEAR:
		{
			int top_index = 0;
			int bot_chw = bottom[0]->count() / bottom[0]->num();
			int bot_hw = bot_chw / bottom[0]->channels();
			int bot_w = bottom[0]->width();
			int bot_h = bottom[0]->height();
			Dtype bot_hDtop_h = (Dtype)bot_h / top_height_;
			Dtype bot_wDtop_w = (Dtype)bot_w / top_width_;
			for (int n = 0; n < bottom[0]->num(); ++n)
			{
				for (int c = 0; c < bottom[0]->channels(); ++c)
				{
					for (int ph = 0; ph < top_height_; ++ph)
					{
						for (int pw = 0; pw < top_width_; ++pw)
						{
							Dtype h_ = __max(0, __min(bot_h - 1, bot_hDtop_h*(ph + 1) - 1));
							Dtype w_ = __max(0, __min(bot_w - 1, bot_wDtop_w*(pw + 1) - 1));
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

							top_data[top_index++] = bottom_data[bot_index_top_left] * (1 - hr)*(1 - wr)
								+ bottom_data[bot_index_top_right] * (1 - hr)*(wr)
								+bottom_data[bot_index_bot_left] * (hr)*(1 - wr)
								+ bottom_data[bot_index_bot_right] * (hr)*(wr);
						}
					}
				}
			}
			break;
		}
		}

	}

	template <typename Dtype>
	void ResizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (!propagate_down[0]) {
			return;
		}
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
		
		switch (this->layer_param_.resize_param().interpolation())
		{
		case ResizeParameter_InterpolationMethod_NN:
		{
			int top_index = 0;
			int bot_chw = bottom[0]->count() / bottom[0]->num();
			int bot_hw = bot_chw / bottom[0]->channels();
			int bot_w = bottom[0]->width();
			int bot_h = bottom[0]->height();
			Dtype bot_hDtop_h = (Dtype)bot_h / top_height_;
			Dtype bot_wDtop_w = (Dtype)bot_w / top_width_;
			for (int n = 0; n < bottom[0]->num(); ++n)
			{
				for (int c = 0; c < bottom[0]->channels(); ++c)
				{
					for (int ph = 0; ph < top_height_; ++ph)
					{
						for (int pw = 0; pw < top_width_; ++pw)
						{
							int h_ = __max(0, __min(bot_h - 1, int(bot_hDtop_h*(ph + 1) - 0.5)));
							int w_ = __max(0, __min(bot_w - 1, int(bot_wDtop_w*(pw + 1) - 0.5)));
							int bot_index = n*bot_chw + c*bot_hw + h_*bot_w + w_;

							bottom_diff[bot_index] += top_diff[top_index++];
						}
					}
				}
			}
			break;
		}
		case ResizeParameter_InterpolationMethod_LINEAR:
		{
			int top_index = 0;
			int bot_chw = bottom[0]->count() / bottom[0]->num();
			int bot_hw = bot_chw / bottom[0]->channels();
			int bot_w = bottom[0]->width();
			int bot_h = bottom[0]->height();
			Dtype bot_hDtop_h = (Dtype)bot_h / top_height_;
			Dtype bot_wDtop_w = (Dtype)bot_w / top_width_;
			for (int n = 0; n < bottom[0]->num(); ++n)
			{
				for (int c = 0; c < bottom[0]->channels(); ++c)
				{
					for (int ph = 0; ph < top_height_; ++ph)
					{
						for (int pw = 0; pw < top_width_; ++pw)
						{
							Dtype h_ = __max(0, __min(bot_h - 1, bot_hDtop_h*(ph + 1) - 1));
							Dtype w_ = __max(0, __min(bot_w - 1, bot_wDtop_w*(pw + 1) - 1));
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

							bottom_diff[bot_index_top_left] += top_diff[top_index]*(1 - hr)*(1 - wr);
							bottom_diff[bot_index_top_right] += top_diff[top_index] *(1 - hr)*(wr);
							bottom_diff[bot_index_bot_left] += top_diff[top_index] * (hr)*(1 - wr);
							bottom_diff[bot_index_bot_right] += top_diff[top_index++] * (hr)*(wr);
						}
					}
				}
			}
			break;
		}
		}
		
	}


#ifdef CPU_ONLY
	STUB_GPU(ResizeLayer);
#endif

	INSTANTIATE_CLASS(ResizeLayer);
	REGISTER_LAYER_CLASS(Resize);
}  // namespace caffe
