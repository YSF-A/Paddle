#pragma once
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/fc_functor.h"

namespace phi {

// TODO(yinshangfei): add param
template <typename T, typename Context>
void FcKernel(const Context& dev_ctx,
              const DenseTensor& x,
              const DenseTensor& w,
              const paddle::optional<DenseTensor>& bias,
              int in_num_col_dims,
              const std::string& activation_type,
              bool use_mkldnn,
              bool padding_weights,
              bool ALL_KERNELS_MUST_COMPUTE_RUNTIME_SHAPE,
              bool use_quantizer,
              const std::string& mkl_data_type,
              float scale_in,
              const std::vector<float>& scale_weights,
              float scale_out,
              bool force_fp32_output,
              DenseTensor* y) {
  bool with_relu = activation_type == "relu" ? true : false;
  auto w_dims = w.dims();

  auto input_dims = x.dims();
  std::vector<int64_t> output_dims;
  auto in_mat_dims = phi::flatten_to_2d(input_dims, in_num_col_dims);
  auto w_dims0 = padding_weights ? w_dims[0] - 4 : w_dims[0];
  auto w_dims1 = padding_weights ? w_dims[1] - 4 : w_dims[1];
  PADDLE_ENFORCE_EQ(
      in_mat_dims[1],
      w_dims0,
      phi::errors::InvalidArgument(
          "The input's second dimension and weight's first dimension is "
          "expected to be the same. But received input's second dimension is"
          "%d, input's shape is %s; weight's first dimension is %d, weight's"
          " shape is %s.",
          in_mat_dims[1],
          in_mat_dims,
          w_dims0,
          phi::make_ddim({w_dims0, w_dims1})));

  output_dims.reserve(static_cast<size_t>(in_num_col_dims + 1));
  for (int i = 0; i < in_num_col_dims; ++i) {
    output_dims.push_back(in_mat_dims[i]);
  }
  output_dims.push_back(w_dims1);

  y->Resize(phi::make_ddim(output_dims));
  y->set_lod(x.lod());

  auto out_dims = y->dims();
  int M = phi::product(out_dims) / w_dims1;

  const T* input_data = x.data<T>();
  const T* w_data = w.data<T>();
  auto* output_data = dev_ctx.template Alloc<T>(y, y->numel() * sizeof(T));
  auto bias_data = bias ? bias.get_ptr()->data<T>() : NULL;

  phi::funcs::FCFunctor<Context, T> fc;
  fc(dev_ctx,
     M,
     w_dims1,
     w_dims0,
     input_data,
     w_data,
     output_data,
     bias_data,
     with_relu,
     padding_weights);
}

}  // namespace phi
