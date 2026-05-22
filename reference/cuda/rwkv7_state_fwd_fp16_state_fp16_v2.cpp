#include <torch/extension.h>
#include "ATen/ATen.h"

typedef at::Half dtype;

void cuda_forward(int B, int T, int C, int H, dtype *state, dtype *r, dtype *w, dtype *k, dtype *v, dtype *a, dtype *b, dtype *y);

void forward(int64_t B, int64_t T, int64_t C, int64_t H,
             torch::Tensor &state,
             torch::Tensor &r, torch::Tensor &w, torch::Tensor &k, torch::Tensor &v,
             torch::Tensor &a, torch::Tensor &b,
             torch::Tensor &y) {
    cuda_forward((int)B, (int)T, (int)C, (int)H,
                 state.data_ptr<dtype>(),
                 r.data_ptr<dtype>(), w.data_ptr<dtype>(), k.data_ptr<dtype>(), v.data_ptr<dtype>(),
                 a.data_ptr<dtype>(), b.data_ptr<dtype>(),
                 y.data_ptr<dtype>());
}

TORCH_LIBRARY(rwkv7_fp16_state_v2, m) {
    m.def("forward", forward);
}
