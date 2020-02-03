// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <torch/extension.h>

template <typename scalar_t>
at::Tensor nms_cpu_kernel(const at::Tensor& dets, const float threshold, const float c_thres) {
  AT_ASSERTM(!dets.type().is_cuda(), "dets must be a CPU tensor");

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  }

  auto x1_t = dets.select(1, 0).contiguous();
  auto y1_t = dets.select(1, 1).contiguous();
  auto x2_t = dets.select(1, 2).contiguous();
  auto y2_t = dets.select(1, 3).contiguous();
  auto scores = dets.select(1, 4).contiguous();
  // conf
  auto x1_c_t = dets.select(1, 5).contiguous();
  auto y1_c_t = dets.select(1, 6).contiguous();
  auto x2_c_t = dets.select(1, 7).contiguous();
  auto y2_c_t = dets.select(1, 8).contiguous();
  

  at::Tensor areas_t = (x2_t - x1_t + 1) * (y2_t - y1_t + 1);

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto ndets = dets.size(0);
  at::Tensor suppressed_t =
      at::zeros({ndets}, dets.options().dtype(at::kByte).device(at::kCPU));
  at::Tensor conf_t =
      at::zeros({ndets * 5}, dets.options().dtype(at::kLong).device(at::kCPU));

  auto suppressed = suppressed_t.data<uint8_t>();
  auto conf = conf_t.data<int64_t>();
  auto order = order_t.data<int64_t>();
  auto x1 = x1_t.data<scalar_t>();
  auto y1 = y1_t.data<scalar_t>();
  auto x2 = x2_t.data<scalar_t>();
  auto y2 = y2_t.data<scalar_t>();

  auto x1_c = x1_c_t.data<scalar_t>();
  auto y1_c = y1_c_t.data<scalar_t>();
  auto x2_c = x2_c_t.data<scalar_t>();
  auto y2_c = y2_c_t.data<scalar_t>();

  auto areas = areas_t.data<scalar_t>();

  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    
    if (suppressed[i] == 1) continue;

    // conf[i*5 + 0] = i + 1;
    // conf[i*5 + 1] = i + 1;
    // conf[i*5 + 2] = i + 1;
    // conf[i*5 + 3] = i + 1;
    // conf[i*5 + 4] = i + 1;

    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];

    auto ixc1 = x1_c[i];
    auto iyc1 = y1_c[i];
    auto ixc2 = x2_c[i];
    auto iyc2 = y2_c[i];

    auto iarea = areas[i];

    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1) continue;

      // conf[j*5 + 0] = j + 1;
      // conf[j*5 + 1] = j + 1;
      // conf[j*5 + 2] = j + 1;
      // conf[j*5 + 3] = j + 1;
      // conf[j*5 + 4] = j + 1;

      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);

      auto jxc1 = x1_c[j];
      auto jyc1 = y1_c[j];
      auto jxc2 = x2_c[j];
      auto jyc2 = y2_c[j];

      auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1 + 1);
      auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1 + 1);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[j] - inter);

      if (ovr >= threshold) {
        suppressed[j] = 1;
        
        // conf[j*5 + 0] = 0;
        // conf[j*5 + 1] = 0;
        // conf[j*5 + 2] = 0;
        // conf[j*5 + 3] = 0;
        // conf[j*5 + 4] = 0;

        // hard coded
        if (ovr >= c_thres) {
          if (ixc1 >= jxc1) {
            conf[i*5 + 0] = i + 1;
          }
          else {
            conf[i*5 + 0] = j + 1;
            x1_c[i] = jxc1;
          }

          if (iyc1 >= jyc1) {
            conf[i*5 + 1] = i + 1;
          }
          else { 
            conf[i*5 + 1] = j + 1;
            y1_c[i] = jyc1;
          }
          
          if (ixc2 >= jxc2) {
            conf[i*5 + 2] = i + 1;
          }
          else {
            conf[i*5 + 2] = j + 1;
            x2_c[i] = jxc2;
          }

          if (iyc2 >= jyc2) {
            conf[i*5 + 3] = i + 1;
          }
          else {
            conf[i*5 + 3] = j + 1;
            y2_c[i] = jyc2;
          }
        }
        else {
          conf[i*5 + 0] = i + 1;
          conf[i*5 + 1] = i + 1;
          conf[i*5 + 2] = i + 1;
          conf[i*5 + 3] = i + 1;
        }
        // score should not be changed
        conf[i*5 + 4] = i + 1;

      }
    }
  }
  return conf_t;
  // return at::nonzero(suppressed_t == 0).squeeze(1);//, conf_t;
}

at::Tensor nms_v2(const at::Tensor& dets, const float threshold, const float c_thres) {
  at::Tensor result;
  AT_DISPATCH_FLOATING_TYPES(dets.type(), "nms_v2", [&] {
    result = nms_cpu_kernel<scalar_t>(dets, threshold, c_thres);
  });
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms_v2", &nms_v2, "non-maximum suppression");
}