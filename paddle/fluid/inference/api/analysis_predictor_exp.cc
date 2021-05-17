// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/inference/api/analysis_predictor_exp.h"
#include <glog/logging.h>
#include <algorithm>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/extension/include/ext_op_meta_info.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/framework/version.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/analysis/passes/memory_optimize_pass.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"

#ifdef PADDLE_WITH_MKLML
#include "paddle/fluid/platform/dynload/mklml.h"
#endif

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/inference/api/mkldnn_quantizer.h"
#endif

#if PADDLE_WITH_TENSORRT
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/trt_int8_calibrator.h"
#endif

namespace paddle {

namespace {
bool IsPersistable(const framework::VarDesc *var) {
  if (var->Persistable() &&
      var->GetType() != framework::proto::VarType::FEED_MINIBATCH &&
      var->GetType() != framework::proto::VarType::FETCH_LIST &&
      var->GetType() != framework::proto::VarType::RAW) {
    return true;
  }
  return false;
}
}  // namespace

AnalysisPredictorExp::AnalysisPredictorExp(const paddle_infer::Config &config):config_(config) {}

AnalysisPredictorExp::AnalysisPredictorExp(paddle_infer::Config &&config):config_(std::move(config)) {}

std::vector<std::string> AnalysisPredictorExp::GetInputNames() const {
  std::vector<std::string> input_names;
  for (auto &item : idx2feeds_) {
    input_names.emplace_back(item.second);
  }
  return input_names;
}

std::unique_ptr<paddle_infer::Tensor> AnalysisPredictorExp::GetInputHandle(const std::string& name) {
  PADDLE_ENFORCE_NOT_NULL(
      executor_->scope()->FindVar(name),
      platform::errors::PreconditionNotMet(
          "The variable named %s is not found in the scope of the exector.",
          name));
  std::unique_ptr<ZeroCopyTensor> res(
      new ZeroCopyTensor(static_cast<void *>(executor_->scope())));
  res->input_or_output_ = true;
  res->SetName(name);
  if (platform::is_cpu_place(place_)) {
    res->SetPlace(PaddlePlace::kCPU);
  } else if (platform::is_xpu_place(place_)) {
    if (config_.lite_engine_enabled()) {
      // Currently, Paddle-Lite's XPU user interface only supports the transfer
      // of host data pointers. If it is currently used as a subgraph, execution
      // efficiency will be sacrificed, so it is temporarily set to cpu place.
      // And, the current lite engine of xpu must execute all parts of the
      // model.
      res->SetPlace(PaddlePlace::kCPU);
    } else {
      auto xpu_place = BOOST_GET_CONST(platform::XPUPlace, place_);
      res->SetPlace(PaddlePlace::kXPU, xpu_place.GetDeviceId());
    }
  } else {
    auto gpu_place = BOOST_GET_CONST(platform::CUDAPlace, place_);
    res->SetPlace(PaddlePlace::kGPU, gpu_place.GetDeviceId());
  }
  return res;
}
bool AnalysisPredictorExp::Run() {
    paddle::platform::SetNumThreads(config_.cpu_math_library_num_threads());
#ifdef PADDLE_WITH_MKLDNN
  if (config_.use_mkldnn_) {
    std::vector<std::vector<int>> shape_vector;
    auto names = GetInputNames();
    for (size_t i = 0; i < names.size(); ++i) {
      auto in_tensor = GetInputHandle(names[i]);
      shape_vector.emplace_back(in_tensor->shape());
    }
    MkldnnPreSet(shape_vector);
  }
#endif

  executor_->Run();
  // Fix TensorArray reuse not cleaned bug.
  tensor_array_batch_cleaner_.CollectTensorArrays(sub_scope_);
  tensor_array_batch_cleaner_.ResetTensorArray();

  // recover the cpu_math_library_num_threads to 1, in order to avoid thread
  // conflict when integrating it into deployment service.
  paddle::platform::SetNumThreads(1);
#ifdef PADDLE_WITH_MKLDNN
  if (config_.use_mkldnn_) MkldnnPostReset();
#endif
#if defined(PADDLE_WITH_MKLML)
  // Frees unused memory allocated by the Intel® MKL Memory Allocator to
  // avoid memory leak. See:
  // https://software.intel.com/en-us/mkl-developer-reference-c-mkl-free-buffers
  platform::dynload::MKL_Free_Buffers();
#endif
  return true;
}

std::vector<std::string> AnalysisPredictorExp::GetOutputNames() const {
  std::vector<std::string> output_names;
  for (auto &item : idx2fetches_) {
    output_names.emplace_back(item.second);
  }
  return output_names;
}

std::unique_ptr<paddle_infer::Tensor> AnalysisPredictorExp::GetOutputHandle(
    const std::string &name) {
  PADDLE_ENFORCE_NOT_NULL(
      executor_->scope()->FindVar(name),
      platform::errors::PreconditionNotMet(
          "he variable named %s is not found in the scope of the exector.",
          name));
  std::unique_ptr<ZeroCopyTensor> res(
      new ZeroCopyTensor(static_cast<void *>(executor_->scope())));
  res->input_or_output_ = false;
  res->SetName(name);
  if (platform::is_cpu_place(place_)) {
    res->SetPlace(PaddlePlace::kCPU);
  } else if (platform::is_xpu_place(place_)) {
    if (config_.lite_engine_enabled()) {
      // Currently, Paddle-Lite's XPU user interface only supports the transfer
      // of host data pointers. If it is currently used as a subgraph, execution
      // efficiency will be sacrificed, so it is temporarily set to cpu place.
      // And, the current lite engine of xpu must execute all parts of the
      // model.
      res->SetPlace(PaddlePlace::kCPU);
    } else {
      auto xpu_place = BOOST_GET_CONST(platform::XPUPlace, place_);
      res->SetPlace(PaddlePlace::kXPU, xpu_place.GetDeviceId());
    }
  } else {
    auto gpu_place = BOOST_GET_CONST(platform::CUDAPlace, place_);
    res->SetPlace(PaddlePlace::kGPU, gpu_place.GetDeviceId());
  }
  return res;
}
std::shared_ptr<paddle_infer::Predictor> AnalysisPredictorExp::Clone() {
  std::lock_guard<std::mutex> lk(clone_mutex_);
  auto *x = new AnalysisPredictorExp(config_);
  x->Init(scope_, inference_program_);
  return std::shared_ptr<paddle_infer::Predictor>(x);
}

void AnalysisPredictorExp::ClearIntermediateTensor() {
  PADDLE_ENFORCE_NOT_NULL(inference_program_.get(),
                          platform::errors::PreconditionNotMet(
                              "The inference program should be loaded first."));
  const auto &global_block = inference_program_->MutableBlock(0);
  for (auto *var : global_block->AllVars()) {
    if (!IsPersistable(var)) {
      const std::string name = var->Name();
      auto *variable = executor_->scope()->FindVar(name);
      if (variable != nullptr && variable->IsType<framework::LoDTensor>() &&
          name != "feed" && name != "fetch") {
        VLOG(3) << "Clear Intermediate Tensor: " << name;
        auto *t = variable->GetMutable<framework::LoDTensor>();
        t->clear();
      }
    }
  }
}

bool AnalysisPredictorExp::Init(
    const std::shared_ptr<framework::Scope> &parent_scope,
    const std::shared_ptr<framework::ProgramDesc> &program) {
  VLOG(3) << "Predictor::init()";
  if (config_.with_profile_) {
    LOG(WARNING) << "Profiler is activated, which might affect the performance";
    auto tracking_device = config_.use_gpu() ? platform::ProfilerState::kAll
                                             : platform::ProfilerState::kCPU;
    platform::EnableProfiler(tracking_device);
  } else {
    LOG(INFO) << "Profiler is deactivated, and no profiling report will be "
                 "generated.";
  }

  // no matter with or without MKLDNN
  paddle::platform::SetNumThreads(config_.cpu_math_library_num_threads());

  if (!PrepareScope(parent_scope)) {
    return false;
  }
  if (!CreateExecutor()) {
    return false;
  }
  if (!PrepareProgram(program)) {
    return false;
  }

  // Prepare executor, create local variables.
  if (!PrepareExecutor()) {
    return true;
  }

  // Get the feed_target_names and fetch_target_names
  PrepareFeedFetch();

  return true;
}

bool AnalysisPredictor::PrepareScope(
    const std::shared_ptr<framework::Scope> &parent_scope) {
  if (parent_scope) {
    PADDLE_ENFORCE_NOT_NULL(
        parent_scope,
        platform::errors::PreconditionNotMet(
            "Both program and parent_scope should be set in Clone mode."));
    scope_ = parent_scope;
    status_is_cloned_ = true;
  } else {
    paddle::framework::InitDevices();
    scope_.reset(new paddle::framework::Scope(), [](framework::Scope *scope) {
      delete scope;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      for (int dev_id = 0; dev_id < paddle::platform::GetCUDADeviceCount();
           ++dev_id) {
        memory::Release(platform::CUDAPlace(dev_id));
      }
#endif
#ifdef PADDLE_WITH_XPU
      for (int dev_id = 0; dev_id < paddle::platform::GetXPUDeviceCount();
           ++dev_id) {
        memory::Release(platform::XPUPlace(dev_id));
      }
#endif
      memory::Release(platform::CPUPlace());
    });
    status_is_cloned_ = false;
  }
  sub_scope_ = &scope_->NewScope();
  return true;
}
bool AnalysisPredictor::PrepareProgram(
    const std::shared_ptr<framework::ProgramDesc> &program) {
  if (!program) {
    if (!LoadProgramDesc()) return false;
    // If not cloned, the parameters should be loaded.
    // If config_.ir_optim() is True, parameters is loaded in
    // OptimizeInferenceProgram(), but other persistable variables
    // (like RAW type var) are not created in scope.
    // If config_.ir_optim() is False, parameters is loaded in LoadParameters(),
    // still need to create other persistable variables.
    // So in both case, create persistable variables at first.
    executor_->CreateVariables(*inference_program_, 0, true, sub_scope_);

    // if enable_ir_optim_ is false,
    // the analysis pass(op fuse, graph analysis, trt subgraph, mkldnn etc) will
    // not be executed.
    OptimizeInferenceProgram();
  } else {
    // If the program is passed from external, no need to optimize it, this
    // logic is used in the clone scenario.
    inference_program_ = program;
  }

  executor_->CreateVariables(*inference_program_, 0, false, sub_scope_);

  return true;
}
bool AnalysisPredictor::CreateExecutor() {
  if (config_.use_gpu()) {
    PADDLE_ENFORCE_EQ(config_.use_xpu(), false,
                      platform::errors::InvalidArgument(
                          "Only one choice can be made between CPU and XPU."));
    place_ = paddle::platform::CUDAPlace(config_.gpu_device_id());
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (config_.thread_local_stream_enabled()) {
      auto *ctx = static_cast<platform::CUDADeviceContext *>(
          platform::DeviceContextPool::Instance().Get(place_));
      VLOG(3) << "The prediction process will be completed using a separate "
                 "normal-priority stream on each thread.";
      ctx->ResetThreadContext(platform::stream::Priority::kNormal);
    }
#endif
  } else if (config_.use_xpu()) {
    if (config_.lite_engine_enabled()) {
#ifdef LITE_SUBGRAPH_WITH_XPU
      // Currently, Paddle-Lite's XPU user interface only supports the transfer
      // of Host data pointers. If it is currently used as a subgraph, execution
      // efficiency will be sacrificed, so it is temporarily set to cpu place.
      // And, the current lite engine of xpu must execute all parts of the
      // model.
      place_ = paddle::platform::CPUPlace();
#else
      PADDLE_THROW(platform::errors::Unavailable(
          "You tried to use an XPU lite engine, but Paddle was not compiled "
          "with it."));
#endif  // LITE_SUBGRAPH_WITH_XPU
    } else {
#ifdef PADDLE_WITH_XPU
      place_ = paddle::platform::XPUPlace(config_.xpu_device_id());
#else
      PADDLE_THROW(platform::errors::Unavailable(
          "You tried to use XPU forward propagation (inference without lite "
          "engine), but Paddle was not compiled "
          "with WITH_XPU."));
#endif  // PADDLE_WITH_XPU
    }
  } else {
    place_ = paddle::platform::CPUPlace();
  }
  executor_.reset(new paddle::framework::NaiveExecutor(place_));
  return true;
}
bool AnalysisPredictor::PrepareExecutor() {
  executor_->Prepare(sub_scope_, *inference_program_, 0,
                     config_.use_feed_fetch_ops_);

  PADDLE_ENFORCE_NOT_NULL(sub_scope_,
                          platform::errors::PreconditionNotMet(
                              "The sub_scope should not be nullptr."));

  return true;
}
#ifdef PADDLE_WITH_MKLDNN

void AnalysisPredictorExp::MkldnnPreSet(const std::vector<std::vector<int>> &inputs_shape) {
  VLOG(2) << "AnalysisPredictor::ZeroCopyRun get_cur_mkldnn_session_id="
          << platform::MKLDNNDeviceContext::tls().get_cur_mkldnn_session_id();
  // In cache clearing mode.
  if (config_.mkldnn_cache_capacity_ > 0) {
    VLOG(2) << "In mkldnn cache clear mode.";
    platform::MKLDNNDeviceContext::tls().set_cur_mkldnn_session_id(
        platform::MKLDNNDeviceContextThreadLocals::
            kMKLDNNSessionID_CacheClearing);
    platform::MKLDNNDeviceContext::tls().set_cur_input_shape_cache_capacity(
        config_.mkldnn_cache_capacity_);
    // Set current_input_shape for caching dynamic shape.
    std::stringstream ss;
    for (size_t i = 0; i < inputs_shape.size(); ++i) {
      for (size_t j = 0; j < inputs_shape[i].size(); ++j) {
        ss << inputs_shape[i][j] << "-";
      }
    }
    VLOG(2) << "Set input shape=" << ss.str();
    platform::MKLDNNDeviceContext::tls().set_cur_input_shape_str(ss.str());
  }
}

void AnalysisPredictorExp::MkldnnPostReset() {
  // In cache clearing mode.
  if (config_.mkldnn_cache_capacity_ > 0) {
    if (VLOG_IS_ON(2)) {
      auto shape_blob_size = static_cast<platform::MKLDNNDeviceContext *>(
                                 (&platform::DeviceContextPool::Instance())
                                     ->Get(platform::CPUPlace()))
                                 ->GetShapeBlobSize();
      CHECK_LE(shape_blob_size,
               static_cast<size_t>(config_.mkldnn_cache_capacity_));
    }
    paddle::platform::MKLDNNDeviceContext::tls().set_cur_mkldnn_session_id(
        platform::MKLDNNDeviceContextThreadLocals::kMKLDNNSessionID_Default);
    platform::MKLDNNDeviceContext::tls().set_cur_input_shape_cache_capacity(0);
    platform::MKLDNNDeviceContext::tls().set_cur_input_shape_str("");
  }
}

#endif

} // namespace paddle