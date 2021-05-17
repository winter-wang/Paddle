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

#pragma once
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/op_compatible_info.h"
#include "paddle/fluid/inference/analysis/argument.h"
#include "paddle/fluid/inference/api/paddle_predictor.h"
#include "paddle/fluid/inference/api/details/reset_tensor_array.h"

#ifdef PADDLE_WITH_TESTING
#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>
#endif

///
/// \file analysis_predictor.h
///
/// \brief Compared to NativePredictor, AnalysisPredictorExp is a high-performance
/// predictor that includes many optimizations
///
/// \author paddle-infer@baidu.com
/// \date 2020-01-01
/// \since 1.7.0
///

namespace paddle {
///
/// \class AnalysisPredictorExp
///
/// \brief The analysis predictor is based on the original native predictor with
/// IR and Analysis support. It will optimize IR and Parameters in the runtime.
///
/// The predictor has the following typical uses:
///
/// Get predictor
/// \code{cpp}
///   auto predictor = CreatePaddlePredictor(config);
/// \endcode
///
/// Get input or output names
/// \code{cpp}
///   auto input_names = predictor->GetInputNames();
///   auto output_names = predictor->GetOutputNames();
/// \endcode
///
/// Get input or output tensors
/// \code{cpp}
///   auto input_t = predictor->GetInputTensor(input_names[0]);
///   auto output_t = predictor->GetOutputTensor(output_names[0]);
/// \endcode
///
/// Run predictor
/// \code{cpp}
///   predictor->ZeroCopyRun();
/// \endcode
///
class AnalysisPredictorExp : public ::paddle_infer::Predictor {
 public:
  ///
  /// \brief Construct a new Analysis Predictor object
  ///
  /// \param[in] paddle_infer::Config config
  ///
  explicit AnalysisPredictorExp(const paddle_infer::Config &config);
  explicit AnalysisPredictorExp(paddle_infer::Config &&config);

  ///
  /// \brief Destroy the Analysis Predictor object
  ///
  ~AnalysisPredictorExp() override;

  ///
  /// \brief Get the input names
  ///
  /// \return input names
  ///
  std::vector<std::string> GetInputNames() const override;

  ///
  /// \brief Get the Input Tensor object
  ///
  /// \param[in] name input name
  /// \return input tensor
  ///
  virtual std::unique_ptr<paddle_infer::Tensor> GetInputHandle(const std::string& name) override;

  ///
  /// \brief Run the prediction engine
  ///
  /// \return Whether the function executed successfully
  ///
  virtual bool Run() override;

  ///
  /// \brief Get the output names
  ///
  /// \return output names
  ///
  virtual std::vector<std::string> GetOutputNames() override;

  ///
  /// \brief Get the Output Tensor object
  ///
  /// \param[in] name otuput name
  /// \return output tensor
  ///
  virtual std::unique_ptr<paddle_infer::Tensor> GetOutputHandle(const std::string& name) override;

  ///
  /// \brief Clone to get the new predictor. thread safe.
  ///
  /// \return get a new predictor
  ///
  virtual std::shared_ptr<paddle_infer::Predictor> Clone() override;

  /// \brief Clear the intermediate tensors of the predictor
  virtual void ClearIntermediateTensor() override;

  ///
  /// \brief Initialize predictor
  ///
  /// Initializing predictor mainly includes the following tasks:
  /// preparing scope, creating executor, preparing program, initializing the
  /// variables required by the executor, getting the feed_target_names and
  /// fetch_target_names, etc.
  ///
  /// \param[in] parent_scope parent scope
  /// \param[in] program program
  /// \return Whether the init function executed successfully
  ///
  bool Init(const std::shared_ptr<framework::Scope> &parent_scope,
            const std::shared_ptr<framework::ProgramDesc> &program = nullptr);

 private:
  
  ///
  /// \brief Prepare predictor's required programs, including loading model
  /// information, graph optimization, and executor creation variables, etc.
  ///
  /// \param[in] program paddle program
  /// \return Whether the function executed successfully
  ///
  bool PrepareProgram(const std::shared_ptr<framework::ProgramDesc> &program);
  ///
  /// \brief Prepare scope environment, each predictor has its own scope
  ///
  /// \param[in] parent_scope The scope of the predictor to be cloned, or null
  /// \return Whether the function executed successfully
  ///
  bool PrepareScope(const std::shared_ptr<framework::Scope> &parent_scope);
  ///
  /// \brief Create an Executor object
  ///
  /// \return Whether the function executed successfully
  ///
  bool CreateExecutor();
  ///
  /// \brief According to the model's program, the executor creates ops
  ///
  /// \return Whether the function executed successfully
  ///
  bool PrepareExecutor();

  ///
  /// \brief Load model program.
  ///
  /// \return Whether the function executed successfully
  ///
  bool LoadProgramDesc();
  ///
  /// \brief Load model parameters.
  ///
  /// \return Whether the function executed successfully
  ///
  bool LoadParameters();
#ifdef PADDLE_WITH_MKLDNN

  ///
  /// \brief PreSet for Mkldnn multi-thread and dynamic shape input.
  ///
  /// Used in AnalysisPredictor::Run(), do not support
  /// AnalysisPredictor::ZeroCopyRun() now.
  ///
  /// \param[in] inputs tensor shape
  ///
  void MkldnnPreSet(const std::vector<std::vector<int>> &inputs_shape);

  ///
  /// \brief PostReset for Mkldnn multi-thread and dynamic shape input.
  ///
  /// Used in AnalysisPredictor::Run(), do not support
  /// AnalysisPredictor::ZeroCopyRun() now.
  ///
  void MkldnnPostReset();
#endif
 private:
  paddle_infer::Config config_;
  inference::analysis::Argument argument_;
  std::unique_ptr<framework::NaiveExecutor> executor_;
  platform::Place place_;
  std::shared_ptr<framework::Scope> scope_;
  framework::Scope *sub_scope_{nullptr};
  std::shared_ptr<framework::ProgramDesc> inference_program_;
  framework::OpCompatibleMap op_compatible_map_;
  std::vector<framework::OpDesc *> feeds_;
  std::map<std::string, size_t> feed_names_;
  // Sorted according to the idx.
  std::map<size_t, std::string> idx2feeds_;

  std::vector<framework::OpDesc *> fetches_;
  std::map<size_t, std::string> idx2fetches_;

#if PADDLE_WITH_MKLDNN
  // Helper class to perform quantization
  class MkldnnQuantizer;
  MkldnnQuantizer *mkldnn_quantizer_{nullptr};

#if PADDLE_WITH_TESTING
  friend class MkldnnQuantizerTest;
#endif
#endif

  // Memory buffer for feed inputs. The temporary LoDTensor will cause serious
  // concurrency problems, wrong results and memory leak, so cache them.
  std::vector<framework::LoDTensor> feed_tensors_;
  details::TensorArrayBatchCleaner tensor_array_batch_cleaner_;
  // A mutex help to make Clone thread safe.
  std::mutex clone_mutex_;

  // For memory optimization.
  const size_t max_shape_collect_count_{1000};
  int need_collect_var_shapes_{-1};  // -1 for default, 0 for false, 1 for true.
  std::vector<std::map<std::string, std::vector<int>>> batch_var_shapes_;
  int predictor_id_;

 private:
  // Some status here that help to determine the status inside the predictor.
  bool status_is_cloned_{false};
};

}  // namespace paddle
