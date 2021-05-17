/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

/*
 * This file contains the definition of a simple Inference API for Paddle.
 *
 * ATTENTION: It requires some C++11 features, for lower version C++ or C, we
 * might release another API.
 */

#pragma once

#include <cassert>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle_config.h"  // NOLINT

///
/// \file paddle_predictor.h
///
/// \brief Paddle Inference API
///
/// \author paddle-infer@baidu.com
/// \date 2021-05-15
/// \since 2.2
///

namespace paddle_infer {

class Tensor;

///
/// \class Predictor
///
/// \brief Predictor is the interface for model prediction.
///
/// The predictor has the following typical uses:
///
/// Get predictor
/// \code{cpp}
///   auto predictor = CreatePredictor(config);
/// \endcode
///
/// Get input or output names
/// \code{cpp}
///   auto input_names = predictor->GetInputNames();
///   auto output_names = predictor->GetOutputNames();
/// \endcode
///
/// Get input or output handle
/// \code{cpp}
///   auto input_t = predictor->GetInputHandle(input_names[0]);
///   auto output_t = predictor->GetOutputHandle(output_names[0]);
/// \endcode
///
/// Run predictor
/// \code{cpp}
///   predictor->Run();
/// \endcode
///
class PD_INFER_DECL Predictor {
 public:
  ///
  /// \brief Virtual deconstruction function 
  ///
  /// \return input names
  ///
  virtual ~Predictor() = default;
  ///
  /// \brief Get the input names
  ///
  /// \return input names
  ///
  virtual std::vector<std::string> GetInputNames() const = 0;

  ///
  /// \brief Get the Input Tensor object
  ///
  /// \param[in] name input name
  /// \return input tensor
  ///
  virtual std::unique_ptr<Tensor> GetInputHandle(const std::string& name) = 0;

  ///
  /// \brief Run the prediction engine
  ///
  /// \return Whether the function executed successfully
  ///
  virtual bool Run() = 0;

  ///
  /// \brief Get the output names
  ///
  /// \return output names
  ///
  virtual std::vector<std::string> GetOutputNames() const = 0;

  ///
  /// \brief Get the Output Tensor object
  ///
  /// \param[in] name otuput name
  /// \return output tensor
  ///
  virtual std::unique_ptr<Tensor> GetOutputHandle(const std::string& name) = 0;

  ///
  /// \brief Clone to get the new predictor. thread safe.
  ///
  /// \return get a new predictor
  ///
  virtual std::shared_ptr<Predictor> Clone() = 0;

  /// \brief Clear the intermediate tensors of the predictor
  virtual void ClearIntermediateTensor() = 0;

  ///
  /// \brief Release all tmp tensor to compress the size of the memory pool.
  /// The memory pool is considered to be composed of a list of chunks, if
  /// the chunk is not occupied, it can be released.
  ///
  /// \return Number of bytes released. It may be smaller than the actual
  /// released memory, because part of the memory is not managed by the
  /// MemoryPool.
  ///
  virtual uint64_t TryShrinkMemory() = 0; 
};

///
/// \brief A factory to help create predictors.
///
/// Usage:
///
/// \code{.cpp}
/// Config config;
/// ... // change the configs.
/// auto predictor = CreatePredictor(config);
/// \endcode
///
PD_INFER_DECL std::shared_ptr<Predictor> CreatePredictor(
    const Config& config);  // NOLINT

PD_INFER_DECL std::shared_ptr<Predictor> CreatePredictor(
    Config&& config);  // NOLINT

} // paddle_innfer