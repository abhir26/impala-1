// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#ifndef IMPALA_EXPRS_AI_FUNCTIONS_H
#define IMPALA_EXPRS_AI_FUNCTIONS_H

#include <string.h>

#include "udf/udf.h"

using namespace impala_udf;

namespace impala {

using impala_udf::FunctionContext;
using impala_udf::StringVal;

class AiFunctions {
 public:
  static const string AI_GENERATE_TXT_JSON_PARSE_ERROR;
  static const string AI_GENERATE_TXT_INVALID_PROTOCOL_ERROR;
  static const string AI_GENERATE_TXT_UNSUPPORTED_ENDPOINT_ERROR;
  static const string AI_GENERATE_TXT_INVALID_PROMPT_ERROR;
  static const string AI_GENERATE_TXT_MSG_OVERRIDE_FORBIDDEN_ERROR;
  /// Sends a prompt to the input AI endpoint using the input model, api_key and
  /// optional params.
  static StringVal AiGenerateText(FunctionContext* ctx, const StringVal& endpoint,
      const StringVal& prompt, const StringVal& model,
      const StringVal& api_key_jceks_secret, const StringVal& params);
  /// Sends a prompt to the default endpoint and uses the default model, default
  /// api-key and default params.
  static StringVal AiGenerateText(FunctionContext* ctx, const StringVal& prompt);
  static StringVal AiGenerateTextDummy(FunctionContext* ctx, const StringVal& prompt);
  /// Set the ai_api_key_ member.
  static void set_api_key(string& api_key) { ai_api_key_ = api_key; }
  /// Validate api end point.
  static bool is_api_endpoint_valid(const string& endpoint);
  /// Check if endpoint is supported
  static bool is_api_endpoint_supported(const string& endpoint);

 private:
  /// The default api_key used for communicating with external APIs.
  static std::string ai_api_key_;
  /// Internal function which implements the logic of parsing user input and sending
  /// request to the external API endpoint. If 'dry_run' is set, the POST request is
  /// returned. 'dry_run' mode is used only for unit tests.
  static StringVal AiGenerateTextInternal(FunctionContext* ctx, const StringVal& endpoint,
      const StringVal& prompt, const StringVal& model,
      const StringVal& api_key_jceks_secret, const StringVal& params, const bool dry_run);

  friend class ExprTest_AiFunctionsTest_Test;
};
} // namespace impala
#endif
