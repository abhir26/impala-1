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

// The functions in this file are specifically not cross-compiled to IR because there
// is no signifcant performance benefit to be gained.

#include <gutil/strings/util.h>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "common/compiler-util.h"
#include "exprs/ai-functions.h"
#include "kudu/util/curl_util.h"
#include "kudu/util/faststring.h"
#include "kudu/util/flag_tags.h"
#include "kudu/util/monotime.h"
#include "kudu/util/status.h"
#include "runtime/exec-env.h"
#include "service/frontend.h"

using namespace rapidjson;
using namespace impala_udf;

DEFINE_string(ai_endpoint, "", "The default API endpoint for an external AI engine.");

DEFINE_string(ai_model, "", "The default AI model used by an external AI engine.");

DEFINE_string(ai_api_key_jceks_secret, "",
    "The jceks secret key used for extracting the api key from configured keystores. "
    "'hadoop.security.credential.provider.path' in core-site must be configured to "
    "include the keystore storing the corresponding secret.");

DEFINE_int32(ai_connection_timeout_s, 10,
    "(Advanced) The time in seconds for connection timed out when communicating with an "
    "external AI engine");
TAG_FLAG(ai_api_key_jceks_secret, sensitive);

namespace impala {

const string AiFunctions::AI_GENERATE_TXT_JSON_PARSE_ERROR = "Invalid Json";
const string AiFunctions::AI_GENERATE_TXT_INVALID_PROTOCOL_ERROR =
    "Invalid Protocol, use https";
const string AiFunctions::AI_GENERATE_TXT_UNSUPPORTED_ENDPOINT_ERROR =
    "Unsupported Endpoint";
const string AiFunctions::AI_GENERATE_TXT_INVALID_PROMPT_ERROR =
    "Invalid Prompt, cannot be null or empty";
const string AiFunctions::AI_GENERATE_TXT_MSG_OVERRIDE_FORBIDDEN_ERROR =
    "Invalid override, 'messages' cannot be overriden";
static const char* AI_API_ENDPOINT_PREFIX = "https://";
// OPEN AI specific constants
static const char* OPEN_AI_AZURE_ENDPOINT = "openai.azure.com";
static const char* OPEN_AI_PUBLIC_ENDPOINT = "api.openai.com";
static const char* OPEN_AI_RESPONSE_FIELD_CHOICES = "choices";
static const char* OPEN_AI_RESPONSE_FIELD_MESSAGE = "message";
static const char* OPEN_AI_RESPONSE_FIELD_CONTENT = "content";
static const char* OPEN_AI_REQUEST_FIELD_CONTENT_TYPE_HEADER =
    "Content-Type: application/json";

static const StringVal NULL_STRINGVAL = StringVal::null();

string AiFunctions::ai_api_key_;

bool AiFunctions::is_api_endpoint_valid(const string& endpoint) {
  // Simple validation for endpoint. It should start with https://
  return (strncaseprefix(endpoint.c_str(), endpoint.size(), AI_API_ENDPOINT_PREFIX,
              sizeof(AI_API_ENDPOINT_PREFIX))
      != nullptr);
}

bool AiFunctions::is_api_endpoint_supported(const string& endpoint) {
  // Only OpenAI endpoints are supported.
  return (gstrcasestr(endpoint.c_str(), OPEN_AI_AZURE_ENDPOINT) != nullptr
      || gstrcasestr(endpoint.c_str(), OPEN_AI_PUBLIC_ENDPOINT) != nullptr);
}

static string AiGenerateTextParseOpenAiResponse(const rapidjson::Document& document) {
  // Check if the "choices" array exists and is not empty
  if (!document.HasMember(OPEN_AI_RESPONSE_FIELD_CHOICES)
      || !document[OPEN_AI_RESPONSE_FIELD_CHOICES].IsArray()
      || document[OPEN_AI_RESPONSE_FIELD_CHOICES].Empty()) {
    return "";
  }

  // Access the first element of the "choices" array
  const rapidjson::Value& firstChoice = document[OPEN_AI_RESPONSE_FIELD_CHOICES][0];

  // Check if the "message" object exists
  if (!firstChoice.HasMember(OPEN_AI_RESPONSE_FIELD_MESSAGE)
      || !firstChoice[OPEN_AI_RESPONSE_FIELD_MESSAGE].IsObject()) {
    return "";
  }

  // Access the "content" field within "message"
  const rapidjson::Value& message = firstChoice[OPEN_AI_RESPONSE_FIELD_MESSAGE];
  if (!message.HasMember(OPEN_AI_RESPONSE_FIELD_CONTENT)
      || !message[OPEN_AI_RESPONSE_FIELD_CONTENT].IsString()) {
    return "";
  }
  return message[OPEN_AI_RESPONSE_FIELD_CONTENT].GetString();
}

const char* stringify_json(Document& json, StringBuffer& buffer) {
  Writer<StringBuffer> writer(buffer);
  json.Accept(writer);
  return buffer.GetString();
}

StringVal AiFunctions::AiGenerateTextInternal(FunctionContext* ctx,
    const StringVal& endpoint, const StringVal& prompt, const StringVal& model,
    const StringVal& api_key_jceks_secret, const StringVal& params, const bool dry_run) {
  // endpoint validation
  string endpoint_str(FLAGS_ai_endpoint);
  if (endpoint.ptr != nullptr && endpoint.len != 0) {
    endpoint_str = string(reinterpret_cast<char*>(endpoint.ptr), endpoint.len);
    // Simple validation for endpoint. It should start with https://
    if (!is_api_endpoint_valid(endpoint_str)) {
      LOG(ERROR) << "AI Generate Text: \ninvalid protocol: " << endpoint_str;
      return StringVal(AI_GENERATE_TXT_INVALID_PROTOCOL_ERROR.c_str());
    }
    // Only OpenAI endpoints are supported.
    if (!is_api_endpoint_supported(endpoint_str)) {
      LOG(ERROR) << "AI Generate Text: \nunsupported endpoint: " << endpoint_str;
      return StringVal(AI_GENERATE_TXT_UNSUPPORTED_ENDPOINT_ERROR.c_str());
    }
  }
  // Generate the header for the POST request
  vector<string> headers;
  headers.emplace_back(OPEN_AI_REQUEST_FIELD_CONTENT_TYPE_HEADER);
  if (api_key_jceks_secret.ptr != nullptr && api_key_jceks_secret.len != 0) {
    string api_key;
    string api_key_secret(
        reinterpret_cast<char*>(api_key_jceks_secret.ptr), api_key_jceks_secret.len);
    Status status = ExecEnv::GetInstance()->frontend()->GetSecretFromKeyStore(
        api_key_secret, &api_key);
    if (!status.ok()) {
      return StringVal::CopyFrom(ctx,
          reinterpret_cast<const uint8_t*>(status.msg().msg().c_str()),
          status.msg().msg().length());
    }
    headers.emplace_back("Authorization: Bearer " + api_key);
  } else {
    headers.emplace_back("Authorization: Bearer " + ai_api_key_);
  }
  // Generate the payload for the POST request
  Document payload;
  payload.SetObject();
  Document::AllocatorType& payload_allocator = payload.GetAllocator();
  if (model.ptr != nullptr && model.len != 0) {
    payload.AddMember("model",
        rapidjson::StringRef(reinterpret_cast<char*>(model.ptr), model.len),
        payload_allocator);
  } else {
    payload.AddMember("model",
        rapidjson::StringRef(FLAGS_ai_model.c_str(), FLAGS_ai_model.length()),
        payload_allocator);
  }
  Value message_array(rapidjson::kArrayType);
  Value message(rapidjson::kObjectType);
  message.AddMember("role", "user", payload_allocator);
  if (prompt.ptr == nullptr || prompt.len == 0) {
    return StringVal(AI_GENERATE_TXT_INVALID_PROMPT_ERROR.c_str());
  }
  message.AddMember("content",
      rapidjson::StringRef(reinterpret_cast<char*>(prompt.ptr), prompt.len),
      payload_allocator);
  message_array.PushBack(message, payload_allocator);
  payload.AddMember("messages", message_array, payload_allocator);
  // Override additional params
  if (params.ptr != nullptr && params.len != 0) {
    Document overrides;
    overrides.Parse(reinterpret_cast<char*>(params.ptr), params.len);
    if (overrides.HasParseError()) {
      LOG(WARNING) << AI_GENERATE_TXT_JSON_PARSE_ERROR
                   << ": error code " << overrides.GetParseError()
                   << ", offset input " << overrides.GetErrorOffset();
      return StringVal(AI_GENERATE_TXT_JSON_PARSE_ERROR.c_str());
    }
    for (auto& m : overrides.GetObject()) {
      if (payload.HasMember(m.name.GetString())) {
        if (m.name == "messages") {
          LOG(WARNING)
              << AI_GENERATE_TXT_JSON_PARSE_ERROR
              << ": 'messages' is constructed from 'prompt', cannot be overridden";
          return StringVal(AI_GENERATE_TXT_MSG_OVERRIDE_FORBIDDEN_ERROR.c_str());
        } else {
          payload[m.name.GetString()] = m.value;
        }
      } else {
        payload.AddMember(m.name, m.value, payload_allocator);
      }
    }
  }
  StringBuffer buffer;
  string payload_str(stringify_json(payload, buffer));
  VLOG(2) << "AI Generate Text: \nendpoint: " << endpoint_str
          << " \npayload: " << payload_str;
  if (UNLIKELY(dry_run)) {
    std::stringstream post_request;
    post_request << endpoint_str;
    for (auto& header : headers) {
      post_request << "\n" << header;
    }
    post_request << "\n" << payload_str;
    return StringVal::CopyFrom(ctx,
        reinterpret_cast<const uint8_t*>(post_request.str().data()),
        post_request.str().length());
  }

  kudu::EasyCurl curl;
  curl.set_timeout(kudu::MonoDelta::FromSeconds(FLAGS_ai_connection_timeout_s));
  curl.set_fail_on_http_error(true);
  kudu::faststring resp;
  kudu::Status status = curl.PostToURL(endpoint_str, payload_str, &resp, headers);
  VLOG(2) << "AI Generate Text: \noriginal response: " << resp.ToString();
  if (!status.ok()) {
    string msg = status.ToString();
    return StringVal::CopyFrom(
        ctx, reinterpret_cast<const uint8_t*>(msg.c_str()), msg.size());
  }
  // Parse the JSON string
  rapidjson::Document document;
  document.Parse(resp.ToString().c_str());
  string response;
  // Check for parse errors
  if (document.HasParseError()) {
    LOG(WARNING) << AI_GENERATE_TXT_JSON_PARSE_ERROR << ": " << resp.ToString();
    return StringVal(AI_GENERATE_TXT_JSON_PARSE_ERROR.c_str());
  }
  response = AiGenerateTextParseOpenAiResponse(document);
  if (response.empty()) {
    LOG(WARNING) << AI_GENERATE_TXT_JSON_PARSE_ERROR << ": " << resp.ToString();
    return StringVal(AI_GENERATE_TXT_JSON_PARSE_ERROR.c_str());
  }
  VLOG(2) << "AI Generate Text: \nresponse: " << response;
  StringVal result(ctx, response.length());
  if (UNLIKELY(result.is_null)) return StringVal::null();
  memcpy(result.ptr, response.c_str(), response.length());
  return result;
}

StringVal AiFunctions::AiGenerateText(FunctionContext* ctx, const StringVal& endpoint,
    const StringVal& prompt, const StringVal& model,
    const StringVal& api_key_jceks_secret, const StringVal& params) {
  return AiGenerateTextInternal(
      ctx, endpoint, prompt, model, api_key_jceks_secret, params, false);
}

StringVal AiFunctions::AiGenerateText(FunctionContext* ctx, const StringVal& prompt) {
  return AiGenerateTextInternal(
      ctx, NULL_STRINGVAL, prompt, NULL_STRINGVAL, NULL_STRINGVAL, NULL_STRINGVAL, false);
}

StringVal AiFunctions::AiGenerateTextDummy(FunctionContext* ctx, const StringVal& prompt) {
  return AiGenerateTextInternal(
      ctx, NULL_STRINGVAL, prompt, NULL_STRINGVAL, NULL_STRINGVAL, NULL_STRINGVAL, false);
}

} // namespace impala
