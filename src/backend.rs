use std::pin::Pin;

use futures_core::Stream;
use futures_util::TryStreamExt;
use reqwest::Client;
use tokio::io::BufReader;

use crate::error::{self, LlmAdapterError};
use crate::models::{
    ChatRequest, ChatResponse, MiniMaxErrorResponse, StreamChunk, ZhipuResponse,
};
use crate::streaming;

pub struct NormalizedRequest {
    pub chat_request: ChatRequest,
    pub provider_name: String,
    pub model_name: String,
    pub remote_name: String,
    pub base_url: String,
    pub timeout: u64,
}

pub trait Backend: Send + Sync {
    fn chat<'a>(
        &'a self,
        request: NormalizedRequest,
        key: &'a str,
    ) -> Pin<Box<dyn std::future::Future<Output = error::Result<ChatResponse>> + Send + 'a>>;

    #[allow(clippy::type_complexity)]
    fn chat_stream<'a>(
        &'a self,
        request: NormalizedRequest,
        key: &'a str,
    ) -> Pin<
        Box<
            dyn std::future::Future<
                    Output = error::Result<
                        Pin<Box<dyn Stream<Item = error::Result<StreamChunk>> + Send>>,
                    >,
                > + Send
                + 'a,
        >,
    >;
}

fn is_success_status(status: u16) -> bool {
    (200..300).contains(&status)
}

pub struct OpenAIBackend {
    client: Client,
}

impl Default for OpenAIBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAIBackend {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }

    pub fn normalize_request(request: &mut NormalizedRequest) {
        match request.provider_name.as_str() {
            "zhipuai" => normalize_for_zhipuai(request),
            "minimax" => normalize_for_minimax(request),
            _ => {}
        }
    }

    async fn send_request(
        &self,
        request: &NormalizedRequest,
        key: &str,
    ) -> error::Result<reqwest::Response> {
        let mut normalized = NormalizedRequest {
            chat_request: ChatRequest {
                model: request.chat_request.model.clone(),
                messages: request.chat_request.messages.clone(),
                temperature: request.chat_request.temperature,
                top_p: request.chat_request.top_p,
                max_tokens: request.chat_request.max_tokens,
                stream: request.chat_request.stream,
                tools: request.chat_request.tools.clone(),
            },
            provider_name: request.provider_name.clone(),
            model_name: request.model_name.clone(),
            remote_name: request.remote_name.clone(),
            base_url: request.base_url.clone(),
            timeout: request.timeout,
        };

        Self::normalize_request(&mut normalized);

        let url = format!("{}/chat/completions", normalized.base_url.trim_end_matches('/'));

        self.client
            .post(&url)
            .header("Authorization", format!("Bearer {key}"))
            .header("Content-Type", "application/json")
            .timeout(std::time::Duration::from_secs(normalized.timeout))
            .json(&normalized.chat_request)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    LlmAdapterError::TimeoutError {
                        provider: normalized.provider_name.clone(),
                        model: normalized.model_name.clone(),
                        timeout_secs: normalized.timeout,
                        source: e.to_string(),
                    }
                } else {
                    LlmAdapterError::HttpError { source: e }
                }
            })
    }

    pub fn classify_response_error(
        status: u16,
        provider: &str,
        model: &str,
        body: &str,
    ) -> LlmAdapterError {
        let error_code = extract_error_code(body);
        let message = extract_error_message(body);

        let retryable = match status {
            401 => false,
            429 => true,
            500 | 502 | 503 => true,
            _ => status >= 500,
        };

        LlmAdapterError::ApiError {
            provider: provider.to_string(),
            model: model.to_string(),
            status_code: status,
            error_code,
            message,
            retryable,
        }
    }

    fn check_minimax_dual_error(
        body: &str,
        provider: &str,
        model: &str,
    ) -> Option<LlmAdapterError> {
        let resp: MiniMaxErrorResponse = serde_json::from_str(body).ok()?;
        let base = resp.base_resp?;
        if base.status_code != 0 {
            return Some(LlmAdapterError::ApiError {
                provider: provider.to_string(),
                model: model.to_string(),
                status_code: base.status_code as u16,
                error_code: base.status_code.to_string(),
                message: base.status_msg,
                retryable: false,
            });
        }
        None
    }

    fn parse_chat_response(
        status: u16,
        body: &str,
        provider: &str,
        model: &str,
    ) -> error::Result<ChatResponse> {
        if !is_success_status(status) {
            return Err(Self::classify_response_error(status, provider, model, body));
        }

        match provider {
            "zhipuai" => {
                let zhipu: ZhipuResponse = serde_json::from_str(body).map_err(|e| {
                    LlmAdapterError::ApiError {
                        provider: provider.to_string(),
                        model: model.to_string(),
                        status_code: 200,
                        error_code: "parse_error".to_string(),
                        message: format!("failed to parse ZhipuAI response: {e}"),
                        retryable: false,
                    }
                })?;

                let choices = zhipu
                    .choices
                    .into_iter()
                    .map(|zc| crate::models::Choice {
                        index: zc.index,
                        message: crate::models::Message {
                            role: zc.message.role,
                            content: crate::models::Content::Text(
                                zc.message.content.unwrap_or_default(),
                            ),
                        },
                        finish_reason: zc.finish_reason,
                    })
                    .collect();

                Ok(ChatResponse {
                    id: zhipu.id,
                    object: zhipu.object,
                    model: zhipu.model,
                    choices,
                    usage: zhipu.usage,
                })
            }
            "minimax" => {
                if let Some(err) = Self::check_minimax_dual_error(body, provider, model) {
                    return Err(err);
                }

                serde_json::from_str::<ChatResponse>(body).map_err(|e| {
                    LlmAdapterError::ApiError {
                        provider: provider.to_string(),
                        model: model.to_string(),
                        status_code: 200,
                        error_code: "parse_error".to_string(),
                        message: format!("failed to parse MiniMax response: {e}"),
                        retryable: false,
                    }
                })
            }
            _ => serde_json::from_str::<ChatResponse>(body).map_err(|e| {
                LlmAdapterError::ApiError {
                    provider: provider.to_string(),
                    model: model.to_string(),
                    status_code: 200,
                    error_code: "parse_error".to_string(),
                    message: format!("failed to parse response: {e}"),
                    retryable: false,
                }
            }),
        }
    }
}

impl Backend for OpenAIBackend {
    fn chat<'a>(
        &'a self,
        request: NormalizedRequest,
        key: &'a str,
    ) -> Pin<Box<dyn std::future::Future<Output = error::Result<ChatResponse>> + Send + 'a>>
    {
        Box::pin(async move {
            let response = self.send_request(&request, key).await?;

            let status = response.status().as_u16();
            let body = response.text().await.map_err(|e| LlmAdapterError::HttpError {
                source: e,
            })?;

            Self::parse_chat_response(status, &body, &request.provider_name, &request.model_name)
        })
    }

    #[allow(clippy::type_complexity)]
    fn chat_stream<'a>(
        &'a self,
        request: NormalizedRequest,
        key: &'a str,
    ) -> Pin<
        Box<
            dyn std::future::Future<
                    Output = error::Result<
                        Pin<Box<dyn Stream<Item = error::Result<StreamChunk>> + Send>>,
                    >,
                > + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            let mut stream_req = NormalizedRequest {
                chat_request: ChatRequest {
                    model: request.chat_request.model.clone(),
                    messages: request.chat_request.messages.clone(),
                    temperature: request.chat_request.temperature,
                    top_p: request.chat_request.top_p,
                    max_tokens: request.chat_request.max_tokens,
                    stream: Some(true),
                    tools: request.chat_request.tools.clone(),
                },
                provider_name: request.provider_name.clone(),
                model_name: request.model_name.clone(),
                remote_name: request.remote_name.clone(),
                base_url: request.base_url.clone(),
                timeout: request.timeout,
            };

            Self::normalize_request(&mut stream_req);

            let url = format!(
                "{}/chat/completions",
                stream_req.base_url.trim_end_matches('/')
            );

            let response = self
                .client
                .post(&url)
                .header("Authorization", format!("Bearer {key}"))
                .header("Content-Type", "application/json")
                .timeout(std::time::Duration::from_secs(stream_req.timeout))
                .json(&stream_req.chat_request)
                .send()
                .await
                .map_err(|e| {
                    if e.is_timeout() {
                        LlmAdapterError::TimeoutError {
                            provider: stream_req.provider_name.clone(),
                            model: stream_req.model_name.clone(),
                            timeout_secs: stream_req.timeout,
                            source: e.to_string(),
                        }
                    } else {
                        LlmAdapterError::HttpError { source: e }
                    }
                })?;

            let status = response.status().as_u16();
            if !is_success_status(status) {
                let body = response.text().await.map_err(|e| LlmAdapterError::HttpError {
                    source: e,
                })?;
                return Err(Self::classify_response_error(
                    status,
                    &stream_req.provider_name,
                    &stream_req.model_name,
                    &body,
                ));
            }

            let byte_stream = response.bytes_stream();
            let async_reader = tokio_util::io::StreamReader::new(byte_stream.map_err(|e| {
                std::io::Error::other(e)
            }));
            let buf_reader = BufReader::new(async_reader);

            let provider = stream_req.provider_name.clone();
            let sse_stream = streaming::parse_sse_stream(buf_reader);
            let mapped: Pin<Box<dyn Stream<Item = error::Result<StreamChunk>> + Send>> =
                Box::pin(async_stream::try_stream! {
                    for await item in sse_stream {
                        let chunk = item.map_err(|e| LlmAdapterError::StreamError {
                            provider: provider.clone(),
                            message: e.to_string(),
                        })?;
                        yield chunk;
                    }
                });

            Ok(mapped)
        })
    }
}

fn normalize_for_zhipuai(request: &mut NormalizedRequest) {
    request.chat_request.model = request.remote_name.clone();
    if let Some(temp) = request.chat_request.temperature
        && temp <= 0.0
    {
        request.chat_request.temperature = None;
    }
}

fn normalize_for_minimax(request: &mut NormalizedRequest) {
    request.chat_request.model = request.remote_name.clone();
    if let Some(temp) = request.chat_request.temperature {
        request.chat_request.temperature = Some(temp.min(1.0));
    }
    request.chat_request.top_p = None;
    request.chat_request.tools = None;
}

fn extract_error_code(body: &str) -> String {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| {
            v.get("error")
                .and_then(|e| e.get("code"))
                .or_else(|| v.get("base_resp").and_then(|b| b.get("status_code")))
                .map(|c| c.to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn extract_error_message(body: &str) -> String {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| {
            v.get("error")
                .and_then(|e| e.get("message"))
                .or_else(|| v.get("base_resp").and_then(|b| b.get("status_msg")))
                .and_then(|m| m.as_str().map(|s| s.to_string()))
        })
        .unwrap_or_else(|| body.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Content, Message, Role};

    fn base_request() -> ChatRequest {
        ChatRequest {
            model: "test-model".to_string(),
            messages: vec![Message {
                role: Role::User,
                content: Content::Text("hello".to_string()),
            }],
            temperature: Some(0.0),
            top_p: Some(0.9),
            max_tokens: Some(100),
            stream: None,
            tools: None,
        }
    }

    fn base_normalized() -> NormalizedRequest {
        NormalizedRequest {
            chat_request: base_request(),
            provider_name: "zhipuai".to_string(),
            model_name: "glm-4".to_string(),
            remote_name: "glm-4".to_string(),
            base_url: "https://api.z.ai/api/coding/paas/v4".to_string(),
            timeout: 30,
        }
    }

    #[test]
    fn zhipuai_removes_temperature_zero() {
        let mut req = base_normalized();
        normalize_for_zhipuai(&mut req);
        assert!(
            req.chat_request.temperature.is_none(),
            "temperature should be None when 0.0"
        );
    }

    #[test]
    fn zhipuai_keeps_positive_temperature() {
        let mut req = base_normalized();
        req.chat_request.temperature = Some(0.7);
        normalize_for_zhipuai(&mut req);
        assert_eq!(req.chat_request.temperature, Some(0.7));
    }

    #[test]
    fn minimax_clamps_temperature() {
        let mut req = NormalizedRequest {
            chat_request: ChatRequest {
                temperature: Some(2.0),
                top_p: Some(0.9),
                tools: Some(vec![serde_json::json!({"type": "function"})]),
                ..base_request()
            },
            provider_name: "minimax".to_string(),
            model_name: "minimax-m2.5".to_string(),
            remote_name: "MiniMax-M2.5".to_string(),
            base_url: "https://api.minimax.io/v1".to_string(),
            timeout: 30,
        };
        normalize_for_minimax(&mut req);
        assert_eq!(req.chat_request.temperature, Some(1.0));
        assert!(req.chat_request.top_p.is_none());
        assert!(req.chat_request.tools.is_none());
    }

    #[test]
    fn classify_401_not_retryable() {
        let err = OpenAIBackend::classify_response_error(401, "zhipuai", "glm-4", "unauthorized");
        assert!(!err.is_retryable());
    }

    #[test]
    fn classify_429_retryable() {
        let err = OpenAIBackend::classify_response_error(429, "zhipuai", "glm-4", "rate limited");
        assert!(err.is_retryable());
    }

    #[test]
    fn classify_503_retryable() {
        let err =
            OpenAIBackend::classify_response_error(503, "minimax", "m2.5", "service unavailable");
        assert!(err.is_retryable());
    }

    #[test]
    fn extract_error_code_from_openai_body() {
        let body = r#"{"error":{"code":"invalid_api_key","message":"Invalid API key"}}"#;
        assert_eq!(extract_error_code(body), "\"invalid_api_key\"");
    }

    #[test]
    fn extract_error_message_from_body() {
        let body = r#"{"error":{"code":"invalid_api_key","message":"Invalid API key"}}"#;
        assert_eq!(extract_error_message(body), "Invalid API key");
    }
}
