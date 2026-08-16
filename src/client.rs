use std::pin::Pin;
use std::time::Instant;

use futures_core::Stream;

use crate::backend::{Backend, NormalizedRequest, OpenAIBackend};
use crate::config::AppConfig;
use crate::error::{LlmAdapterError, Result};
use crate::models::{ChatRequest, ChatResponse, Content, Message, Role, StreamChunk};
use crate::retry::{self, RetryPolicy};
use crate::router::Router;

pub struct Client {
    router: Router,
    backend: OpenAIBackend,
    retry_policy: RetryPolicy,
    #[allow(dead_code)]
    config: AppConfig,
}

impl Client {
    pub async fn from_config_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let config = AppConfig::from_file(path.as_ref()).map_err(|e| {
            LlmAdapterError::ConfigError {
                message: e.to_string(),
            }
        })?;
        let router = Router::new(config.clone()).await;
        let backend = OpenAIBackend::new();
        let retry_policy = RetryPolicy::from_config(&config.retry);
        Ok(Self {
            router,
            backend,
            retry_policy,
            config,
        })
    }

    pub fn chat(&self) -> ChatBuilder<'_> {
        ChatBuilder {
            client: self,
            model: None,
            messages: Vec::new(),
            temperature: None,
            max_tokens: None,
            stream: false,
        }
    }
}

pub struct ChatBuilder<'a> {
    client: &'a Client,
    model: Option<String>,
    messages: Vec<Message>,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
    stream: bool,
}

impl<'a> ChatBuilder<'a> {
    pub fn model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }

    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    pub fn message(mut self, role: Role, content: &str) -> Self {
        self.messages.push(Message {
            role,
            content: Content::Text(content.to_string()),
        });
        self
    }

    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp);
        self
    }

    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    pub fn stream(mut self, enabled: bool) -> Self {
        self.stream = enabled;
        self
    }

    pub async fn send(self) -> Result<ChatResponse> {
        let model = self.model.ok_or_else(|| LlmAdapterError::RequestValidation {
            field: "model".to_string(),
            reason: "model is required".to_string(),
        })?;

        if self.messages.is_empty() {
            return Err(LlmAdapterError::RequestValidation {
                field: "messages".to_string(),
                reason: "at least one message is required".to_string(),
            });
        }

        let client = self.client;
        let messages = self.messages;
        let temperature = self.temperature;
        let max_tokens = self.max_tokens;
        let start = Instant::now();

        let result = retry::with_retry(&client.retry_policy, |_attempt| {
            let router = &client.router;
            let backend = &client.backend;
            let model = model.clone();
            let messages = messages.clone();
            let temperature = temperature;
            let max_tokens = max_tokens;

            async move {
                let decision = router.resolve(&model).await?;
                let max_tokens = effective_max_tokens(max_tokens, decision.max_output_tokens);

                let chat_request = ChatRequest {
                    model,
                    messages,
                    temperature,
                    top_p: None,
                    max_tokens,
                    stream: Some(false),
                    tools: None,
                };
                let normalized = NormalizedRequest {
                    chat_request,
                    provider_name: decision.provider_name.clone(),
                    model_name: decision.model_name.clone(),
                    remote_name: decision.remote_name.clone(),
                    base_url: decision.base_url.clone(),
                    timeout: decision.timeout,
                };

                let result = backend.chat(normalized, &decision.key).await;

                match &result {
                    Ok(_) => {
                        router
                            .record_success(
                                &decision.provider_name,
                                decision.key_index,
                                &decision.model_name,
                                start.elapsed().as_millis() as u64,
                            )
                            .await;
                        tracing::info!(
                            model_requested = %decision.model_name,
                            provider = %decision.provider_name,
                            key = %decision.key,
                            latency_ms = start.elapsed().as_millis() as u64,
                            "request completed"
                        );
                    }
                    Err(e) => {
                        let http_status = e.http_status_code();
                        let transport_hint = match &e {
                            LlmAdapterError::HttpError { source } if source.status().is_none() => {
                                let kind = if source.is_connect() {
                                    "connect"
                                } else if source.is_request() {
                                    "request_build"
                                } else {
                                    "transport"
                                };
                                Some(kind)
                            }
                            _ => None,
                        };
                        tracing::warn!(
                            provider = %decision.provider_name,
                            key_index = decision.key_index,
                            key_prefix = %&decision.key[..8.min(decision.key.len())],
                            http_status = ?http_status,
                            transport_kind = ?transport_hint,
                            error = %e,
                            "request failed"
                        );
                        if matches!(e, LlmAdapterError::TimeoutError { .. }) {
                            router.record_timeout(&decision.provider_name, decision.key_index).await;
                        } else if e.should_penalize_api_key() {
                            router.record_failure(&decision.provider_name, decision.key_index).await;
                        }
                    }
                }

                result
            }
        })
        .await;

        result
    }

    pub async fn send_stream(
        self,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let model = self.model.ok_or_else(|| LlmAdapterError::RequestValidation {
            field: "model".to_string(),
            reason: "model is required".to_string(),
        })?;

        if self.messages.is_empty() {
            return Err(LlmAdapterError::RequestValidation {
                field: "messages".to_string(),
                reason: "at least one message is required".to_string(),
            });
        }

        let decision = self.client.router.resolve(&model).await?;

        let chat_request = ChatRequest {
            model: model.clone(),
            messages: self.messages,
            temperature: self.temperature,
            top_p: None,
            max_tokens: self.max_tokens,
            stream: Some(true),
            tools: None,
        };

        let normalized = NormalizedRequest {
            chat_request,
            provider_name: decision.provider_name.clone(),
            model_name: decision.model_name.clone(),
            remote_name: decision.remote_name.clone(),
            base_url: decision.base_url.clone(),
            timeout: decision.stream_timeout,
        };

        tracing::info!(
            model_requested = %model,
            model_used = %decision.model_name,
            provider = %decision.provider_name,
            key = %decision.key,
            "stream request started"
        );

        self.client
            .backend
            .chat_stream(normalized, &decision.key)
            .await
    }
}

fn effective_max_tokens(requested: Option<u32>, model_limit: u32) -> Option<u32> {
    requested.or(Some(model_limit))
}

#[cfg(test)]
mod tests {
    use super::effective_max_tokens;

    #[test]
    fn omitted_max_tokens_falls_back_to_the_model_limit() {
        assert_eq!(effective_max_tokens(None, 8000), Some(8000));
    }

    #[test]
    fn explicit_max_tokens_wins_over_the_limit() {
        assert_eq!(effective_max_tokens(Some(50), 8000), Some(50));
    }
}
