use std::future::Future;
use std::time::Duration;

use rand::Rng;

use crate::config;
use crate::error::{LlmAdapterError, Result};

const MAX_BACKOFF_MS: u64 = 30_000;

pub struct RetryPolicy {
    pub rate_limit: u32,
    pub timeout: u32,
    pub server_error: u32,
    pub auth: u32,
    pub content_policy: u32,
    pub default: u32,
    pub base_wait_ms: u64,
}

impl RetryPolicy {
    pub fn from_config(retry_config: &config::RetryConfig) -> Self {
        Self {
            rate_limit: retry_config.rate_limit_retries,
            timeout: retry_config.timeout_retries,
            server_error: retry_config.server_error_retries,
            auth: retry_config.auth_error_retries,
            content_policy: retry_config.content_policy_retries,
            default: retry_config.max_retries,
            base_wait_ms: retry_config.base_wait_seconds * 1000,
        }
    }

    pub fn max_retries_for(&self, error: &LlmAdapterError) -> u32 {
        match error {
            LlmAdapterError::ApiError {
                status_code: 429, ..
            } => self.rate_limit,
            LlmAdapterError::ApiError {
                status_code: 401, ..
            } => self.auth,
            LlmAdapterError::ApiError {
                status_code: 500 | 502 | 503,
                ..
            } => self.server_error,
            LlmAdapterError::ApiError {
                error_code,
                message,
                ..
            } => {
                let ec = error_code.to_lowercase();
                let msg = message.to_lowercase();
                if ec.contains("content_policy")
                    || ec.contains("content_filter")
                    || msg.contains("content_policy")
                    || msg.contains("content_filter")
                {
                    self.content_policy
                } else {
                    self.default
                }
            }
            LlmAdapterError::TimeoutError { .. } => self.timeout,
            LlmAdapterError::HttpError { source } => {
                if let Some(s) = source.status() {
                    let code = s.as_u16();
                    if code == 429 {
                        return self.rate_limit;
                    }
                    if matches!(code, 500 | 502 | 503) {
                        return self.server_error;
                    }
                    return self.default;
                }
                self.server_error.max(self.default)
            }
            _ => self.default,
        }
    }

    pub fn backoff(&self, attempt: u32) -> u64 {
        let exponential = self.base_wait_ms * 2u64.pow(attempt);
        let jitter_max = self.base_wait_ms / 2;
        let jitter = if jitter_max == 0 {
            0
        } else {
            rand::rng().random_range(0..=jitter_max)
        };
        (exponential + jitter).min(MAX_BACKOFF_MS)
    }
}

pub async fn with_retry<F, Fut, T>(policy: &RetryPolicy, operation: F) -> Result<T>
where
    F: Fn(u32) -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let mut attempt = 0u32;
    loop {
        match operation(attempt).await {
            Ok(result) => return Ok(result),
            Err(error) => {
                let max = policy.max_retries_for(&error);
                if attempt >= max {
                    return Err(error);
                }
                let wait = policy.backoff(attempt);
                tracing::warn!(
                    attempt,
                    max,
                    wait_ms = wait,
                    error = %error,
                    "retrying after error"
                );
                tokio::time::sleep(Duration::from_millis(wait)).await;
                attempt += 1;
            }
        }
    }
}
