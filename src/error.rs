use std::fmt;

pub type Result<T> = std::result::Result<T, LlmAdapterError>;

#[derive(Debug)]
pub enum LlmAdapterError {
    ConfigError {
        message: String,
    },
    ModelNotFound {
        model: String,
        available: Vec<String>,
    },
    AllKeysExhausted {
        provider: String,
        model: String,
    },
    ApiError {
        provider: String,
        model: String,
        status_code: u16,
        error_code: String,
        message: String,
        retryable: bool,
    },
    TimeoutError {
        provider: String,
        model: String,
        timeout_secs: u64,
        source: String,
    },
    StreamError {
        provider: String,
        message: String,
    },
    RequestValidation {
        field: String,
        reason: String,
    },
    FallbackExhausted {
        original_model: String,
        attempted: Vec<String>,
    },
    HttpError {
        source: reqwest::Error,
    },
}

impl LlmAdapterError {
    pub fn http_status_code(&self) -> Option<u16> {
        match self {
            Self::ApiError { status_code, .. } => Some(*status_code),
            Self::HttpError { source } => source.status().map(|s| s.as_u16()),
            _ => None,
        }
    }

    pub fn should_penalize_api_key(&self) -> bool {
        match self {
            Self::ApiError { status_code: 429, .. } => false,
            Self::HttpError { source } => match source.status() {
                Some(s) => s.as_u16() != 429,
                None => false,
            },
            _ => true,
        }
    }

    pub fn is_retryable(&self) -> bool {
        match self {
            Self::ApiError { retryable, .. } => *retryable,
            Self::TimeoutError { .. } => true,
            Self::HttpError { source } => {
                if let Some(status) = source.status() {
                    let code = status.as_u16();
                    return code == 429 || code == 500 || code == 502 || code == 503;
                }
                true
            }
            Self::ConfigError { .. }
            | Self::ModelNotFound { .. }
            | Self::AllKeysExhausted { .. }
            | Self::StreamError { .. }
            | Self::RequestValidation { .. }
            | Self::FallbackExhausted { .. } => false,
        }
    }

    pub fn api_error(
        provider: impl Into<String>,
        model: impl Into<String>,
        status_code: u16,
        error_code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        let retryable =
            status_code == 429 || status_code == 500 || status_code == 502 || status_code == 503;
        Self::ApiError {
            provider: provider.into(),
            model: model.into(),
            status_code,
            error_code: error_code.into(),
            message: message.into(),
            retryable,
        }
    }
}

impl fmt::Display for LlmAdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConfigError { message } => write!(f, "config error: {message}"),
            Self::ModelNotFound { model, available } => {
                write!(
                    f,
                    "model not found: '{model}' (available: {})",
                    available.join(", ")
                )
            }
            Self::AllKeysExhausted { provider, model } => {
                write!(
                    f,
                    "all API keys exhausted for provider '{provider}', model '{model}'"
                )
            }
            Self::ApiError {
                provider,
                model,
                status_code,
                error_code,
                message,
                ..
            } => {
                write!(
                    f,
                    "API error from '{provider}' for model '{model}': [{status_code} {error_code}] {message}"
                )
            }
            Self::TimeoutError {
                provider,
                model,
                timeout_secs,
                source,
            } => {
                write!(
                    f,
                    "request to '{provider}' for model '{model}' timed out after {timeout_secs}s ({source})"
                )
            }
            Self::StreamError { provider, message } => {
                write!(f, "stream error from '{provider}': {message}")
            }
            Self::RequestValidation { field, reason } => {
                write!(f, "validation error on field '{field}': {reason}")
            }
            Self::FallbackExhausted {
                original_model,
                attempted,
            } => {
                write!(
                    f,
                    "fallback exhausted for '{original_model}', attempted: {}",
                    attempted.join(", ")
                )
            }
            Self::HttpError { source } => write!(f, "HTTP error: {source}"),
        }
    }
}

impl std::error::Error for LlmAdapterError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::HttpError { source } => Some(source),
            _ => None,
        }
    }
}
