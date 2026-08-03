use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config file {path}: {detail}")]
    FileRead { path: String, detail: String },
    #[error("failed to parse config: {message}")]
    Parse { message: String },
    #[error("validation error: {message}")]
    Validation { message: String },
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct AppConfig {
    pub providers: HashMap<String, ProviderConfig>,
    pub models: HashMap<String, ModelConfig>,
    pub routing: RoutingConfig,
    pub retry: RetryConfig,
    #[serde(default)]
    pub fallbacks: HashMap<String, Vec<String>>,
    #[serde(default)]
    pub context_window_fallbacks: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ProviderConfig {
    pub base_url: String,
    pub keys: Vec<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelConfig {
    pub provider: String,
    pub remote_name: String,
    pub max_input_tokens: u32,
    pub max_output_tokens: u32,
    pub supports_vision: bool,
    pub timeout: u64,
    pub stream_timeout: u64,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RoutingConfig {
    pub strategy: String,
    pub allowed_fails: u32,
    pub cooldown_seconds: u64,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub base_wait_seconds: u64,
    pub rate_limit_retries: u32,
    pub timeout_retries: u32,
    pub server_error_retries: u32,
    pub auth_error_retries: u32,
    pub content_policy_retries: u32,
}

#[derive(Debug, Clone)]
pub struct ResolvedModelEntry {
    pub model_name: String,
    pub provider_name: String,
    pub remote_name: String,
    pub base_url: String,
    pub keys: Vec<String>,
    pub max_input_tokens: u32,
    pub max_output_tokens: u32,
    pub supports_vision: bool,
    pub timeout: u64,
    pub stream_timeout: u64,
}

impl AppConfig {
    pub fn from_file(path: &Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path).map_err(|e| ConfigError::FileRead {
            path: path.display().to_string(),
            detail: e.to_string(),
        })?;
        let config: AppConfig = serde_yaml::from_str(&content).map_err(|e| ConfigError::Parse {
            message: e.to_string(),
        })?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), ConfigError> {
        for (model_name, model) in &self.models {
            if !self.providers.contains_key(&model.provider) {
                return Err(ConfigError::Validation {
                    message: format!(
                        "model '{}' references unknown provider '{}'",
                        model_name, model.provider
                    ),
                });
            }
            if model.timeout == 0 {
                return Err(ConfigError::Validation {
                    message: format!("model '{}' has timeout == 0", model_name),
                });
            }
            if model.stream_timeout == 0 {
                return Err(ConfigError::Validation {
                    message: format!("model '{}' has stream_timeout == 0", model_name),
                });
            }
        }

        for (name, provider) in &self.providers {
            if provider.keys.is_empty() {
                return Err(ConfigError::Validation {
                    message: format!("provider '{}' has no keys", name),
                });
            }
        }

        for (source, targets) in &self.fallbacks {
            if !self.models.contains_key(source) {
                return Err(ConfigError::Validation {
                    message: format!("fallback source '{}' is not a known model", source),
                });
            }
            for target in targets {
                if !self.models.contains_key(target) {
                    return Err(ConfigError::Validation {
                        message: format!(
                            "fallback for '{}' targets unknown model '{}'",
                            source, target
                        ),
                    });
                }
            }
        }

        for (source, targets) in &self.context_window_fallbacks {
            if !self.models.contains_key(source) {
                return Err(ConfigError::Validation {
                    message: format!(
                        "context_window_fallback source '{}' is not a known model",
                        source
                    ),
                });
            }
            for target in targets {
                if !self.models.contains_key(target) {
                    return Err(ConfigError::Validation {
                        message: format!(
                            "context_window_fallback for '{}' targets unknown model '{}'",
                            source, target
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    pub fn resolve_model(&self, model_name: &str) -> Option<ResolvedModelEntry> {
        let model = self.models.get(model_name)?;
        let provider = self.providers.get(&model.provider)?;
        Some(ResolvedModelEntry {
            model_name: model_name.to_string(),
            provider_name: model.provider.clone(),
            remote_name: model.remote_name.clone(),
            base_url: provider.base_url.clone(),
            keys: provider.keys.clone(),
            max_input_tokens: model.max_input_tokens,
            max_output_tokens: model.max_output_tokens,
            supports_vision: model.supports_vision,
            timeout: model.timeout,
            stream_timeout: model.stream_timeout,
        })
    }
}
