use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

use rand::Rng;
use tokio::sync::RwLock;

use crate::config::{AppConfig, RoutingConfig};
use crate::error::{LlmAdapterError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

pub struct CircuitBreaker {
    state: CircuitState,
    fail_count: u32,
    allowed_fails: u32,
    cooldown: std::time::Duration,
    opened_at: Option<std::time::Instant>,
}

impl CircuitBreaker {
    pub fn new(allowed_fails: u32, cooldown_seconds: u64) -> Self {
        Self {
            state: CircuitState::Closed,
            fail_count: 0,
            allowed_fails,
            cooldown: std::time::Duration::from_secs(cooldown_seconds),
            opened_at: None,
        }
    }

    pub fn allow_request(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                if let Some(opened_at) = self.opened_at
                    && opened_at.elapsed() >= self.cooldown
                {
                    self.state = CircuitState::HalfOpen;
                    tracing::info!(
                        "circuit breaker half-open"
                    );
                    return true;
                }
                false
            }
            CircuitState::HalfOpen => true,
        }
    }

    pub fn record_success(&mut self) {
        let was_open = self.state == CircuitState::Open || self.state == CircuitState::HalfOpen;
        self.fail_count = 0;
        self.opened_at = None;
        self.state = CircuitState::Closed;
        if was_open {
            tracing::info!(
                "circuit breaker recovered"
            );
        }
    }

    pub fn record_failure(&mut self) {
        self.fail_count += 1;
        if self.fail_count >= self.allowed_fails && self.state != CircuitState::Open {
            self.state = CircuitState::Open;
            self.opened_at = Some(std::time::Instant::now());
            tracing::warn!(
                failures = self.fail_count,
                "circuit breaker opened"
            );
        }
    }

    pub fn record_timeout(&mut self) {
        // Timeout immediately opens circuit breaker to force fallback
        if self.state != CircuitState::Open {
            self.state = CircuitState::Open;
            self.opened_at = Some(std::time::Instant::now());
            tracing::warn!(
                "circuit breaker opened due to timeout (immediate fallback)"
            );
        }
    }

    pub fn state(&self) -> CircuitState {
        self.state
    }
}

pub struct LatencyTracker {
    ema: f64,
    alpha: f64,
    sample_count: u64,
}

impl LatencyTracker {
    pub fn new(alpha: f64) -> Self {
        Self {
            ema: 0.0,
            alpha,
            sample_count: 0,
        }
    }

    pub fn update(&mut self, latency_ms: u64) {
        if self.sample_count == 0 {
            self.ema = latency_ms as f64;
        } else {
            self.ema = self.alpha * latency_ms as f64 + (1.0 - self.alpha) * self.ema;
        }
        self.sample_count += 1;
    }

    pub fn avg_ms(&self) -> f64 {
        self.ema
    }

    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }
}

pub struct KeyPool {
    keys: Vec<String>,
    index: AtomicU32,
}

impl KeyPool {
    pub fn new(keys: Vec<String>) -> Self {
        assert!(!keys.is_empty(), "KeyPool requires at least one key");
        let start = rand::rng().random_range(0..keys.len() as u32);
        Self {
            keys,
            index: AtomicU32::new(start),
        }
    }

    pub fn next_key(&self) -> (usize, &str) {
        let idx = self.index.fetch_add(1, Ordering::Relaxed) as usize;
        let key_idx = idx % self.keys.len();
        (key_idx, &self.keys[key_idx])
    }

    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    pub fn next_healthy_key(
        &self,
        circuits: &HashMap<(String, usize), CircuitBreaker>,
        provider: &str,
    ) -> Option<(usize, &str)> {
        let total = self.keys.len();
        for _offset in 0..total {
            let idx = self.index.fetch_add(1, Ordering::Relaxed) as usize % total;
            if let Some(breaker) = circuits.get(&(provider.to_string(), idx)) {
                if breaker.state() != CircuitState::Open {
                    return Some((idx, &self.keys[idx]));
                }
            } else {
                return Some((idx, &self.keys[idx]));
            }
        }
        None
    }
}

pub struct FallbackChain {
    alternates: Vec<String>,
    attempted: Vec<String>,
}

impl FallbackChain {
    pub fn new(alternates: Vec<String>) -> Self {
        Self {
            alternates,
            attempted: Vec::new(),
        }
    }

    pub fn try_next(&mut self) -> Option<&str> {
        let model = self.alternates.iter().find(|m| !self.attempted.contains(m))?;
        self.attempted.push(model.clone());
        Some(model)
    }

    pub fn attempted_models(&self) -> &[String] {
        &self.attempted
    }

    pub fn is_exhausted(&self) -> bool {
        self.attempted.len() >= self.alternates.len()
    }
}

#[derive(Debug)]
pub struct RoutingDecision {
    pub model_name: String,
    pub provider_name: String,
    pub remote_name: String,
    pub base_url: String,
    pub key: String,
    pub key_index: usize,
    pub max_input_tokens: u32,
    pub max_output_tokens: u32,
    pub supports_vision: bool,
    pub timeout: u64,
    pub stream_timeout: u64,
}

pub struct Router {
    config: AppConfig,
    circuits: RwLock<HashMap<(String, usize), CircuitBreaker>>,
    latency: RwLock<HashMap<String, LatencyTracker>>,
    key_pools: HashMap<String, KeyPool>,
    fallbacks: HashMap<String, Vec<String>>,
    context_window_fallbacks: HashMap<String, Vec<String>>,
}

impl Router {
    pub async fn new(config: AppConfig) -> Self {
        let allowed_fails = config.routing.allowed_fails;
        let cooldown_seconds = config.routing.cooldown_seconds;

        let mut circuits = HashMap::new();
        let mut latency = HashMap::new();
        let mut key_pools = HashMap::new();

        for model_name in config.models.keys() {
            latency.insert(model_name.clone(), LatencyTracker::new(0.3));
        }

        for (provider_name, provider) in &config.providers {
            let pool = KeyPool::new(provider.keys.clone());
            for key_idx in 0..pool.key_count() {
                circuits.insert(
                    (provider_name.clone(), key_idx),
                    CircuitBreaker::new(allowed_fails, cooldown_seconds),
                );
            }
            key_pools.insert(provider_name.clone(), pool);
        }

        let fallbacks = config.fallbacks.clone();
        let context_window_fallbacks = config.context_window_fallbacks.clone();

        Self {
            config,
            circuits: RwLock::new(circuits),
            latency: RwLock::new(latency),
            key_pools,
            fallbacks,
            context_window_fallbacks,
        }
    }

    pub fn routing_config(&self) -> &RoutingConfig {
        &self.config.routing
    }

    pub async fn resolve(&self, model_name: &str) -> Result<RoutingDecision> {
        let resolved = self
            .config
            .resolve_model(model_name)
            .ok_or_else(|| LlmAdapterError::ModelNotFound {
                model: model_name.to_string(),
                available: self.config.models.keys().cloned().collect(),
            })?;

        let pool = self
            .key_pools
            .get(&resolved.provider_name)
            .ok_or_else(|| LlmAdapterError::AllKeysExhausted {
                provider: resolved.provider_name.clone(),
                model: model_name.to_string(),
            })?;

        let circuits = self.circuits.read().await;

        if let Some((key_index, key)) =
            pool.next_healthy_key(&circuits, &resolved.provider_name)
        {
            let avg = self
                .latency
                .read()
                .await
                .get(model_name)
                .map(|t| t.avg_ms())
                .unwrap_or(0.0);
            tracing::info!(
                provider = %resolved.provider_name,
                key_index = key_index,
                total_keys = pool.key_count(),
                avg_latency_ms = avg,
                "selected key"
            );
            return Ok(RoutingDecision {
                model_name: resolved.model_name,
                provider_name: resolved.provider_name,
                remote_name: resolved.remote_name,
                base_url: resolved.base_url,
                key: key.to_string(),
                key_index,
                max_input_tokens: resolved.max_input_tokens,
                max_output_tokens: resolved.max_output_tokens,
                supports_vision: resolved.supports_vision,
                timeout: resolved.timeout,
                stream_timeout: resolved.stream_timeout,
            });
        }
        drop(circuits);

        self.try_fallback(model_name).await
    }

    fn build_decision(
        &self,
        resolved: &crate::config::ResolvedModelEntry,
        key: &str,
        key_index: usize,
    ) -> RoutingDecision {
        RoutingDecision {
            model_name: resolved.model_name.clone(),
            provider_name: resolved.provider_name.clone(),
            remote_name: resolved.remote_name.clone(),
            base_url: resolved.base_url.clone(),
            key: key.to_string(),
            key_index,
            max_input_tokens: resolved.max_input_tokens,
            max_output_tokens: resolved.max_output_tokens,
            supports_vision: resolved.supports_vision,
            timeout: resolved.timeout,
            stream_timeout: resolved.stream_timeout,
        }
    }

    async fn try_fallback(&self, original_model: &str) -> Result<RoutingDecision> {
        let mut chain = self.fallback_chain_for(original_model);
        let mut attempted = vec![original_model.to_string()];

        while let Some(fallback_model) = chain.try_next() {
            let resolved = match self.config.resolve_model(fallback_model) {
                Some(r) => r,
                None => {
                    attempted.push(fallback_model.to_string());
                    continue;
                }
            };

            tracing::info!(
                original_model = %original_model,
                fallback_model = %fallback_model,
                reason = "circuit_open",
                "trying fallback"
            );

            let pool = self
                .key_pools
                .get(&resolved.provider_name)
                .ok_or_else(|| LlmAdapterError::AllKeysExhausted {
                    provider: resolved.provider_name.clone(),
                    model: fallback_model.to_string(),
                })?;

            let circuits = self.circuits.read().await;
            match pool.next_healthy_key(&circuits, &resolved.provider_name) {
                Some((key_index, key)) => {
                    return Ok(self.build_decision(&resolved, key, key_index));
                }
                None => {
                    drop(circuits);
                    attempted.push(fallback_model.to_string());
                    continue;
                }
            }
        }

        Err(LlmAdapterError::FallbackExhausted {
            original_model: original_model.to_string(),
            attempted,
        })
    }

    pub fn fallback_chain_for(&self, model_name: &str) -> FallbackChain {
        let alternates = self
            .fallbacks
            .get(model_name)
            .cloned()
            .unwrap_or_default();
        FallbackChain::new(alternates)
    }

    pub fn context_window_fallback_for(&self, model_name: &str) -> Option<&str> {
        self.context_window_fallbacks
            .get(model_name)
            .and_then(|list| list.first().map(String::as_str))
    }

    pub async fn record_success(
        &self,
        provider_name: &str,
        key_index: usize,
        model_name: &str,
        latency_ms: u64,
    ) {
        if let Some(breaker) = self
            .circuits
            .write()
            .await
            .get_mut(&(provider_name.to_string(), key_index))
        {
            breaker.record_success();
        }
        if let Some(tracker) = self.latency.write().await.get_mut(model_name) {
            tracker.update(latency_ms);
        }
    }

    pub async fn record_failure(&self, provider_name: &str, key_index: usize) {
        if let Some(breaker) = self
            .circuits
            .write()
            .await
            .get_mut(&(provider_name.to_string(), key_index))
        {
            breaker.record_failure();
        }
    }

    pub async fn record_timeout(&self, provider_name: &str, key_index: usize) {
        if let Some(breaker) = self
            .circuits
            .write()
            .await
            .get_mut(&(provider_name.to_string(), key_index))
        {
            breaker.record_timeout();
        }
    }

    pub async fn latency_ms(&self, model_name: &str) -> f64 {
        let latency = self.latency.read().await;
        latency
            .get(model_name)
            .map(|t| t.avg_ms())
            .unwrap_or(0.0)
    }

    pub async fn circuit_state(
        &self,
        provider_name: &str,
        key_index: usize,
    ) -> Option<CircuitState> {
        let circuits = self.circuits.read().await;
        circuits
            .get(&(provider_name.to_string(), key_index))
            .map(|b| b.state())
    }

    pub fn available_models(&self) -> Vec<String> {
        self.config.models.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn test_routing_config() -> RoutingConfig {
        RoutingConfig {
            strategy: "round_robin".to_string(),
            allowed_fails: 3,
            cooldown_seconds: 120,
        }
    }

    fn test_app_config() -> AppConfig {
        use crate::config::{ModelConfig, ProviderConfig, RetryConfig};
        AppConfig {
            providers: HashMap::from([(
                "test_provider".to_string(),
                ProviderConfig {
                    base_url: "https://api.test.com/v1".to_string(),
                    keys: vec!["key1".to_string(), "key2".to_string(), "key3".to_string()],
                },
            )]),
            models: HashMap::from([(
                "model_a".to_string(),
                ModelConfig {
                    provider: "test_provider".to_string(),
                    remote_name: "remote-a".to_string(),
                    max_input_tokens: 8000,
                    max_output_tokens: 4000,
                    supports_vision: false,
                    timeout: 30,
                    stream_timeout: 60,
                },
            )]),
            routing: test_routing_config(),
            retry: RetryConfig {
                max_retries: 3,
                base_wait_seconds: 1,
                rate_limit_retries: 5,
                timeout_retries: 2,
                server_error_retries: 3,
                auth_error_retries: 0,
                content_policy_retries: 0,
            },
            fallbacks: HashMap::from([(
                "model_a".to_string(),
                vec!["model_b".to_string()],
            )]),
            context_window_fallbacks: HashMap::new(),
        }
    }

    #[test]
    fn circuit_breaker_opens_after_allowed_fails() {
        let mut cb = CircuitBreaker::new(3, 120);
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.allow_request());

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.allow_request());
    }

    #[test]
    fn circuit_breaker_half_open_after_cooldown() {
        let mut cb = CircuitBreaker::new(2, 0);
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(cb.allow_request());
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn circuit_breaker_resets_on_success() {
        let mut cb = CircuitBreaker::new(3, 120);
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);

        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.fail_count, 0);
    }

    #[test]
    fn latency_tracker_ema_averages() {
        let mut tracker = LatencyTracker::new(0.3);
        assert_eq!(tracker.avg_ms(), 0.0);

        tracker.update(100);
        assert_eq!(tracker.avg_ms(), 100.0);

        tracker.update(200);
        let expected = 0.3 * 200.0 + 0.7 * 100.0;
        assert!((tracker.avg_ms() - expected).abs() < 0.01);

        tracker.update(300);
        let expected2 = 0.3 * 300.0 + 0.7 * expected;
        assert!((tracker.avg_ms() - expected2).abs() < 0.01);
    }

    #[test]
    fn latency_tracker_ema_never_forgets() {
        let mut tracker = LatencyTracker::new(0.3);
        tracker.update(100);
        tracker.update(200);
        tracker.update(300);
        tracker.update(400);

        assert!(tracker.sample_count() == 4);
        assert!(tracker.avg_ms() > 0.0);
    }

    #[test]
    fn key_pool_round_robin() {
        // KeyPool starts at a random index (anti-thundering-herd) — assert the
        // sequence cycles in order from wherever it starts, not from zero.
        let keys = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let pool = KeyPool::new(keys.clone());
        let (first_idx, first_key) = pool.next_key();
        assert_eq!(keys[first_idx], first_key, "index and key agree at the start");
        for step in 1..keys.len() * 2 {
            let (idx, key) = pool.next_key();
            let want = (first_idx + step) % keys.len();
            assert_eq!(idx, want, "rotation advances and wraps");
            assert_eq!(keys[want], key, "key follows rotation");
        }
    }

    #[test]
    fn key_pool_single_key() {
        let pool = KeyPool::new(vec!["only".to_string()]);
        assert_eq!(pool.next_key(), (0, "only"));
        assert_eq!(pool.next_key(), (0, "only"));
    }

    #[test]
    fn key_pool_skips_open_circuits() {
        let pool = KeyPool::new(vec!["key1".to_string(), "key2".to_string(), "key3".to_string()]);
        let mut circuits: HashMap<(String, usize), CircuitBreaker> = HashMap::new();

        let mut cb0 = CircuitBreaker::new(2, 120);
        cb0.record_failure();
        cb0.record_failure();
        circuits.insert(("provider".to_string(), 0), cb0);

        let mut cb1 = CircuitBreaker::new(2, 120);
        cb1.record_failure();
        cb1.record_failure();
        circuits.insert(("provider".to_string(), 1), cb1);

        let result = pool.next_healthy_key(&circuits, "provider");
        assert!(result.is_some());
        let (idx, key) = result.unwrap();
        assert_eq!(idx, 2);
        assert_eq!(key, "key3");
    }

    #[test]
    fn key_pool_returns_none_when_all_open() {
        let pool = KeyPool::new(vec!["key1".to_string(), "key2".to_string()]);
        let mut circuits: HashMap<(String, usize), CircuitBreaker> = HashMap::new();

        let mut cb0 = CircuitBreaker::new(1, 120);
        cb0.record_failure();
        circuits.insert(("provider".to_string(), 0), cb0);

        let mut cb1 = CircuitBreaker::new(1, 120);
        cb1.record_failure();
        circuits.insert(("provider".to_string(), 1), cb1);

        let result = pool.next_healthy_key(&circuits, "provider");
        assert!(result.is_none());
    }

    #[test]
    fn fallback_chain_tries_alternates() {
        let mut chain = FallbackChain::new(vec![
            "alt_1".to_string(),
            "alt_2".to_string(),
        ]);
        assert_eq!(chain.try_next(), Some("alt_1"));
        assert_eq!(chain.try_next(), Some("alt_2"));
        assert_eq!(chain.try_next(), None);
        assert!(chain.is_exhausted());
    }

    #[test]
    fn fallback_chain_empty() {
        let mut chain = FallbackChain::new(vec![]);
        assert_eq!(chain.try_next(), None);
        assert!(chain.is_exhausted());
    }

    #[tokio::test]
    async fn router_resolves_model() {
        let config = test_app_config();
        let router = Router::new(config).await;
        let decision = router.resolve("model_a").await.unwrap();
        assert_eq!(decision.model_name, "model_a");
        assert_eq!(decision.provider_name, "test_provider");
        assert_eq!(decision.remote_name, "remote-a");
    }

    #[tokio::test]
    async fn router_returns_fallback_exhausted_for_unknown() {
        let config = test_app_config();
        let router = Router::new(config).await;
        let err = router.resolve("nonexistent").await.unwrap_err();
        assert!(matches!(err, LlmAdapterError::ModelNotFound { .. }));
    }

    #[tokio::test]
    async fn router_opens_circuit_after_failures() {
        let config = test_app_config();
        let router = Router::new(config).await;

        router.record_failure("test_provider", 0).await;
        router.record_failure("test_provider", 0).await;
        router.record_failure("test_provider", 0).await;

        assert_eq!(
            router.circuit_state("test_provider", 0).await,
            Some(CircuitState::Open)
        );
    }

    #[tokio::test]
    async fn router_tracks_latency() {
        let config = test_app_config();
        let router = Router::new(config).await;

        router
            .record_success("test_provider", 0, "model_a", 100)
            .await;
        router
            .record_success("test_provider", 0, "model_a", 200)
            .await;

        let avg = router.latency_ms("model_a").await;
        let expected = 0.3 * 200.0 + 0.7 * 100.0;
        assert!((avg - expected).abs() < 0.01);
    }
}
