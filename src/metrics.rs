use crate::config::CallConfig;

#[cfg(feature = "metrics")]
use std::borrow::Cow;
#[cfg(feature = "metrics")]
use tokio::time::Instant;

#[cfg(feature = "metrics")]
pub(crate) struct MetricsRecorder {
    start_time: Instant,
    service: Cow<'static, str>,
    model: Cow<'static, str>,
}

pub(crate) trait MetricsRecorderTrait {
    fn first_token(&self);
    fn last_token(&self);
}

#[cfg(not(feature = "metrics"))]
pub(crate) struct MetricsRecorder;

#[cfg(feature = "metrics")]
impl MetricsRecorder {
    pub(crate) fn new(config: &CallConfig, model: &str) -> Self {
        Self {
            start_time: Instant::now(),
            service: Cow::Owned(config.api_base.clone()),
            model: Cow::Owned(model.to_string()),
        }
    }

    fn record_elapsed_ms(&self, metric: &'static str) {
        // The precision loss is fine here, as we are only using it for metrics.
        // TODO use as_millis_f64() once it is stable
        #[allow(clippy::cast_precision_loss)]
        let duration_ms = self.start_time.elapsed().as_millis() as f64;
        metrics::histogram!(metric, "service" => self.service.clone(), "model" => self.model.clone())
            .record(duration_ms);
    }
}

#[cfg(feature = "metrics")]
impl MetricsRecorderTrait for MetricsRecorder {
    fn first_token(&self) {
        self.record_elapsed_ms("llm_time_to_first_token_ms");
    }

    fn last_token(&self) {
        self.record_elapsed_ms("llm_time_to_last_token_ms");
    }
}

#[cfg(not(feature = "metrics"))]
impl MetricsRecorder {
    pub(crate) fn new(_config: &CallConfig, _model: &str) -> Self {
        Self
    }
}

#[cfg(not(feature = "metrics"))]
impl MetricsRecorderTrait for MetricsRecorder {
    fn first_token(&self) {}

    fn last_token(&self) {}
}
