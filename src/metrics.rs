use crate::types::CallConfig;

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

    pub(crate) fn first_token(&self) {
        // The precision loss is fine here, as we are only using it for metrics.
        // TODO use as_millis_f64() once it is stable
        #[allow(clippy::cast_precision_loss)]
        metrics::histogram!(
            "llm_time_to_first_token_ms",
            "service" => self.service.clone(),
            "model" => self.model.clone(),
        )
        .record(self.start_time.elapsed().as_millis() as f64);
    }

    pub(crate) fn last_token(&self) {
        // The precision loss is fine here, as we are only using it for metrics.
        // TODO use as_millis_f64() once it is stable
        #[allow(clippy::cast_precision_loss)]
        metrics::histogram!(
            "llm_time_to_last_token_ms",
            "service" => self.service.clone(),
            "model" => self.model.clone(),
        )
        .record(self.start_time.elapsed().as_millis() as f64);
    }
}

#[cfg(not(feature = "metrics"))]
impl MetricsRecorder {
    pub(crate) fn new(_config: &CallConfig, _model: &str) -> Self {
        Self
    }

    pub(crate) fn first_token(&self) {}

    pub(crate) fn last_token(&self) {}
}
