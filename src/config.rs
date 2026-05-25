use async_openai::config::Config;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderName, HeaderValue};
use schemars::JsonSchema;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::time::Duration;
use typed_builder::TypedBuilder;

pub(crate) const ORGANIZATION_HEADER: HeaderName = HeaderName::from_static("openai-organization");
pub(crate) const PROJECT_HEADER: HeaderName = HeaderName::from_static("openai-project");

pub(crate) const DEFAULT_API_BASE: &str = "https://api.openai.com/v1";

/// Connection and timeout settings for an LLM API endpoint.
#[derive(TypedBuilder, Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CallConfig {
    /// Covers the full request lifecycle including retries.
    #[builder(default = Duration::from_secs(30))]
    pub total_timeout: Duration,
    /// Per-attempt limit; for streaming, only applies to the initial connection.
    #[builder(default = Duration::from_secs(20))]
    pub iteration_timeout: Duration,
    #[builder(default = DEFAULT_API_BASE.into())]
    pub api_base: String,
    #[builder(default)]
    #[serde(skip)]
    pub api_key: SecretString,
    #[builder(default)]
    pub org_id: String,
    #[builder(default)]
    pub project_id: String,
    #[builder(default)]
    #[serde(skip)]
    pub custom_headers: HeaderMap,
}

impl Config for CallConfig {
    fn headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        insert_header(&mut headers, &ORGANIZATION_HEADER, &self.org_id);
        insert_header(&mut headers, &PROJECT_HEADER, &self.project_id);

        let mut header_value = HeaderValue::from_str(&format!("Bearer {}", self.api_key.expose_secret())).unwrap();
        header_value.set_sensitive(true);
        headers.insert(AUTHORIZATION, header_value);

        // Merge custom headers, with custom headers taking precedence
        for (key, value) in &self.custom_headers {
            headers.insert(key, value.clone());
        }

        headers
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.api_base, path)
    }

    fn query(&self) -> Vec<(&str, &str)> {
        vec![]
    }

    fn api_base(&self) -> &str {
        &self.api_base
    }

    fn api_key(&self) -> &SecretString {
        &self.api_key
    }
}

fn insert_header(headers: &mut HeaderMap, header: &HeaderName, value: &str) {
    headers.insert(
        header,
        value
            .parse()
            .inspect_err(|error| {
                tracing::error!(
                    header = header.as_str(),
                    value,
                    error = error as &dyn Error,
                    "invalid OpenAI header value"
                );
            })
            .unwrap(),
    );
}
