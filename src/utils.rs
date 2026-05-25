use regex::Regex;
use std::sync::LazyLock;

pub(crate) static THINKING_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<think>(.*?)</think>").expect("thinking regex is invalid"));

pub(crate) fn extract_thinking(s: &str) -> (Option<String>, String) {
    let thinking = THINKING_RE
        .captures(s)
        .and_then(|caps| caps.get(1).map(|m| m.as_str().to_string()));
    let cleaned = THINKING_RE.replace_all(s, "").to_string();
    (thinking, cleaned)
}

pub(crate) fn reject_empty(data: String) -> Option<String> {
    if data.is_empty() { None } else { Some(data) }
}
