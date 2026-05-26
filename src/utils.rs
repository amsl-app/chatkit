use regex::Regex;
use std::sync::LazyLock;

pub(crate) static THINKING_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<think>(.*?)</think>").expect("thinking regex is invalid"));

pub(crate) fn extract_thinking(s: String) -> (Option<String>, String) {
    let thinking = THINKING_RE
        .captures(&s)
        .and_then(|caps| caps.get(1).map(|m| m.as_str().to_string()));
    let cleaned = if thinking.is_some() {
        THINKING_RE.replace_all(&s, "").to_string()
    } else {
        // There's no thinking tag, so we can skip applying the regex again.
        s
    };
    (thinking, cleaned)
}

pub(crate) fn reject_empty(data: String) -> Option<String> {
    if data.is_empty() { None } else { Some(data) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_thinking() {
        let (thinking, cleaned) = extract_thinking("<think>thinking hard</think>The answer is 42".into());
        assert_eq!(thinking, Some("thinking hard".to_string()));
        assert_eq!(cleaned, "The answer is 42");
    }

    #[test]
    fn test_no_thinking() {
        let (thinking, cleaned) = extract_thinking("The answer is 42".into());
        assert_eq!(thinking, None);
        assert_eq!(cleaned, "The answer is 42");
    }

    #[test]
    fn test_reject_empty() {
        assert_eq!(reject_empty("".to_string()), None);
    }
}
