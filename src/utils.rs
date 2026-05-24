use regex::Regex;
use std::sync::LazyLock;

pub(crate) static THINKING_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<think>(.*?)</think>").expect("thinking regex is invalid"));
