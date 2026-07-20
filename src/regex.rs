//! Faber `regex` runtime carrier.

/// Pattern carrier for Faber `regex` (compile-time literal only today).
#[derive(Clone, PartialEq, Eq)]
pub struct Regex {
    pattern: String,
}

impl Regex {
    #[must_use]
    pub fn new(pattern: &str) -> Self {
        Self {
            pattern: pattern.to_owned(),
        }
    }

    #[must_use]
    pub fn pattern(&self) -> &str {
        &self.pattern
    }
}

impl std::fmt::Debug for Regex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "regex {:?}", self.pattern)
    }
}

impl std::fmt::Display for Regex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.pattern)
    }
}
