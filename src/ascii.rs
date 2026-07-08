//! Faber `ascii` runtime newtype.

/// ASCII-only text carrier for Faber `ascii`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ascii(String);

impl Ascii {
    pub fn new(value: &str) -> Self {
        debug_assert!(value.is_ascii());
        Self(value.to_owned())
    }

    pub fn try_from_textus(text: &str) -> Option<Self> {
        if text.is_ascii() {
            Some(Self(text.to_owned()))
        } else {
            None
        }
    }

    /// WHY: `octeti ↦ ascii` conversio needs a direct bytes→ascii path that fails
    /// on any byte ≥ 128, independent of UTF-8 validity. Going through `textus`
    /// first would conflate ASCII validity with UTF-8 validity. This validates
    /// ASCII directly and constructs the carrier.
    pub fn try_from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.is_ascii() {
            // SAFETY: ASCII is a strict subset of UTF-8, so any ASCII byte
            // sequence is valid UTF-8.
            let text = std::str::from_utf8(bytes).expect("ascii byte slice must be valid utf-8");
            Some(Self(text.to_owned()))
        } else {
            None
        }
    }

    pub fn to_textus(&self) -> String {
        self.0.clone()
    }
}

impl std::fmt::Display for Ascii {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(crate::display_text_payload(self.as_ref()))
    }
}

impl std::ops::Deref for Ascii {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<str> for Ascii {
    fn as_ref(&self) -> &str {
        &self.0
    }
}
