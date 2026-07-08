//! Faber `intervallum<T>` runtime — numeric interval with glyph-encoded inclusivity.

use crate::Tensor;

/// Endpoint inclusion policy declared at construction (`‥` vs `…` / `usque`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntervallumKind {
    /// Half-open `[initium, finis)`.
    Exclusive,
    /// Closed `[initium, finis]`.
    Inclusive,
}

/// Bindable numeric interval: two bounds of `T` plus an inclusivity tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Intervallum<T> {
    pub initium: T,
    pub finis: T,
    pub kind: IntervallumKind,
}

impl<T: PartialOrd + Copy> Intervallum<T> {
    pub fn exclusive(initium: T, finis: T) -> Self {
        Self {
            initium,
            finis,
            kind: IntervallumKind::Exclusive,
        }
    }

    pub fn inclusive(initium: T, finis: T) -> Self {
        Self {
            initium,
            finis,
            kind: IntervallumKind::Inclusive,
        }
    }

    /// Point containment (`intra`): honors the interval's declared inclusivity.
    pub fn continet(&self, value: &T) -> bool {
        match self.kind {
            IntervallumKind::Exclusive => value >= &self.initium && value < &self.finis,
            IntervallumKind::Inclusive => value >= &self.initium && value <= &self.finis,
        }
    }

    /// Interval intersection; `None` when disjoint (distinct from range clamp).
    pub fn inter(self, other: Self) -> Option<Self> {
        let initium = max_bound(self.initium, other.initium);
        let finis = min_bound(self.finis, other.finis);
        if initium > finis {
            return None;
        }
        if initium == finis {
            if !point_in_both(initium, &self, &other) {
                return None;
            }
            return Some(Self {
                initium,
                finis,
                kind: intersection_kind_at_point(&self, &other),
            });
        }
        Some(Self {
            initium,
            finis,
            kind: if self.continet(&finis) && other.continet(&finis) {
                IntervallumKind::Inclusive
            } else {
                IntervallumKind::Exclusive
            },
        })
    }

    /// Interval union when overlap or adjacent; `None` when a gap separates them.
    pub fn union(self, other: Self) -> Option<Self> {
        if self.inter(other).is_none() && !self.touches(other) {
            return None;
        }
        let initium = min_bound(self.initium, other.initium);
        let finis = max_bound(self.finis, other.finis);
        let kind = if self.kind == IntervallumKind::Inclusive
            && other.kind == IntervallumKind::Inclusive
        {
            IntervallumKind::Inclusive
        } else {
            IntervallumKind::Exclusive
        };
        Some(Self {
            initium,
            finis,
            kind,
        })
    }

    fn touches(self, other: Self) -> bool {
        if self.finis == other.initium {
            return self.continet(&other.initium) || other.continet(&self.finis);
        }
        if other.finis == self.initium {
            return other.continet(&self.initium) || self.continet(&other.finis);
        }
        false
    }
}

fn max_bound<T: PartialOrd + Copy>(a: T, b: T) -> T {
    if a >= b {
        a
    } else {
        b
    }
}

fn min_bound<T: PartialOrd + Copy>(a: T, b: T) -> T {
    if a <= b {
        a
    } else {
        b
    }
}

fn point_in_both<T: PartialOrd + Copy>(
    point: T,
    left: &Intervallum<T>,
    right: &Intervallum<T>,
) -> bool {
    left.continet(&point) && right.continet(&point)
}

fn intersection_kind_at_point<T: PartialOrd + Copy>(
    left: &Intervallum<T>,
    right: &Intervallum<T>,
) -> IntervallumKind {
    if left.kind == IntervallumKind::Inclusive && right.kind == IntervallumKind::Inclusive {
        IntervallumKind::Inclusive
    } else {
        IntervallumKind::Exclusive
    }
}

impl Intervallum<i64> {
    /// Clamp `value` into this interval (refinement-target conversio: result is `numerus`).
    pub fn coercere(&self, value: i64) -> i64 {
        if self.continet(&value) {
            return value;
        }
        if value < self.initium {
            return self.initium;
        }
        match self.kind {
            IntervallumKind::Exclusive => self.finis.saturating_sub(1),
            IntervallumKind::Inclusive => self.finis,
        }
    }

    /// Range-to-range clamp: each bound coerced into `target`; result inherits `target.kind`.
    pub fn coercere_intervallum(&self, target: &Self) -> Self {
        Self {
            initium: target.coercere(self.initium),
            finis: target.coercere(self.finis),
            kind: target.kind,
        }
    }

    /// Materialize interval values into an eager list (honors declared inclusivity).
    pub fn ad_lista(&self) -> Vec<i64> {
        let step = if self.initium <= self.finis { 1 } else { -1 };
        let mut out = Vec::new();
        let mut cursor = self.initium;
        if step > 0 {
            while match self.kind {
                IntervallumKind::Exclusive => cursor < self.finis,
                IntervallumKind::Inclusive => cursor <= self.finis,
            } {
                out.push(cursor);
                if cursor == i64::MAX {
                    break;
                }
                cursor += step;
            }
        } else {
            while match self.kind {
                IntervallumKind::Exclusive => cursor > self.finis,
                IntervallumKind::Inclusive => cursor >= self.finis,
            } {
                out.push(cursor);
                if cursor == i64::MIN {
                    break;
                }
                cursor += step;
            }
        }
        out
    }

    /// Discrete span count for `numerus` intervals (same cardinality as `ad_lista()`).
    pub fn longitudo(&self) -> i64 {
        let span = if self.initium <= self.finis {
            self.finis.saturating_sub(self.initium)
        } else {
            self.initium.saturating_sub(self.finis)
        };
        match self.kind {
            IntervallumKind::Exclusive => span,
            IntervallumKind::Inclusive => span.saturating_add(1),
        }
    }

    /// Materialize interval values into a 1-d tensor (honors declared inclusivity).
    pub fn ad_tensor(&self) -> Tensor<i64> {
        Tensor::linea(self.ad_lista())
    }
}
