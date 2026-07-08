use super::{Intervallum, IntervallumKind, Tensor};

#[test]
fn exclusive_half_open_containment() {
    let range = Intervallum::exclusive(0_i64, 10);
    assert!(range.continet(&5));
    assert!(!range.continet(&10));
    assert!(!range.continet(&-1));
}

#[test]
fn inclusive_closed_containment() {
    let range = Intervallum::inclusive(0_i64, 10);
    assert!(range.continet(&10));
    assert!(!range.continet(&11));
}

#[test]
fn coercere_half_open_clamps_above_range() {
    let range = Intervallum::exclusive(0, 10);
    assert_eq!(range.coercere(15), 9);
    assert_eq!(range.coercere(5), 5);
    assert_eq!(range.coercere(-3), 0);
}

#[test]
fn coercere_inclusive_clamps_above_range() {
    let range = Intervallum::inclusive(0, 10);
    assert_eq!(range.coercere(15), 10);
}

#[test]
fn ad_lista_honors_inclusivity() {
    let half = Intervallum::exclusive(0, 3);
    assert_eq!(half.ad_lista(), vec![0, 1, 2]);
    let closed = Intervallum::inclusive(0, 3);
    assert_eq!(closed.ad_lista(), vec![0, 1, 2, 3]);
}

#[test]
fn ad_lista_honors_inclusive_extrema() {
    let near_max = Intervallum::inclusive(i64::MAX - 2, i64::MAX);
    assert_eq!(
        near_max.ad_lista(),
        vec![i64::MAX - 2, i64::MAX - 1, i64::MAX]
    );

    let near_min = Intervallum::inclusive(i64::MIN + 2, i64::MIN);
    assert_eq!(
        near_min.ad_lista(),
        vec![i64::MIN + 2, i64::MIN + 1, i64::MIN]
    );

    assert_eq!(
        Intervallum::inclusive(i64::MAX, i64::MAX).ad_lista(),
        vec![i64::MAX]
    );
    assert_eq!(
        Intervallum::inclusive(i64::MIN, i64::MIN).ad_lista(),
        vec![i64::MIN]
    );
}

#[test]
fn coercere_intervallum_inherits_target_kind() {
    let wide = Intervallum::exclusive(0, 100);
    let target = Intervallum::inclusive(10, 50);
    let narrow = wide.coercere_intervallum(&target);
    assert_eq!(narrow.kind, IntervallumKind::Inclusive);
    assert_eq!(narrow.initium, 10);
    assert_eq!(narrow.finis, 50);
}

#[test]
fn ad_tensor_materializes_one_dimensional_half_open() {
    let range = Intervallum::exclusive(0, 3);
    let tensor: Tensor<i64> = range.ad_tensor();
    assert_eq!(tensor.magnitudines(), vec![3]);
    assert_eq!(tensor.planata(), vec![0, 1, 2]);
}

#[test]
fn inter_disjoint_returns_none() {
    let left = Intervallum::exclusive(0, 5);
    let right = Intervallum::exclusive(6, 10);
    assert!(left.inter(right).is_none());
}

#[test]
fn inter_overlapping_half_open() {
    let left = Intervallum::exclusive(0, 10);
    let right = Intervallum::exclusive(5, 15);
    let hit = left.inter(right).expect("overlap");
    assert_eq!(hit, Intervallum::exclusive(5, 10));
}

#[test]
fn union_adjacent_half_open_merges() {
    let left = Intervallum::exclusive(0, 5);
    let right = Intervallum::exclusive(5, 10);
    let merged = left.union(right).expect("adjacent");
    assert_eq!(merged, Intervallum::exclusive(0, 10));
}

#[test]
fn union_gap_returns_none() {
    let left = Intervallum::exclusive(0, 5);
    let right = Intervallum::exclusive(6, 10);
    assert!(left.union(right).is_none());
}

#[test]
fn longitudo_counts_materialized_values() {
    let half = Intervallum::exclusive(0, 10);
    assert_eq!(half.longitudo(), 10);
    let closed = Intervallum::inclusive(0, 10);
    assert_eq!(closed.longitudo(), 11);
}

#[test]
fn longitudo_matches_ad_lista_cardinality() {
    for &(initium, finis, kind) in &[
        (0, 10, IntervallumKind::Exclusive),
        (0, 10, IntervallumKind::Inclusive),
        (10, 0, IntervallumKind::Exclusive),
        (5, 5, IntervallumKind::Inclusive),
    ] {
        let range = Intervallum {
            initium,
            finis,
            kind,
        };
        assert_eq!(range.longitudo(), range.ad_lista().len() as i64);
    }
}
