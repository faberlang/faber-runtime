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
    // Intersection of [0,10) ∧ [5,15) is [5,10) = {5,6,7,8,9}.
    // Encoded as Inclusive 5…9 = {5,6,7,8,9} (equivalent point set).
    assert_eq!(hit, Intervallum::inclusive(5, 9));
}

#[test]
fn union_adjacent_half_open_merges() {
    let left = Intervallum::exclusive(0, 5);
    let right = Intervallum::exclusive(5, 10);
    let merged = left.union(right).expect("adjacent");
    // Union of [0,5) ∪ [5,10) is [0,10) = {0,…,9}.
    // Encoded as Inclusive 0…9 = {0,…,9} (equivalent point set).
    assert_eq!(merged, Intervallum::inclusive(0, 9));
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
        // SAFETY: test data is small.
        #[allow(clippy::cast_possible_wrap)]
        let len = range.ad_lista().len() as i64;
        assert_eq!(range.longitudo(), len);
    }
}

// --- Descending span tests (Option B — directional ranges) ---

#[test]
fn descending_exclusive_continet() {
    // 5‥0 = points {5, 4, 3, 2, 1}
    let range = Intervallum::exclusive(5, 0);
    assert!(range.continet(&5));
    assert!(range.continet(&4));
    assert!(range.continet(&1));
    assert!(!range.continet(&0)); // finis excluded
    assert!(!range.continet(&6));
    assert!(!range.continet(&-1));
}

#[test]
fn descending_inclusive_continet() {
    // 5…0 = points {5, 4, 3, 2, 1, 0}
    let range = Intervallum::inclusive(5, 0);
    assert!(range.continet(&5));
    assert!(range.continet(&0)); // finis included
    assert!(!range.continet(&6));
    assert!(!range.continet(&-1));
}

#[test]
fn descending_exclusive_coercere() {
    // 5‥0 = valid range [1, 5]
    let range = Intervallum::exclusive(5, 0);
    assert_eq!(range.coercere(5), 5);
    assert_eq!(range.coercere(3), 3); // inside
    assert_eq!(range.coercere(1), 1);
    assert_eq!(range.coercere(0), 1); // finis excluded, clamp up
    assert_eq!(range.coercere(-1), 1); // below range
    assert_eq!(range.coercere(6), 5); // above range
    assert_eq!(range.coercere(10), 5);
}

#[test]
fn descending_inclusive_coercere() {
    // 5…0 = valid range [0, 5]
    let range = Intervallum::inclusive(5, 0);
    assert_eq!(range.coercere(5), 5);
    assert_eq!(range.coercere(0), 0); // finis included
    assert_eq!(range.coercere(-1), 0); // below, clamp to min valid
    assert_eq!(range.coercere(6), 5); // above, clamp to max valid
}

#[test]
fn descending_exclusive_ad_lista() {
    let range = Intervallum::exclusive(5, 0);
    assert_eq!(range.ad_lista(), vec![5, 4, 3, 2, 1]);
}

#[test]
fn descending_inclusive_ad_lista() {
    let range = Intervallum::inclusive(5, 0);
    assert_eq!(range.ad_lista(), vec![5, 4, 3, 2, 1, 0]);
}

#[test]
fn descending_exclusive_longitudo() {
    let range = Intervallum::exclusive(5, 0);
    assert_eq!(range.longitudo(), 5);
    assert_eq!(range.longitudo(), range.ad_lista().len() as i64);
}

#[test]
fn descending_inclusive_longitudo() {
    let range = Intervallum::inclusive(5, 0);
    assert_eq!(range.longitudo(), 6);
    assert_eq!(range.longitudo(), range.ad_lista().len() as i64);
}

#[test]
fn descending_inter_overlapping() {
    // 5‥0 = {5,4,3,2,1}, 7‥1 = {7,6,5,4,3,2}
    // Intersection = {5,4,3,2}
    // Left operand is descending → result descending.
    let left = Intervallum::exclusive(5, 0);
    let right = Intervallum::exclusive(7, 1);
    let hit = left.inter(right).expect("overlap");
    // {5,4,3,2} as descending Inclusive: initium=5, finis=2
    assert_eq!(hit, Intervallum::inclusive(5, 2));
}

#[test]
fn descending_inter_disjoint() {
    // 5‥0 = {5,4,3,2,1}, 10‥8 = {10,9}
    let left = Intervallum::exclusive(5, 0);
    let right = Intervallum::exclusive(10, 8);
    assert!(left.inter(right).is_none());
}

#[test]
fn descending_union_adjacent() {
    // 5‥1 = {5,4,3,2}, 1‥-3 = {1,0,-1,-2}
    // Union = {5,4,3,2,1,0,-1,-2}
    let left = Intervallum::exclusive(5, 1);
    let right = Intervallum::exclusive(1, -3);
    let merged = left.union(right).expect("adjacent");
    // Descending: initium=5, finis=-2, Inclusive
    assert_eq!(merged, Intervallum::inclusive(5, -2));
    assert_eq!(merged.ad_lista(), vec![5, 4, 3, 2, 1, 0, -1, -2]);
}

#[test]
fn mixed_direction_inter() {
    // ascending 0‥5 = {0,1,2,3,4}, descending 7‥1 = {7,6,5,4,3,2}
    // Intersection = {2,3,4}
    let asc = Intervallum::exclusive(0, 5);
    let desc = Intervallum::exclusive(7, 1);
    let hit = asc.inter(desc).expect("overlap");
    // Left is ascending → result ascending: initium=2, finis=4, Inclusive
    assert_eq!(hit, Intervallum::inclusive(2, 4));
    assert_eq!(hit.ad_lista(), vec![2, 3, 4]);
}

#[test]
fn mixed_direction_union_touching() {
    // descending 4‥1 = {4,3,2}, ascending 5…10 = {5,6,7,8,9,10}
    // Adjacent at 4→5; union = {4,3,2,5,6,7,8,9,10}
    let desc = Intervallum::exclusive(4, 1);
    let asc = Intervallum::inclusive(5, 10);
    let merged = desc.union(asc).expect("adjacent");
    // Left is descending → result descending: initium=10, finis=2, Inclusive
    assert_eq!(merged, Intervallum::inclusive(10, 2));
    // Materialized descending: {10,9,8,7,6,5,4,3,2}
    assert_eq!(merged.ad_lista(), vec![10, 9, 8, 7, 6, 5, 4, 3, 2]);
}

#[test]
fn ad_tensor_descending_preserves_order() {
    let range = Intervallum::exclusive(5, 0);
    let tensor: Tensor<i64> = range.ad_tensor();
    assert_eq!(tensor.magnitudines(), vec![5]);
    assert_eq!(tensor.planata(), vec![5, 4, 3, 2, 1]);
}
