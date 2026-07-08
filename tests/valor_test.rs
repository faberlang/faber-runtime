use faber::Valor;
use std::collections::BTreeMap;

#[test]
fn valor_default_is_nihil() {
    assert!(Valor::default().is_nihil());
}

#[test]
fn valor_tabula_preserves_btree_order() {
    let mut map = BTreeMap::new();
    map.insert("b".to_owned(), Valor::Numerus(2));
    map.insert("a".to_owned(), Valor::Numerus(1));
    let valor = Valor::Tabula(map);
    let Valor::Tabula(entries) = valor else {
        panic!("expected tabula");
    };
    let keys: Vec<_> = entries.keys().cloned().collect();
    assert_eq!(keys, vec!["a", "b"]);
}

#[test]
fn valor_from_scalars() {
    assert_eq!(Valor::from(42i64), Valor::Numerus(42));
    assert_eq!(Valor::from("hi"), Valor::Textus("hi".to_owned()));
}
