use crate::valor::{FromValor, Valor};
use std::collections::{BTreeMap, HashMap};

#[test]
fn from_valor_extracts_lista_atomically() {
    assert_eq!(Vec::<i64>::from_valor(&Valor::Lista(vec![])), Some(vec![]));

    let valor = Valor::Lista(vec![Valor::Numerus(1), Valor::Numerus(2)]);
    assert_eq!(Vec::<i64>::from_valor(&valor), Some(vec![1, 2]));
    assert_eq!(
        Vec::<i64>::from_valor(&Valor::Lista(vec![
            Valor::Numerus(1),
            Valor::Textus("x".into())
        ])),
        None
    );
}

#[test]
fn from_valor_extracts_tabula_atomically() {
    assert_eq!(
        HashMap::<String, i64>::from_valor(&Valor::Tabula(BTreeMap::new())),
        Some(HashMap::new())
    );

    let mut tab = BTreeMap::new();
    tab.insert("a".to_owned(), Valor::Numerus(1));
    tab.insert("b".to_owned(), Valor::Numerus(2));
    let valor = Valor::Tabula(tab);
    let mut expected = HashMap::new();
    expected.insert("a".to_owned(), 1);
    expected.insert("b".to_owned(), 2);
    assert_eq!(HashMap::<String, i64>::from_valor(&valor), Some(expected));
    assert_eq!(
        HashMap::<String, i64>::from_valor(&Valor::Lista(vec![Valor::Numerus(1)])),
        None
    );
}
