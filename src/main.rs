use polars::prelude::*;
mod btree;
use std::marker::PhantomData;

fn main() {
    let df: DataFrame = df!(
        "integer" => &[1, 2, 3],
        "float" => &[4.0, 5.0, 6.0],
        "string" => &["a", "b", "c"],
    )
    .unwrap();
    let df2 = CsvReader::from_path("iris.csv")
        .unwrap()
        .has_header(true)
        .finish();
    println!("{}", df);
    if let Ok(df3) = df2 {
        println!("{}", df3);
    }
    println!("Hello, world!");
}

struct Decision<'a, T> {
    dimension : & 'a str,
    cutoff : f64,
    metric : f64,
    confidence : f64,
    prediction : T
}

struct DTreeBuilder<T>{
    max_level : usize,
    min_size : usize,
    phantom : PhantomData<T>
}

impl<T> DTreeBuilder<T>{
    fn build_node<'a>(& self, data : DataFrame, level : usize) -> btree::Node<Decision<'a, T>>{
        todo!("implement me!")
    }
    fn build<'a>(& self, data : DataFrame) -> btree::Tree<Decision<'a, T>>{
        btree::Tree::from_node(self.build_node(data, 1))
    }
}
