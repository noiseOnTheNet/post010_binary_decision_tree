use polars::prelude::*;
use std::cmp::{Eq, PartialEq};
use std::hash::Hash;
mod btree;
use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
};

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

struct Rule<'a> {
    dimension: &'a str,
    cutoff: f64,
    metric: f64,
}

struct Decision<'a, T> {
    rule: Option<Rule<'a>>,
    confidence: f64,
    prediction: T,
}

struct DTreeBuilder<T> {
    max_level: usize,
    min_size: usize,
    phantom: PhantomData<T>,
}

// uses a struct to define trees constraints
impl<T : PolarsDataType + Copy + Hash + Eq> DTreeBuilder<T> {
    fn build_node<'a>(
        &self,
        data: DataFrame,
        level: usize,
        features: HashSet<&str>,
        target: &str,
    ) -> Option<btree::Node<Decision<'a, T>>> {
        let selection : &ChunkedArray<T> = data.column(target).unwrap().clone().unpack().unwrap();
        let node = predict_majority(& mut selection.iter());
        todo!("implement me!")
    }
    pub fn build<'a>(
        &self,
        data: DataFrame,
        features: HashSet<&str>,
        target: &str,
    ) -> btree::Tree<Decision<'a, T>> {
        if let Some(node) = self.build_node(data, 1, features, target) {
            btree::Tree::from_node(node)
        } else {
            btree::Tree::new()
        }
    }
}

// Gini impurity metric
fn gini(counts: &[u64]) -> f64 {
    let sum: u64 = counts.iter().sum();
    let result: f64 = 1.0
        - counts
            .iter()
            .map(|c| ((*c as f64) / (sum as f64)).powi(2))
            .sum::<f64>();
    result
}

// fn predict_majority(& values : Series){
//     let count = values;
// }

// when creating a node first check which would be the prodicted outcome
fn predict_majority<'a, T: Hash + Eq + PartialEq + Copy>(values: & mut dyn Iterator<Item=T>) -> Option<Decision<'a, T>> {
    let summary: HashMap<T, u64> = values.fold(HashMap::new(), |mut result, value| {
        result.insert(value, result.get(&value).unwrap_or(&0) + 1);
        result
    });
    let (prediction, count, total) = summary
        .iter()
        .fold((None, 0, 0), |(result, count, total), (key, value)| {
            if *value > count {
                (Some(key), *value, total + value)
            } else {
                (result, count, total)
            }
        });
    if let Some(result) = prediction {
        Some(Decision {
            rule: None,
            confidence: count as f64 / total as f64,
            prediction: *result,
        })
    } else {
        None
    }
}
