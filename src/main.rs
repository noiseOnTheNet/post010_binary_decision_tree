use polars::prelude::*;
mod btree;
use std::collections::{HashMap, HashSet};

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

#[derive(Debug)]
struct Rule<'a> {
    dimension: &'a str,
    cutoff: f32,
    metric: f32,
}

// categorical types are mapped to u32 because:
// 1. do not are equivalent to rust enums which are actually sum types
// 2. we target also 32bit execution platforms like webasm
#[derive(Debug)]
struct Decision<'a> {
    rule: Option<Rule<'a>>,
    confidence: f32,
    prediction: u32,
}

struct DTreeBuilder {
    max_level: usize,
    min_size: usize
}

// uses a struct to define trees constraints
impl DTreeBuilder {
    fn build_node<'a>(
        &self,
        data: DataFrame,
        level: usize,
        features: HashSet<&str>,
        target: &str
    ) -> Option<btree::Node<Decision<'a>>> {
        let selection : &ChunkedArray<UInt32Type> = data.column(target).unwrap().u32().unwrap();
        let node = predict_majority(& mut selection.iter());
        todo!("implement me!")
    }
    pub fn build<'a>(
        &self,
        data: DataFrame,
        features: HashSet<& str>,
        target: & str
    ) -> btree::Tree<Decision<'a>> {
        if let Some(node) = self.build_node(data, 1, features, target) {
            btree::Tree::from_node(node)
        } else {
            btree::Tree::new()
        }
    }
}

// Gini impurity metric
fn gini(counts: &[u32]) -> f32 {
    let sum: u32 = counts.iter().sum();
    let result: f32 = 1.0
        - counts
            .iter()
            .map(|c| ((*c as f32) / (sum as f32)).powi(2))
            .sum::<f32>();
    result
}

fn count_groups(values: & mut dyn Iterator<Item=Option<u32>>) -> HashMap<u32, u32>{
        values.filter_map(|s| s)
        .fold(HashMap::new(), |mut result, value| {
            result.insert(value, result.get(&value).unwrap_or(&0) + 1);
            result
        })
}

// when creating a node first check which would be the prodicted outcome
fn predict_majority<'a>(values: & mut dyn Iterator<Item=Option<u32>>) -> Option<Decision<'a>> {
    let summary: HashMap<u32, u32> = count_groups(values);
    print!("<<<<<<");
    let (prediction, count, total) = summary
        .iter()
        .fold((None, 0, 0), |(result, count, total), (key, value)| {
            if *value > count {
                (Some(key), *value, total + value)
            } else {
                (result, count, total + value)
            }
        });
    println!("count = {}, total = {}, confidence = {}", count, total, count as f32 / total as f32);
    if let Some(result) = prediction {
        Some(Decision {
            rule: None,
            confidence: count as f32 / total as f32,
            prediction: *result,
        })
    } else {
        None
    }
}

#[cfg(test)]
mod test{
    use crate::{predict_majority, count_groups};

    #[test]
    fn test_count_groups(){
        let input : [Option<u32>; 14]=[
            Some(1u32), Some(1u32), Some(3u32), Some(2u32), None,
            Some(1u32), None,    Some(2u32), Some(3u32), None,
            Some(2u32), Some(2u32), None,    None];
        let result = count_groups(& mut input.iter().map(|s| *s));
        println!("{:?}",result);
        assert_eq!(result.get(&2u32),Some(&4u32));
    }

    #[test]
    fn test_predict_majority(){
        let input : [Option<u32>; 14]=[
            Some(1u32), Some(1u32), Some(3u32), Some(2u32), None,
            Some(1u32), None,    Some(2u32), Some(3u32), None,
            Some(2u32), Some(2u32), None,    None];
        let result = predict_majority(& mut input.iter().map(|s| *s));
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.prediction,2u32);
        println!(">>>>> result = {:?}",result);
        assert!(result.confidence < 0.5);
        assert!(result.confidence > 0.4);
    }
}
