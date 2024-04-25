use polars::prelude::*;
use polars::series::Series;
use polars::chunked_array::ChunkedArray;
mod btree;
use std::collections::{HashMap, HashSet};
use std::fs::File;

fn main() {
    let df: DataFrame = df!(
        "integer" => &[1, 2, 3],
        "float" => &[4.0, 5.0, 6.0],
        "string" => &["a", "b", "c"],
    )
    .unwrap();
    let feature = "sepal_length";
    let target = "variety";
    let mut data = CsvReader::from_path("iris.csv")
        .unwrap()
        .has_header(true)
        .finish()
        .unwrap();

    let mut metrics = generate_all_splitting_points(& data, feature, target);
    let buffer = File::create("metrics.csv").unwrap();
    CsvWriter::new(buffer)
        .finish(&mut metrics)
        .unwrap();
}

#[derive(Debug)]
struct Rule<'a> {
    dimension: &'a str,
    cutoff: f64,
    metric: f64,
}

// categorical types are mapped to u64 because:
// 1. do not are equivalent to rust enums which are actually sum types
// 2. we target also 64bit execution platforms like webasm
#[derive(Debug)]
struct Decision<'a> {
    rule: Option<Rule<'a>>,
    confidence: f64,
    prediction: u64,
}

struct DTreeBuilder {
    max_level: usize,
    min_size: usize,
}

// uses a struct to define trees constraints
impl DTreeBuilder {
    fn build_node<'a>(
        &self,
        data: DataFrame,
        level: usize,
        features: HashSet<&str>,
        target: &str,
    ) -> Option<btree::Node<Decision<'a>>> {
        let selection: &ChunkedArray<UInt64Type> = data.column(target).unwrap().u64().unwrap();
        let node = predict_majority(&mut selection.iter());
        todo!("implement me!")
    }
    pub fn build<'a>(
        &self,
        data: DataFrame,
        features: HashSet<&str>,
        target: &str,
    ) -> btree::Tree<Decision<'a>> {
        if let Some(node) = self.build_node(data, 1, features, target) {
            btree::Tree::from_node(node)
        } else {
            btree::Tree::new()
        }
    }
}

// Gini impurity metric
fn gini(counts: Vec<u64>) -> f64 {
    let sum: u64 = counts.iter().sum();
    let result: f64 = 1.0
        - counts
            .iter()
            .map(|c| ((*c as f64) / (sum as f64)).powi(2))
            .sum::<f64>();
    result
}

fn dataframe_gini(data : DataFrame,target : & str) -> f64{
    let labels =  data.column(target)
                               .unwrap()
                               .u64()
                               .unwrap();
    let groups = count_groups(& mut labels.iter());
    let groups_count : Vec<u64> = groups.values().into_iter().map(|s| *s).collect();
    gini(groups_count)
}

fn count_groups(values: &mut dyn Iterator<Item = Option<u64>>) -> HashMap<u64, u64> {
    values
        .filter_map(|s| s)
        .fold(HashMap::new(), |mut result, value| {
            result.insert(value, result.get(&value).unwrap_or(&0) + 1);
            result
        })
}

// when creating a node first check which would be the prodicted outcome
fn predict_majority<'a>(values: &mut dyn Iterator<Item = Option<u64>>) -> Option<Decision<'a>> {
    let summary: HashMap<u64, u64> = count_groups(values);
    let (prediction, count, total) =
        summary
            .iter()
            .fold((None, 0, 0), |(result, count, total), (key, value)| {
                if *value > count {
                    (Some(key), *value, total + value)
                } else {
                    (result, count, total + value)
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

fn generate_all_splitting_points(data : & DataFrame, feature: & str, target: & str) -> DataFrame{
    let values = data.column(feature)
        .unwrap()
        .f64()
        .unwrap();
    let sorted_values : ChunkedArray<Float64Type> = values.sort(false);
    let splitting_points = sorted_values.rolling_map_float(2,|vs| vs.mean().map(|v| v as f64))
        .unwrap();
    let metrics : Vec<f64>= splitting_points.iter()
        .filter_map(|s| s)
        .map(
            |sp| {
                let higher = data.clone().filter(& values.gt_eq(sp))
                    .unwrap();
                let lower = data.clone().filter(& values.lt(sp))
                    .unwrap();
                dataframe_gini(higher, target) + dataframe_gini(lower, target)
            }
        )
        .collect();
    let splitting_points = splitting_points.into_series().with_name("split");
    let metrics = Series::new("metrics",metrics);
    DataFrame::new(vec![splitting_points,metrics]).unwrap()
}


#[cfg(test)]
mod test {
    use crate::{count_groups, predict_majority};

    #[test]
    fn test_count_groups() {
        let input: [Option<u64>; 14] = [
            Some(1u64),
            Some(1u64),
            Some(3u64),
            Some(2u64),
            None,
            Some(1u64),
            None,
            Some(2u64),
            Some(3u64),
            None,
            Some(2u64),
            Some(2u64),
            None,
            None,
        ];
        let result = count_groups(&mut input.iter().map(|s| *s));
        println!("{:?}", result);
        assert_eq!(result.get(&2u64), Some(&4u64));
    }

    #[test]
    fn test_predict_majority() {
        let input: [Option<u64>; 14] = [
            Some(1u64),
            Some(1u64),
            Some(3u64),
            Some(2u64),
            None,
            Some(1u64),
            None,
            Some(2u64),
            Some(3u64),
            None,
            Some(2u64),
            Some(2u64),
            None,
            None,
        ];
        let result = predict_majority(&mut input.iter().map(|s| *s));
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.prediction, 2u64);
        assert!(result.confidence < 0.5);
        assert!(result.confidence > 0.4);
    }
}
