use polars::prelude::*;
use polars::series::Series;
mod btree;
use std::collections::{HashMap, HashSet};
use std::fs::File;

fn main() -> polars::prelude::PolarsResult<()> {
    let mut df: DataFrame = df!(
        "integer" => &[1, 2, 3],
        "float" => &[4.0, 5.0, 6.0],
        "string" => &["a", "b", "c"],
    )
    .unwrap();
    println!(">>>>>>>>>>>>>>> {:?}", df);
    df.try_apply("string", |s| {
        s.cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
    })?;
    println!(">>>>>>>>>>>>>>> {:?}", df);
    let feature = "sepal_length";
    let target = "variety";
    let mut data = CsvReader::from_path("iris.csv")
        .unwrap()
        .has_header(true)
        .finish()
        .unwrap();
    println!(">>>>>>>>>>>>>>> {:?}", data);
    data.try_apply(target, |s| {
        s.cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
    })?;
    println!(">>>>>>>>>>>>>>> {:?}", data);
    println!(
        ">>>>>>>>>>>>>>> {:?}",
        data.column(target)?
            .value_counts(true, true)
    );
    let metrics = evaluate_metric(&data, feature, target);
    println!(">>>>>>>>>>>>>>>> {:?}", metrics);
    let mut metrics =  metrics?;
    let buffer = File::create("metrics.csv").unwrap();
    CsvWriter::new(buffer).finish(&mut metrics).unwrap();
    Ok(())
}

#[derive(Debug)]
struct Rule<'a> {
    dimension: &'a str,
    cutoff: f64,
    metric: f64,
}

// categorical types are mapped to u32 because:
// 1. do not are equivalent to rust enums which are actually sum types
// 2. we target also 64bit execution platforms like webasm
#[derive(Debug)]
struct Decision<'a> {
    rule: Option<Rule<'a>>,
    confidence: f64,
    prediction: u32,
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
        let selection: &UInt32Chunked = data.column(target).unwrap().u32().unwrap();
        let node = predict_majority(&mut selection.iter())?;
        Some(btree::Node::new(node))
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
fn gini(counts: &[u32]) -> f64 {
    let sum: u32 = counts.iter().sum();
    let result: f64 = 1.0
        - counts
            .iter()
            .map(|c| ((*c as f64) / (sum as f64)).powi(2))
            .sum::<f64>();
    result
}

fn dataframe_gini(data: DataFrame, target: &str) -> f64 {
    let labels = data
        .column(target)
        .unwrap()
        .categorical()
        .unwrap()
        .physical();
    let groups = count_groups(&mut labels.iter());
    let groups_count: Vec<u32> = groups.values().into_iter().map(|s| *s).collect();
    gini(&groups_count)
}

fn count_groups(values: &mut dyn Iterator<Item = Option<u32>>) -> HashMap<u32, u32> {
    values
        .filter_map(|s| s)
        .fold(HashMap::new(), |mut result, value| {
            result.insert(value, result.get(&value).unwrap_or(&0) + 1);
            result
        })
}

// when creating a node first check which would be the prodicted outcome
fn predict_majority<'a>(values: &mut dyn Iterator<Item = Option<u32>>) -> Option<Decision<'a>> {
    let summary: HashMap<u32, u32> = count_groups(values);
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

fn evaluate_metric(data: &DataFrame, feature: &str, target: &str) -> PolarsResult<DataFrame> {
    let values = data.column(feature)?;
    let unique = values.unique()?;
    let result = df!(feature => unique)?
        .lazy()
        .with_columns([((col(feature) + col(feature).shift(lit(-1))) / lit(2.0)).alias("split")])
        .collect()?;
    let split = result.column("split")?.f64()?;
    let metrics: Series = split
        .iter()
        .map(|spm| {
            if let Some(sp) = spm {
                let higher = data.clone().filter(&values.gt_eq(sp).ok()?).ok()?;
                let lower = data.clone().filter(&values.lt(sp).ok()?).ok()?;
                Some(
                    ((higher.shape().0 as f64) * dataframe_gini(higher, target)
                        + (lower.shape().0 as f64) * dataframe_gini(lower, target))
                        / (values.len() as f64),
                )
            } else {
                None
            }
        })
        .collect();
    return Ok(df!(
            "split" => split.clone().into_series(),
            "metrics" => metrics,
    )?);
}

#[cfg(test)]
mod test {
    use crate::{count_groups, predict_majority};

    #[test]
    fn test_count_groups() {
        let input: [Option<u32>; 14] = [
            Some(1u32),
            Some(1u32),
            Some(3u32),
            Some(2u32),
            None,
            Some(1u32),
            None,
            Some(2u32),
            Some(3u32),
            None,
            Some(2u32),
            Some(2u32),
            None,
            None,
        ];
        let result = count_groups(&mut input.iter().map(|s| *s));
        println!("{:?}", result);
        assert_eq!(result.get(&2u32), Some(&4u32));
    }

    #[test]
    fn test_predict_majority() {
        let input: [Option<u32>; 14] = [
            Some(1u32),
            Some(1u32),
            Some(3u32),
            Some(2u32),
            None,
            Some(1u32),
            None,
            Some(2u32),
            Some(3u32),
            None,
            Some(2u32),
            Some(2u32),
            None,
            None,
        ];
        let result = predict_majority(&mut input.iter().map(|s| *s));
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.prediction, 2u32);
        assert!(result.confidence < 0.5);
        assert!(result.confidence > 0.4);
    }
}
