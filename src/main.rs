use polars::lazy::dsl::first;
use polars::prelude::*;
use polars::series::Series;
use std::fs::File;
use decision::count_groups;

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
    predict_majority_dataframe(& data, target);
    let mut metrics =  metrics?;
    let buffer = File::create("metrics.csv").unwrap();
    CsvWriter::new(buffer).finish(&mut metrics).unwrap();
    Ok(())
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

fn dataframe_gini(data: & DataFrame, target: &str) -> f64 {
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

fn predict_majority_dataframe(data: & DataFrame, target: &str){
    let labels = data
        .column(target)
        .unwrap()
        .categorical()
        .unwrap();
    let result_count = labels.value_counts()
        .unwrap();
    println!("{1:->0$}{2:?}{1:-<0$}",20,"\n",result_count);
    let result_cat = result_count
        .column(target)
        .unwrap()
        .head(Some(1));
    println!("{1:->0$}{2:?}{1:-<0$}",20,"\n",result_cat);
    let actual_cat = result_cat
        .categorical()
        .unwrap()
        .iter_str();
    println!("{1:->0$}{2:?}{1:-<0$}",20,"\n",actual_cat);
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
                    ((higher.shape().0 as f64) * dataframe_gini(& higher, target)
                        + (lower.shape().0 as f64) * dataframe_gini(& lower, target))
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

