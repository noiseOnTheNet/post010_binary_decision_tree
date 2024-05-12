use polars::lazy::dsl::Expr;
use polars::prelude::*;
use polars::series::Series;
use std::fs::File;
use std::ops::Deref;

fn main() -> polars::prelude::PolarsResult<()> {
    let features = ["sepal_length", "sepal_width", "petal_length", "petal_width"];
    let target = "variety";

    // read data file
    let mut data = CsvReader::from_path("iris.csv")?
        .has_header(true)
        .finish()?;
    println!("\ndata\n{1:->0$}{2:?}{1:-<0$}\n",20,"\n",data);

    // set target column as categorical
    data.try_apply(target, |s| {
        s.cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
    })?;
    println!("\ndata\n{1:->0$}{2:?}{1:-<0$}\n",20,"\n",data);

    // iteratively evaluate the metric on all features
    let metrics: PolarsResult<Vec<LazyFrame>> = features
        .iter()
        .map(|feature|
                    Ok(evaluate_metric(&data, feature, target)?
                    .lazy()
                    .with_column(feature.lit().alias("feature")))
        )
        .collect();

    // join all results in a single dataframe
    let concat_rules = UnionArgs {
        parallel: true,
        rechunk: true,
        to_supertypes: true,
    };
    let mut concat_metrics: DataFrame = concat(metrics?,concat_rules)?
        .collect()?;
    println!("\nconcat_metrics\n{1:->0$}{2:?}{1:-<0$}\n",20,"\n",concat_metrics);

    // search for the best split
    let expr : Expr = col("metrics").lt_eq(col("metrics").min());
    let best_split : LazyFrame= concat_metrics
        .clone()
        .lazy()
        .filter(expr)
        .select([col("feature"),col("split"),col("metrics")]);
    println!("\nbest_split\n{1:->0$}{2:?}{1:-<0$}\n",20,"\n",best_split.collect()?);

    // save the metrics into a file
    let buffer = File::create("metrics.csv").unwrap();
    CsvWriter::new(buffer).finish(&mut concat_metrics).unwrap();
    let majority_class = predict_majority_dataframe(& data, target)?;
    println!("\nmajority_class\n{1:->0$}{2:?}{1:-<0$}\n",20,"\n",majority_class);
    Ok(())
}


// Gini impurity metric
fn estimate_gini(data: & DataFrame, target: & str) -> PolarsResult<f64> {
    let label_count: DataFrame = data
        .column(target)?
        .categorical()?
        .value_counts()?;

    let expr: Expr = (col("counts")
        .cast(DataType::Float64)
        / col("counts").sum())
        .pow(2)
        .alias("squares");

    let squared: DataFrame = label_count
        .lazy()
        .select([expr])
        .collect()?;

    let square_sum: f64 = squared
        .column("squares")?
        .sum()?;

    Ok(1.0 - square_sum)
}

fn predict_majority_dataframe<'a>(data: & 'a DataFrame, target: &str) -> PolarsResult<String>{
    // extract the categorical target column
    let labels = data
        .column(target)?
        .categorical()?;

    // count all categories and sort them
    let result_count = labels.value_counts()?;
    println!("{1:->0$}{2:?}{1:-<0$}",20,"\n",result_count);

    // get the most frequent category
    let result_cat = result_count
        .column(target)?
        .head(Some(1));
    println!("{1:->0$}{2:?}{1:-<0$}",20,"\n",result_cat);

    // transform the series into a categorical vector
    let actual_cat= result_cat
        .categorical()?;

    // collect all categories as strings
    let string_cat: Vec<String>=actual_cat
        .iter_str()
        .flatten()
        .map(|name| (*name).into())
        .collect();
    println!("{1:->0$}{2:?}{1:-<0$}",20,"\n",string_cat);

    // return the most common category as a string
    return Ok(string_cat.get(0)
        .unwrap()
        .deref()
        .into());
}


fn evaluate_metric(data: &DataFrame, feature: &str, target: &str) -> PolarsResult<DataFrame> {
    // grabs the unique values
    let values = data.column(feature)?;
    let unique = values.unique()?;

    // create a lagged column to identify split points
    let split = df!(feature => unique)?
        .lazy()
        .with_columns([(
            (col(feature) + col(feature).shift(lit(-1))) /
                lit(2.0)).alias("split")
        ])
        .collect()?;
    let split_values : Vec<f64> = split
        .column("split")?
        .f64()?
        .iter()
        .flatten() // drop missing values created by lag
        .collect();

    // iterate over split points
    let metrics: PolarsResult<Series> = split_values
        .iter()
        .map(|sp| {
            // split dataframe
            let higher = data.clone().filter(& values.gt_eq(*sp)?)?;
            let lower = data.clone().filter(& values.lt(*sp)?)?;

            // calculate metrics
            let higher_metric = estimate_gini(& higher, target)?;
            let lower_metric = estimate_gini(& lower, target)?;

            Ok(
                ((higher.shape().0 as f64) * higher_metric
                 + (lower.shape().0 as f64) * lower_metric)
                    / (values.len() as f64),
            )
        })
        .collect();

    // return a dataframe with a metric evaluation
    // for each split point
    return Ok(df!(
        "split" => Series::new("split", split_values),
        "metrics" => metrics?,
    )?);
}

