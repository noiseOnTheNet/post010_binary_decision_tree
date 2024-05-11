use polars::lazy::dsl::Expr;
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
    let features = ["sepal_length", "sepal_width", "petal_length", "petal_width"];
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

    let metrics: [LazyFrame; 4] = features.
        map(|feature|
            evaluate_metric(&data, feature, target)
                .unwrap()
                .lazy()
                .with_column(feature.lit().alias("feature"))
        );
    let concat_rules = UnionArgs {
        parallel: true,
        rechunk: true,
        to_supertypes: true,
    };
    let mut concat_metrics: DataFrame = concat(metrics,concat_rules)
        .unwrap()
        .collect()?;
    println!("\nconcat_metrics\n{1:->0$}{2:?}{1:-<0$}\n",20,"\n",concat_metrics);
    let expr : Expr = col("metrics").lt_eq(col("metrics").min());
    let best_split : LazyFrame= concat_metrics
        .clone()
        .lazy()
        .filter(expr)
        .select([col("feature"),col("split"),col("metrics")]);
    println!("\nbest_split\n{1:->0$}{2:?}{1:-<0$}\n",20,"\n",best_split.collect()?);
    let buffer = File::create("metrics.csv").unwrap();
    CsvWriter::new(buffer).finish(&mut concat_metrics).unwrap();
    predict_majority_dataframe(& data, target);
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

// calculate the gini impurity metric from
// the categorical target
fn dataframe_gini(data: & DataFrame, target: &str) -> f64 {
    // get the categorical target and returns it as u32 symbols
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

fn estimate_gini(data: & DataFrame, target: & str) -> f64 {
    let label_count: DataFrame = data
        .column(target)
        .unwrap()
        .categorical()
        .unwrap()
        .value_counts()
        .unwrap();
    // println!("\nlabel count\n{1:->0$}{2:?}{1:-<0$}\n",20,"\n",label_count);

    let expr: Expr = (col("counts")
        .cast(DataType::Float64)
        / col("counts").sum())
        .pow(2)
        .alias("squares");

    let squared: DataFrame = label_count
        .lazy()
        .select([expr])
        .collect()
        .unwrap();

    // println!("\nsquared\n{1:->0$}{2:?}{1:-<0$}\n",20,"\n",squared);

    let square_sum: f64 = squared
        .column("squares")
        .unwrap()
        .sum()
        .unwrap();

    1.0 - square_sum
}

fn predict_majority_dataframe<'a>(data: & 'a DataFrame, target: &str) -> String{
    // extract the categorical target column
    let labels = data
        .column(target)
        .unwrap()
        .categorical()
        .unwrap();

    // count all categories and sort them
    let result_count = labels.value_counts()
        .unwrap();
    println!("{1:->0$}{2:?}{1:-<0$}",20,"\n",result_count);

    // get the most frequent category
    let result_cat = result_count
        .column(target)
        .unwrap()
        .head(Some(1));
    println!("{1:->0$}{2:?}{1:-<0$}",20,"\n",result_cat);

    // transform the series into a categorical vector
    let actual_cat= result_cat
        .categorical()
        .unwrap();

    // collect all categories as strings
    let string_cat: Vec<Option<&str>>=actual_cat
        .iter_str()
        .collect();
    println!("{1:->0$}{2:?}{1:-<0$}",20,"\n",string_cat);

    // return the most common category as a string
    return string_cat.get(0)
        .unwrap()
        .unwrap()
        .into();
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
                    ((higher.shape().0 as f64) * estimate_gini(& higher, target)
                        + (lower.shape().0 as f64) * estimate_gini(& lower, target))
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

