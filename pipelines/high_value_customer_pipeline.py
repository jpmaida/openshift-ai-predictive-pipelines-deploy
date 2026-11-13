
from kfp import dsl, compiler, kubernetes


# ============================================================
# COMPONENT 1: FEATURE ENGINEERING
# ============================================================

@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas==2.2.3",
        "numpy==2.1.3",
        "openpyxl==3.1.5",
        "pyarrow==18.1.0",
        "boto3==1.35.76"
    ]
)
def feature_engineering(
    s3_endpoint: str,
    s3_bucket: str,
    s3_object_key: str,
    reference_date: str,
    prediction_days: int,
    output_dataset: dsl.Output[dsl.Dataset]
):
    import os
    import boto3
    import pandas as pd

    from botocore.config import Config

    # --------------------------------------------------------
    # Download dataset from MinIO
    # --------------------------------------------------------

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"}
        )
    )

    local_file = "/tmp/online_retail.xlsx"

    s3.download_file(
        s3_bucket,
        s3_object_key,
        local_file
    )

    df = pd.read_excel(local_file)

    print(f"Original records: {len(df)}")

    # --------------------------------------------------------
    # Data cleaning
    # --------------------------------------------------------

    df = df.dropna(subset=["CustomerID"]).copy()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = pd.to_numeric(
        df["CustomerID"], errors="coerce"
    )
    df["Quantity"] = pd.to_numeric(
        df["Quantity"], errors="coerce"
    )
    df["UnitPrice"] = pd.to_numeric(
        df["UnitPrice"], errors="coerce"
    )

    df = df.dropna(
        subset=["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"]
    )

    df = df[
        ~df["InvoiceNo"].astype(str).str.startswith("C")
    ]

    df = df[
        (df["Quantity"] > 0) &
        (df["UnitPrice"] > 0)
    ].copy()

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    # --------------------------------------------------------
    # Observation and prediction windows
    # --------------------------------------------------------

    reference = pd.Timestamp(reference_date)
    prediction_end = reference + pd.Timedelta(days=prediction_days)

    historical = df[df["InvoiceDate"] < reference].copy()

    future = df[
        (df["InvoiceDate"] >= reference) &
        (df["InvoiceDate"] < prediction_end)
    ].copy()

    if historical.empty:
        raise ValueError("No historical transactions found.")

    # --------------------------------------------------------
    # Order-level aggregation
    # --------------------------------------------------------

    orders = (
        historical
        .groupby(["CustomerID", "InvoiceNo"])
        .agg(
            OrderValue=("Revenue", "sum"),
            Items=("Quantity", "sum"),
            InvoiceDate=("InvoiceDate", "min")
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Customer features
    # --------------------------------------------------------

    features = (
        historical
        .groupby("CustomerID")
        .agg(
            Monetary=("Revenue", "sum"),
            TotalQuantity=("Quantity", "sum"),
            UniqueProducts=("StockCode", "nunique"),
            LastPurchase=("InvoiceDate", "max"),
            FirstPurchase=("InvoiceDate", "min"),
            SpendLast30Days=("Revenue", lambda x: 0)
        )
        .drop(columns=["SpendLast30Days"])
    )

    order_stats = (
        orders.groupby("CustomerID")
        .agg(
            Frequency=("InvoiceNo", "nunique"),
            AvgOrderValue=("OrderValue", "mean"),
            AvgItemsPerOrder=("Items", "mean")
        )
    )

    features = features.join(order_stats)

    features["Recency"] = (
        reference - features["LastPurchase"]
    ).dt.days

    features["CustomerTenure"] = (
        reference - features["FirstPurchase"]
    ).dt.days

    # Average interval between distinct order dates
    avg_intervals = {}

    for customer_id, group in orders.groupby("CustomerID"):
        dates = (
            group["InvoiceDate"]
            .drop_duplicates()
            .sort_values()
        )

        intervals = dates.diff().dt.total_seconds() / 86400

        avg_intervals[customer_id] = (
            intervals.mean() if len(dates) > 1 else 0
        )

    features["AvgDaysBetweenOrders"] = pd.Series(
        avg_intervals
    )

    # Spending windows
    spend_last_30 = historical[
        historical["InvoiceDate"] >=
        reference - pd.Timedelta(days=30)
    ].groupby("CustomerID")["Revenue"].sum()

    spend_previous_60 = historical[
        (historical["InvoiceDate"] >=
         reference - pd.Timedelta(days=90)) &
        (historical["InvoiceDate"] <
         reference - pd.Timedelta(days=30))
    ].groupby("CustomerID")["Revenue"].sum()

    features["SpendLast30Days"] = (
        spend_last_30.reindex(features.index).fillna(0)
    )

    features["SpendPrevious60Days"] = (
        spend_previous_60.reindex(features.index).fillna(0)
    )

    country = (
        historical.sort_values("InvoiceDate")
        .groupby("CustomerID")["Country"]
        .last()
    )

    features["Country"] = country

    # --------------------------------------------------------
    # Future revenue and target
    # --------------------------------------------------------

    future_revenue = (
        future.groupby("CustomerID")["Revenue"]
        .sum()
        .rename("FutureRevenue")
    )

    features = features.join(future_revenue)

    features["FutureRevenue"] = (
        features["FutureRevenue"].fillna(0)
    )

    positive_revenue = features.loc[
        features["FutureRevenue"] > 0,
        "FutureRevenue"
    ]

    if positive_revenue.empty:
        raise ValueError(
            "No customers with future revenue. "
            "Check reference_date and prediction_days."
        )

    threshold = positive_revenue.quantile(0.80)

    features["HighValueCustomer"] = (
        (features["FutureRevenue"] > 0) &
        (features["FutureRevenue"] >= threshold)
    ).astype(int)

    features = features.reset_index()

    print(f"Reference date: {reference}")
    print(f"Prediction end: {prediction_end}")
    print(f"Customers: {len(features)}")
    print(f"High value threshold: {threshold:.2f}")
    print(
        "Positive class:",
        int(features["HighValueCustomer"].sum())
    )

    # --------------------------------------------------------
    # Save dataset artifact
    # --------------------------------------------------------

    features.to_parquet(
        output_dataset.path,
        index=False
    )


# ============================================================
# COMPONENT 2: TRAINING, EVALUATION AND MODEL STORAGE
# ============================================================

@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas==2.2.3",
        "numpy==2.1.3",
        "pyarrow==18.1.0",
        "scikit-learn==1.9.0",
        "joblib==1.4.2",
        "boto3==1.35.76"
    ]
)
def train_evaluate_save(
    input_dataset: dsl.Input[dsl.Dataset],
    s3_endpoint: str,
    s3_bucket: str,
    model_prefix: str,
    output_metrics: dsl.Output[dsl.Artifact]
):
    import os
    import json
    import joblib
    import boto3
    import pandas as pd

    from botocore.config import Config

    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_parquet(input_dataset.path)

    target = "HighValueCustomer"

    excluded = [
        "CustomerID",
        "FutureRevenue",
        "HighValueCustomer",
        "LastPurchase",
        "FirstPurchase"
    ]

    X = df.drop(
        columns=excluded,
        errors="ignore"
    )

    y = df[target]

    if y.nunique() < 2:
        raise ValueError("Training dataset must contain both classes.")

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )
    }

    results = []
    predictions = {}
    trained_models = {}

    # --------------------------------------------------------
    # Training and evaluation
    # --------------------------------------------------------

    for model_name, classifier in models.items():

        print(f"Training {model_name}...")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)

        predicted = pipeline.predict(X_test)

        class_index = list(
            pipeline.classes_
        ).index(1)

        probabilities = pipeline.predict_proba(
            X_test
        )[:, class_index]

        metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, predicted),
            "Precision": precision_score(
                y_test, predicted, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predicted, zero_division=0
            ),
            "F1": f1_score(
                y_test, predicted, zero_division=0
            ),
            "ROC_AUC": roc_auc_score(
                y_test, probabilities
            ),
            "PR_AUC": average_precision_score(
                y_test, probabilities
            )
        }

        results.append(metrics)

        predictions[model_name] = pd.DataFrame({
            "Actual": y_test.to_numpy(),
            "Predicted": predicted,
            "Probability": probabilities
        })

        trained_models[model_name] = pipeline

        print(metrics)

    # --------------------------------------------------------
    # Select best model by PR-AUC
    # --------------------------------------------------------

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        "PR_AUC",
        ascending=False
    )

    best_model_name = comparison.iloc[0]["Model"]

    best_model = trained_models[best_model_name]

    print(f"Best model: {best_model_name}")

    # --------------------------------------------------------
    # Feature importance (Random Forest)
    # --------------------------------------------------------

    rf = trained_models["RandomForest"]

    feature_names = rf.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf.named_steps[
            "classifier"
        ].feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    # --------------------------------------------------------
    # Save local artifacts
    # --------------------------------------------------------

    os.makedirs("/tmp/model-artifacts", exist_ok=True)

    local_model = "/tmp/model-artifacts/best_model.joblib"

    joblib.dump(best_model, local_model)

    comparison.to_csv(
        "/tmp/model-artifacts/model_comparison.csv",
        index=False
    )

    importance.to_csv(
        "/tmp/model-artifacts/feature_importance.csv",
        index=False
    )

    for name, prediction_df in predictions.items():
        prediction_df.to_csv(
            f"/tmp/model-artifacts/predictions_{name}.csv",
            index=False
        )

    metadata = {
        "best_model": best_model_name,
        "selection_metric": "PR_AUC",
        "metrics": comparison.to_dict(orient="records")
    }

    with open(
        "/tmp/model-artifacts/model_metrics.json",
        "w"
    ) as file:
        json.dump(metadata, file, indent=2)

    # --------------------------------------------------------
    # Upload artifacts to MinIO
    # --------------------------------------------------------

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"}
        )
    )

    files = [
        "best_model.joblib",
        "model_comparison.csv",
        "feature_importance.csv",
        "model_metrics.json",
        "predictions_LogisticRegression.csv",
        "predictions_RandomForest.csv"
    ]

    for filename in files:
        local_path = f"/tmp/model-artifacts/{filename}"

        s3.upload_file(
            local_path,
            s3_bucket,
            f"{model_prefix}/{filename}"
        )

        print(f"Uploaded: {model_prefix}/{filename}")

    # KFP output artifact with evaluation information
    with open(output_metrics.path, "w") as file:
        json.dump(metadata, file, indent=2)


# ============================================================
# PIPELINE DEFINITION
# ============================================================

@dsl.pipeline(
    name="high-value-customer-training",
    description="Automated feature engineering, training and evaluation"
)
def high_value_customer_pipeline(
    s3_endpoint: str = "http://aistor.aistor.svc.cluster.local:9000",
    s3_bucket: str = "ml-artifacts",
    s3_object_key: str = "datasets/online-retail.xlsx",
    reference_date: str = "2011-09-01",
    prediction_days: int = 90,
    model_prefix: str = "models"
):

    features_task = feature_engineering(
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        s3_object_key=s3_object_key,
        reference_date=reference_date,
        prediction_days=prediction_days
    )

    training_task = train_evaluate_save(
        input_dataset=features_task.outputs["output_dataset"],
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        model_prefix=model_prefix
    )

    # Inject MinIO credentials into both components
    for task in [features_task, training_task]:
        kubernetes.use_secret_as_env(
            task,
            secret_name="minio-s3-access",
            secret_key_to_env={
                "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY"
            }
        )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=high_value_customer_pipeline,
        package_path="pipelines/high-value-customer-training.yaml"
    )

    print("Pipeline compiled successfully!")
