import logging
import warnings

import joblib
import pandas as pd
import typer

from build_features import BuildFeatures
from build_model import ModelBuilder
from config import config
from evaluate import Evaluate
from make_dataset import Ingestor, LabelEncoder, ProcessData
from predict import Predict
from utils import split_dataset

warnings.filterwarnings("ignore")

# Initialize Typer CLI app
app = typer.Typer()


@app.command()
def ingest(file_name: str = typer.Argument(..., help="Name of the file to ingest")):
    """Ingest data from data/raw folder"""
    logging.info("Ingesting data...")
    ingestor = Ingestor(file_name=file_name)
    df = ingestor.load_dataset()
    logging.info("✅ Ingested data!")
    return df


@app.command()
def encode(
    file_name: str = typer.Argument(..., help="Name of the file  ingest"),
):
    """Encode categorical columns"""
    logging.info("Encoding categorical columns...")
    df = ingest(file_name=file_name)
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols.remove("date")
    encoder = LabelEncoder(df=df, cat_cols=cat_cols)
    df = encoder.fit_transform()
    logging.info("✅ Encoded categorical columns!")
    return df


@app.command()
def process(
    file_name: str = typer.Argument(..., help="Name of the file "),
):
    """ """
    logging.info("Processing data...")
    df = encode(file_name=file_name)
    processor = ProcessData(df=df)
    df = processor.remove_null_values()
    # remove date, id, sku_id column
    df = df.drop(["id", "sku_id"], axis=1)
    df = df.drop("date", axis=1)
    logging.info("✅ Processed data!")
    return df


@app.command()
def buildfeature(
    file_name: str = typer.Argument(..., help="Name of the file "),
):
    """ """
    logging.info("Building features...")
    df = process(file_name=file_name)
    builder = BuildFeatures(df=df)
    df = builder.build_features()
    df.fillna(0, inplace=True)
    X_train, X_test, y_train, y_test = split_dataset(df=df)
    logging.info("✅ Built features!")
    return X_train, X_test, y_train, y_test


@app.command()
def buildmodel(
    file_name: str = typer.Argument(..., help="Name of the file "),
):
    """ """
    logging.info("Building model...")
    X_train, X_test, y_train, y_test = buildfeature(file_name=file_name)
    builder = ModelBuilder(X=X_train, y=y_train)
    model = builder.build_model()

    joblib.dump(model, "saved_models/model.pkl")
    logging.info("✅ Built model!")

    evals = Evaluate(model=model, x=X_test, y=y_test)
    evals.evaluate()
    logging.info("✅ Evaluated model!")

    predict = Predict(input_data=X_test, model=model)
    y_pred = predict.predict()

    # take out max sales value for each sku_id
    # merge x_test and y_pred
    y_pred = pd.DataFrame(y_pred, columns=["sales"])
    y_pred = pd.concat([X_test, y_pred], axis=1)
    # add sku_id column from df
    df = pd.read_csv(config.DATA_DIR / file_name)
    y_pred = pd.merge(
        y_pred, df[["sku_id", "sku_name", "cost"]], left_index=True, right_index=True
    )

    # group by sku_id, sku_name_y, cost and  take max sales value
    y_pred = y_pred.groupby(["sku_id", "sku_name_y", "cost_y"]).agg({"sales": "max"}).reset_index()
    y_pred.to_csv(config.DATA_DIR / "predictions.csv", index=False)
    logging.info("✅ Saved predictions!")

    return model


if __name__ == "__main__":
    app()
