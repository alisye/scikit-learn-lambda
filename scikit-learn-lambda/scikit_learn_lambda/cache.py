import pickle
import os
import tempfile
import boto3
import joblib

from . import s3_url
from . import model_explainer


def load_model_from_file(model_file):
    if model_file.endswith(".pkl") or model_file.endswith(".pickle"):
        with open(model_file, "rb") as f:
            return pickle.load(f)
    elif model_file.endswith(".joblib"):
        return joblib.load(model_file)


def load_model_from_path(model_path):
    if model_path.startswith("s3://"):
        url = s3_url.S3Url(model_path)
        obj = boto3.resource("s3").Object(url.bucket, url.key)
        with tempfile.TemporaryDirectory() as td:
            local_path = os.path.join(td, url.filename)
            obj.download_file(local_path)
            return load_model_from_file(local_path)
    else:
        return load_model_from_file(model_path)


class Cache:
    __model = None
    __scaler = None
    __explainer = None

    @staticmethod
    def get_model():
        if Cache.__model is None:
            Cache.__model = load_model_from_path(os.environ["SKLEARN_MODEL_PATH"])

        return Cache.__model

    @staticmethod
    def get_scaler():
        if Cache.__scaler is None:
            Cache.__scaler = load_model_from_path(os.environ["SKLEARN_SCALER_PATH"])

        return Cache.__scaler

    @staticmethod
    def get_explainer():
        if Cache.__explainer is None:
            Cache.__explainer = model_explainer.LinearExplainer(Cache.get_model())

        return Cache.__explainer

    @staticmethod
    def clear():
        Cache.__model = None
        Cache.__scaler = None
        Cache.__explainer = None
