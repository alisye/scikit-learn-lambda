import json
import numpy as np
from . import cache


def handler_response(status_code, response):
    body = json.dumps(response, sort_keys=True, default=str)
    return {
        "isBase64Encoded": False,
        "statusCode": status_code,
        "body": body,
    }


def convert_bytes_to_str(arr):
    # If the model response is a numpy byte-string, it cannot be serialized to
    # JSON as bytes. Assume a unicode encoding in this case.
    if isinstance(arr, np.ndarray) and arr.dtype.kind == "S":
        arr = arr.astype("U")
    return arr


def get_probabilities(model, input_):
    probabilities = model.predict_proba(input_)[:, 1]
    probabilities = convert_bytes_to_str(probabilities)
    return probabilities


def handler(event, _):
    try:
        model, scaler, explainer = cache.Cache.get_model(), cache.Cache.get_scaler(), cache.Cache.get_explainer()
    except Exception as e:
        return handler_response(500, {"error": "Failed to load model: {}".format(e)})

    try:
        body = json.loads(event["body"])
    except Exception as e:
        return handler_response(
            400, {"error": "Failed to parse request body as JSON: {}".format(str(e))},
        )

    if "input" not in body:
        return handler_response(
            400,
            {"error": "Failed to find an 'input' key in request body: {}".format(body)},
        )

    response = {}

    try:
        inputs = scaler.transform(body["input"])
        response["probabilities"] = get_probabilities(model, inputs)
        response["feature_weights"] = explainer.explain(inputs)
    except Exception as e:
        return handler_response(
            500, {"error": "Failed to get model probabilities: {}".format(str(e))},
        )

    return handler_response(200, response)
