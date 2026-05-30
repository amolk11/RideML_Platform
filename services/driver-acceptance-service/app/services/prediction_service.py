from app.inference.model_loader import (
    load_model,
    load_artifacts
)

from app.inference.feature_builder import (
    build_features
)

model = load_model()

artifacts = load_artifacts()


def predict(
    request_data: dict
):

    features = build_features(
        request_data,
        artifacts
    )

    probability = (
        model.predict_proba(
            features
        )[0][1]
    )

    prediction = int(
        probability >= 0.5
    )

    return {

        "prediction":
            prediction,

        "probability":
            float(
                probability
            )
    }
    