import numpy as np
import pandas as pd


def apply_missing_value_imputation(
    df: pd.DataFrame,
    artifacts: dict
) -> pd.DataFrame:

    imputer = artifacts["imputer"]

    if pd.isna(
        df.loc[0, "Booking Value"]
    ):

        vehicle = (
            str(
                df.loc[
                    0,
                    "Vehicle Type"
                ]
            )
        )

        pickup = (
            df.loc[
                0,
                "Pickup Location"
            ]
        )

        drop = (
            df.loc[
                0,
                "Drop Location"
            ]
        )

        rv_key = (
            f"{vehicle}|"
            f"{pickup}|"
            f"{drop}"
        )

        route = (
            f"{pickup}_{drop}"
        )

        rv_median = (
            imputer["rv_median"]
            .get(rv_key)
        )

        if rv_median is not None:

            df.loc[
                0,
                "Booking Value"
            ] = rv_median

        else:

            route_mean = (
                imputer[
                    "route_mean"
                ].get(route)
            )

            vehicle_scale = (
                imputer[
                    "vehicle_scale"
                ].get(vehicle)
            )

            if (
                route_mean is not None
                and
                vehicle_scale
                is not None
            ):

                df.loc[
                    0,
                    "Booking Value"
                ] = (
                    route_mean *
                    vehicle_scale
                )

            else:

                v_median = (
                    imputer[
                        "v_median"
                    ].get(vehicle)
                )

                if v_median is not None:

                    df.loc[
                        0,
                        "Booking Value"
                    ] = v_median

                else:

                    df.loc[
                        0,
                        "Booking Value"
                    ] = (
                        imputer[
                            "global_mean"
                        ]
                    )

    return df


def build_features(
    request_data: dict,
    artifacts: dict
) -> pd.DataFrame:

    df = pd.DataFrame(
        [request_data]
    )
    
    df.rename(
        columns={
            "Vehicle_Type": "Vehicle Type",
            "Pickup_Location": "Pickup Location",
            "Drop_Location": "Drop Location",
            "Booking_Value": "Booking Value"
        },
        inplace=True
    )

    df["Datetime"] = (
        pd.to_datetime(
            df["Datetime"]
        )
    )

    df = apply_missing_value_imputation(
        df,
        artifacts
    )

    df["day_of_week"] = (
        df["Datetime"]
        .dt.dayofweek
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["is_peak_hour"] = (
        df["Datetime"]
        .dt.hour
        .isin(
            [
                8,
                9,
                10,
                17,
                18,
                19,
                20
            ]
        )
    ).astype(int)

    vehicle_mapping = (
        artifacts[
            "vehicle_mapping"
        ]
    )

    df["Vehicle Type"] = (
        df["Vehicle Type"]
        .astype(str)
        .map(vehicle_mapping)
        .fillna(0)
    )

    pickup_freq = (
        artifacts[
            "pickup_freq"
        ]
    )

    drop_freq = (
        artifacts[
            "drop_freq"
        ]
    )

    route_avg_price = (
        artifacts[
            "route_avg_price"
        ]
    )

    route = (
        df["Pickup Location"]
        + "_"
        + df["Drop Location"]
    )

    df["location_demand"] = (

        df["Pickup Location"]
        .map(pickup_freq)
        .fillna(0)

        +

        df["Drop Location"]
        .map(drop_freq)
        .fillna(0)
    )

    route_avg = (
        route
        .map(route_avg_price)
        .fillna(
            np.mean(
                list(
                    route_avg_price
                    .values()
                )
            )
        )
    )

    df["price_vs_route"] = (
        df["Booking Value"]
        /
        route_avg
    )

    df["log_price"] = (
        np.log1p(
            df["Booking Value"]
        )
    )

    df["log_route_avg"] = (
        np.log1p(
            route_avg
        )
    )

    threshold = (
        df["Booking Value"]
        .median()
    )

    df["is_high_value"] = (
        df["Booking Value"]
        >
        threshold
    ).astype(int)

    model_features = [

        "Vehicle Type",

        "day_of_week",

        "is_weekend",

        "is_peak_hour",

        "price_vs_route",

        "log_price",

        "log_route_avg",

        "is_high_value",

        "location_demand"
    ]

    return df[
        model_features
    ]
    