from services import data_loader
from utils.helpers import df_to_records, filter_df, paginate
from utils.nan_cleaner import clean_for_json


def get_forecast_overview():
    baseline = data_loader.get_baseline_results()
    fi = data_loader.get_feature_importance()
    forecast = data_loader.get_demand_forecast()
    return clean_for_json({
        "total_rows": len(forecast),
        "unique_feeders": int(forecast["feeder_id_hash"].nunique()),
        "baselines": df_to_records(baseline),
        "top_features": df_to_records(fi.head(10)),
    })


def get_forecast_by_feeder(feeder_id: str, test_only: bool = False):
    df = data_loader.get_demand_forecast()
    df = df[df["feeder_id_hash"] == feeder_id]
    if test_only:
        df = df[df["is_test_set"] == 1]
    if df.empty:
        return None
    return clean_for_json(df_to_records(df))


def get_forecast_timeseries(feeder_id: str = None, page: int = 1, page_size: int = 200):
    df = data_loader.get_demand_forecast()
    if feeder_id:
        df = df[df["feeder_id_hash"] == feeder_id]
    df = df.sort_values("timestamp_hour_ist")
    page_df, pagination = paginate(df, page, page_size)
    return clean_for_json({"data": df_to_records(page_df), "pagination": pagination})


def get_baseline_comparison():
    return clean_for_json(df_to_records(data_loader.get_baseline_results()))


def get_feature_importance():
    return clean_for_json(df_to_records(data_loader.get_feature_importance()))
