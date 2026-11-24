"""Train and evaluate one-session forecasts using daily adjusted stock bars."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from xgboost import XGBRegressor


TICKERS = ("AAPL", "JPM", "JNJ", "WMT", "XOM")
START = "2010-01-01"
END = "2026-01-01"  # Exclusive
TEST_YEAR = 2023
VALIDATION_YEARS = (2020, 2021, 2022)
LOOKBACK = 20
BASIS_POINTS = 10_000
SEED = 42
BAR_COLUMNS = ["open", "high", "low", "close", "volume"]
CACHE_COLUMNS = ["date", "ticker", *BAR_COLUMNS, "source", "retrieved_at_utc"]
NO_CHANGE = "No change"
MODEL_SETTINGS = {
    "Ridge": [{"alpha": alpha} for alpha in (1, 100, 1000)],
    "XGBoost": [
        {"max_depth": depth, "n_estimators": trees}
        for depth, trees in product((2, 3), (100, 250))
    ],
    "LSTM": [
        {"hidden_size": width, "epochs": epochs}
        for width, epochs in ((16, 8), (16, 16), (32, 16))
    ],
}
MODELS = (NO_CHANGE, *MODEL_SETTINGS)


def configure_cpu():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def download_yahoo_bars():
    import yfinance as yf

    frames = []
    for ticker in TICKERS:
        history = yf.Ticker(ticker).history(
            start=START, end=END, interval="1d", auto_adjust=True,
            actions=False, raise_errors=True, timeout=20,
        )
        if history.empty:
            raise ValueError(f"Yahoo Finance returned no rows for {ticker}")
        history.index = history.index.tz_localize(None).normalize()
        bars = history.rename(columns=str.lower)[BAR_COLUMNS].reset_index(names="date")
        frames.append(bars.assign(ticker=ticker))
    return pd.concat(frames, ignore_index=True), f"Yahoo Finance via yfinance {yf.__version__}"


def parse_yahoo_history_page(page, ticker):
    from lxml import html

    chart_scripts = html.fromstring(page).xpath(
        "//script[@type='application/json' and contains(@data-url, '/v8/finance/chart/')]"
    )
    for script in chart_scripts:
        payload = json.loads(script.text)
        results = json.loads(payload["body"])["chart"]["result"]
        for result in results or []:
            if result["meta"]["symbol"] != ticker or result["meta"]["dataGranularity"] != "1d":
                continue
            indicators = result["indicators"]
            quote = pd.DataFrame(indicators["quote"][0])
            adjusted_close = pd.Series(indicators["adjclose"][0]["adjclose"])
            factor = adjusted_close / quote["close"]
            quote[["open", "high", "low"]] = quote[["open", "high", "low"]].mul(factor, axis=0)
            quote["close"] = adjusted_close
            dates = pd.to_datetime(result["timestamp"], unit="s", utc=True)
            dates = dates.tz_convert(result["meta"]["exchangeTimezoneName"])
            return quote[BAR_COLUMNS].assign(date=dates.tz_localize(None).normalize(), ticker=ticker)
    raise ValueError(f"Full-precision chart data is unavailable on Yahoo Finance's {ticker} history page")


def download_yahoo_history_pages():
    import requests

    start, end = (int(pd.Timestamp(date, tz="UTC").timestamp()) for date in (START, END))
    frames = []
    with requests.Session() as session:
        for ticker in TICKERS:
            url = f"https://finance.yahoo.com/quote/{ticker}/history/?period1={start}&period2={end}"
            response = session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            response.raise_for_status()
            frames.append(parse_yahoo_history_page(response.content, ticker))
    return pd.concat(frames, ignore_index=True), "Yahoo Finance historical pages (embedded daily chart)"


def download_bars():
    try:
        bars, source = download_yahoo_bars()
        validate_bars(bars, exchange_sessions())
        return bars, source
    except Exception as api_error:
        try:
            return download_yahoo_history_pages()
        except Exception as page_error:
            raise RuntimeError(
                f"Yahoo Finance download failed via yfinance ({api_error}) "
                f"and the historical pages ({page_error})."
            ) from page_error


def exchange_sessions():
    # Include the session following the snapshot to demonstrate a forecast without a label.
    calendar = xcals.get_calendar("XNYS", start=START, end="2026-01-10")
    return calendar.sessions


def read_snapshot(path):
    bars = pd.read_csv(path, parse_dates=["date"], float_precision="round_trip")
    for column in ("source", "retrieved_at_utc"):
        if bars[column].isna().any() or bars[column].nunique() != 1:
            raise ValueError(f"Expected one {column} for the entire snapshot")
    validate_bars(bars, exchange_sessions())
    return bars


def load_snapshot(path, download=download_bars):
    if path.exists():
        return read_snapshot(path)

    bars, source = download()
    bars = bars.assign(
        source=source, retrieved_at_utc=datetime.now(timezone.utc).isoformat()
    )[CACHE_COLUMNS].sort_values(["date", "ticker"])
    validate_bars(bars, exchange_sessions())
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        bars.to_csv(temporary_path, index=False)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return read_snapshot(path)


def snapshot_metadata(path, bars):
    return {
        "source": bars["source"].iat[0],
        "retrieved_at_utc": bars["retrieved_at_utc"].iat[0],
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "first_bar": bars["date"].min().date().isoformat(),
        "last_bar": bars["date"].max().date().isoformat(),
        "rows": len(bars),
    }


def validate_bars(bars, sessions):
    if set(bars["ticker"]) != set(TICKERS):
        raise ValueError("The snapshot must cover all five study tickers")
    if bars.duplicated(["ticker", "date"]).any():
        raise ValueError("The snapshot contains duplicate ticker and date pairs")
    expected = sessions[sessions < END]
    if not bars["date"].isin(expected).all():
        raise ValueError("The snapshot contains dates outside the study's exchange sessions")
    observed = bars.dropna(subset=BAR_COLUMNS).groupby("ticker")["date"].agg(["min", "max", "count"])
    coverage = observed.reindex(TICKERS)
    full_span = coverage["min"].eq(expected[0]) & coverage["max"].eq(expected[-1])
    enough_rows = coverage["count"].ge(0.95 * len(expected))
    if not (full_span & enough_rows).all():
        raise ValueError(f"Incomplete snapshot: {coverage['count'].to_dict()} of {len(expected)} sessions")


def daily_features(bars):
    log_close = np.log(bars["close"])
    one_day_return = log_close.diff()
    volume = bars["volume"]
    return pd.DataFrame(
        {
            "return_1d": one_day_return,
            "return_5d": log_close.diff(5),
            "volatility_20d": one_day_return.rolling(20).std(ddof=0),
            "intraday_range": np.log(bars["high"] / bars["low"]),
            "intraday_return": np.log(bars["close"] / bars["open"]),
            "relative_volume": np.log(volume / volume.rolling(20).median()),
        },
        index=bars.index,
    )


@dataclass
class Samples:
    rows: pd.DataFrame
    windows: np.ndarray

    def select(self, mask):
        mask = np.asarray(mask, dtype=bool)
        return Samples(self.rows.loc[mask].reset_index(drop=True), self.windows[mask])

    @property
    def returns(self):
        return self.rows["actual_return_bps"].to_numpy()


def build_samples(bars, sessions):
    """Keep valid feature windows even if their next-session outcome is unavailable."""
    row_blocks, window_blocks = [], []
    for ticker, group in bars.groupby("ticker", sort=True):
        aligned = group.set_index("date").reindex(sessions)[BAR_COLUMNS].astype(float)
        usable = np.isfinite(aligned).all(axis=1) & aligned.gt(0).all(axis=1)
        aligned.loc[~usable, :] = np.nan
        features = daily_features(aligned).to_numpy()
        windows = np.lib.stride_tricks.sliding_window_view(features, LOOKBACK, axis=0)
        windows = windows.swapaxes(1, 2)
        valid = np.isfinite(windows).all(axis=(1, 2))

        close = aligned["close"]
        target_close = close.shift(-1)
        rows = pd.DataFrame(
            {
                "ticker": ticker,
                "origin": sessions,
                "target_date": pd.Series(sessions).shift(-1).to_numpy(),
                "origin_adjusted_close": close.to_numpy(),
                "actual_adjusted_close": target_close.to_numpy(),
                "actual_return_bps": BASIS_POINTS * np.log(target_close / close).to_numpy(),
            }
        )
        row_blocks.append(rows.iloc[LOOKBACK - 1:].loc[valid])
        window_blocks.append(windows[valid])

    rows = pd.concat(row_blocks, ignore_index=True)
    order = rows.sort_values(["target_date", "ticker"]).index.to_numpy()
    return Samples(rows.iloc[order].reset_index(drop=True), np.concatenate(window_blocks)[order])


def split_by_year(samples, year):
    target = samples.rows["target_date"]
    train = samples.select(target.dt.year < year)
    validation = samples.select(target.dt.year == year)
    assert train.rows["target_date"].max() <= validation.rows["origin"].min()
    return train, validation


def flat_windows(windows):
    return windows.reshape(len(windows), -1)


class LSTMRegressor(nn.Module):
    def __init__(self, feature_count, hidden_size):
        super().__init__()
        self.sequence = nn.LSTM(feature_count, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, windows):
        outputs, _ = self.sequence(windows)
        return self.output(outputs[:, -1]).squeeze(-1)


@dataclass
class ScaledLSTM:
    network: LSTMRegressor
    scaler: StandardScaler

    def transform(self, windows):
        values = self.scaler.transform(windows.reshape(-1, windows.shape[-1]))
        return torch.from_numpy(values.reshape(windows.shape).astype(np.float32))

    def predict(self, windows):
        self.network.eval()
        with torch.no_grad():
            return (100 * self.network(self.transform(windows))).numpy()


def fit_lstm(train, hidden_size, epochs):
    torch.manual_seed(SEED)
    feature_count = train.windows.shape[-1]
    scaler = StandardScaler().fit(train.windows.reshape(-1, feature_count))
    network = LSTMRegressor(feature_count, hidden_size)
    model = ScaledLSTM(network, scaler)
    inputs = model.transform(train.windows)
    targets = torch.from_numpy((train.returns / 100).astype(np.float32))
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    loss = nn.SmoothL1Loss(beta=0.5)
    for _ in range(epochs):
        for batch in torch.randperm(len(inputs)).split(128):
            optimizer.zero_grad()
            loss(network(inputs[batch]), targets[batch]).backward()
            optimizer.step()
    return model


def fit_model(name, settings, train):
    if name == "LSTM":
        return fit_lstm(train, **settings)
    if name == "Ridge":
        model = make_pipeline(StandardScaler(), Ridge(**settings))
    elif name == "XGBoost":
        model = XGBRegressor(
            **settings, objective="reg:absoluteerror", learning_rate=0.03,
            min_child_weight=20, tree_method="hist", n_jobs=1, random_state=SEED,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return model.fit(flat_windows(train.windows), train.returns)


def predict_returns(name, model, windows):
    if name == NO_CHANGE:
        return np.zeros(len(windows))
    inputs = windows if name == "LSTM" else flat_windows(windows)
    return model.predict(inputs).astype(float)


def select_settings(development):
    folds = {year: split_by_year(development, year) for year in VALIDATION_YEARS}
    validation_rows, selected = [], {}
    for name, candidates in MODEL_SETTINGS.items():
        candidate_scores = []
        for settings in candidates:
            errors = []
            for year, (train, validation) in folds.items():
                model = fit_model(name, settings, train)
                predicted = predict_returns(name, model, validation.windows)
                error = float(np.abs(predicted - validation.returns).mean())
                errors.append(error)
                validation_rows.append(
                    {"model": name, "settings": settings, "target_year": year, "mae_bps": error}
                )
            candidate_scores.append(np.mean(errors))
        selected[name] = candidates[int(np.argmin(candidate_scores))]
    for year, (_, validation) in folds.items():
        validation_rows.append({
            "model": NO_CHANGE, "settings": {}, "target_year": year,
            "mae_bps": np.abs(validation.returns).mean(),
        })
    return selected, pd.DataFrame(validation_rows)


def forecast(samples, fitted):
    outputs = []
    for name in MODELS:
        predicted = predict_returns(name, fitted.get(name), samples.windows)
        outputs.append(samples.rows.assign(
            model=name,
            predicted_return_bps=predicted,
            predicted_adjusted_close=(
                samples.rows["origin_adjusted_close"] * np.exp(predicted / BASIS_POINTS)
            ),
            predicted_direction=np.sign(predicted).astype(int),
        ))
    return pd.concat(outputs, ignore_index=True)


def score(predictions, majority_direction, by=None):
    results = []
    groups = ["model"] if by is None else ["model", by]
    for keys, group in predictions.groupby(groups, sort=False):
        name, *segment = keys
        errors = group["predicted_return_bps"] - group["actual_return_bps"]
        price_errors = group["predicted_adjusted_close"] - group["actual_adjusted_close"]
        moving = group.loc[group["actual_return_bps"].ne(0)]
        directions = np.sign(moving["actual_return_bps"])
        majority_accuracy = (directions == majority_direction).mean()
        results.append({
            "model": name,
            **({by: segment[0]} if by else {}),
            "n": len(group),
            "mae_bps": errors.abs().mean(),
            "rmse_bps": np.sqrt(np.square(errors).mean()),
            **({"adjusted_price_mae_usd": price_errors.abs().mean()} if by == "ticker" else {}),
            "direction_n": len(moving),
            "direction_accuracy": (
                np.nan if name == NO_CHANGE else moving["predicted_direction"].eq(directions).mean()
            ),
            "training_majority_accuracy": majority_accuracy,
        })
    table = pd.DataFrame(results)
    baseline = table.loc[table["model"].eq(NO_CHANGE)]
    baseline_mae = table[by].map(baseline.set_index(by)["mae_bps"]) if by else baseline["mae_bps"].iat[0]
    table["mae_change_vs_no_change_pct"] = (
        100 * (table["mae_bps"] / np.where(baseline_mae > 0, baseline_mae, np.nan) - 1)
    )
    return table


def check_alignment(bars, sessions, samples):
    cutoff = pd.Timestamp("2022-06-17")
    altered = bars.copy()
    future = altered["date"].gt(cutoff)
    altered.loc[future, BAR_COLUMNS] *= 2
    earlier = samples.select(samples.rows["origin"].le(cutoff))
    rebuilt = build_samples(altered, sessions)
    rebuilt = rebuilt.select(rebuilt.rows["origin"].le(cutoff))
    keys = ["ticker", "origin", "target_date"]
    pd.testing.assert_frame_equal(earlier.rows[keys], rebuilt.rows[keys])
    np.testing.assert_array_equal(earlier.windows, rebuilt.windows)

    # June 20, 2022 was the Juneteenth market holiday.
    friday = samples.rows.loc[samples.rows["origin"].eq(cutoff)]
    assert len(friday) == len(TICKERS)
    assert friday["target_date"].eq(pd.Timestamp("2022-06-21")).all()


def check_predictions(predictions, holdout):
    for name in MODELS:
        rows = predictions.loc[predictions["model"].eq(name), holdout.rows.columns]
        pd.testing.assert_frame_equal(rows.reset_index(drop=True), holdout.rows)
    assert np.isfinite(predictions[["predicted_return_bps", "predicted_adjusted_close"]]).all().all()
    baseline = predictions.loc[predictions["model"].eq(NO_CHANGE)]
    np.testing.assert_array_equal(
        baseline["predicted_adjusted_close"], baseline["origin_adjusted_close"]
    )
