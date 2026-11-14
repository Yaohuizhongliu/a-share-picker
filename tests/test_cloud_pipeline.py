import json

import pandas as pd

from ashare_picker.cloud_pipeline import _parse_tencent_price_payload, clean_price_frame


def test_tencent_jsonp_parser_normalizes_units_and_columns():
    symbol = "sz000001"
    row = ["2026-09-30", "10.00", "10.20", "10.30", "9.90", "1234", "x", "1.2", "5678"]
    text = "kline_day2026=" + json.dumps({"data": {symbol: {"day": [row]}}})
    parsed = _parse_tencent_price_payload(text, symbol)
    cleaned = clean_price_frame(
        parsed, pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-30")
    )
    assert list(cleaned.columns) == [
        "date", "open", "high", "low", "close", "volume", "amount"
    ]
    assert cleaned.iloc[0]["close"] == 10.20
    assert cleaned.iloc[0]["volume"] == 123400
    assert cleaned.iloc[0]["amount"] == 56780000
