import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch
from ra_testbed.data.loader import DataLoader


def make_fake_yf_data(n: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": np.full(n, 100.0),
            "High": np.full(n, 105.0),
            "Low": np.full(n, 95.0),
            "Close": np.full(n, 102.0),
            "Volume": np.full(n, 1_000_000.0),
        },
        index=dates,
    )


class TestDataLoader:
    def test_downloads_on_cache_miss(self, tmp_path):
        fake = make_fake_yf_data()
        with patch("ra_testbed.data.loader.yf.download", return_value=fake) as mock_dl:
            DataLoader(tickers=["SPY"], cache_dir=str(tmp_path)).load("2020-01-01", "2020-06-30")
            assert mock_dl.call_count == 1

    def test_reads_cache_on_second_call(self, tmp_path):
        fake = make_fake_yf_data()
        with patch("ra_testbed.data.loader.yf.download", return_value=fake):
            DataLoader(tickers=["SPY"], cache_dir=str(tmp_path)).load("2020-01-01", "2020-06-30")

        with patch("ra_testbed.data.loader.yf.download") as mock_dl2:
            DataLoader(tickers=["SPY"], cache_dir=str(tmp_path)).load("2020-01-01", "2020-06-30")
            mock_dl2.assert_not_called()

    def test_cache_file_created(self, tmp_path):
        fake = make_fake_yf_data()
        with patch("ra_testbed.data.loader.yf.download", return_value=fake):
            DataLoader(tickers=["TLT"], cache_dir=str(tmp_path)).load("2020-01-01", "2020-06-30")
        assert (tmp_path / "TLT.parquet").exists()

    def test_returns_correct_tickers(self, tmp_path):
        def fake_download(ticker, **kwargs):
            return make_fake_yf_data()

        with patch("ra_testbed.data.loader.yf.download", side_effect=fake_download):
            loader = DataLoader(tickers=["SPY", "TLT", "GLD"], cache_dir=str(tmp_path))
            close, open_ = loader.load("2020-01-01", "2020-12-31")

        assert set(close.columns) == {"SPY", "TLT", "GLD"}
        assert set(open_.columns) == {"SPY", "TLT", "GLD"}

    def test_date_range_slicing(self, tmp_path):
        fake = make_fake_yf_data(n=500)
        with patch("ra_testbed.data.loader.yf.download", return_value=fake):
            close, _ = DataLoader(tickers=["SPY"], cache_dir=str(tmp_path)).load(
                "2020-06-01", "2020-09-30"
            )

        assert all(close.index >= pd.Timestamp("2020-06-01"))
        assert all(close.index <= pd.Timestamp("2020-09-30"))

    def test_output_is_datetime_index(self, tmp_path):
        fake = make_fake_yf_data()
        with patch("ra_testbed.data.loader.yf.download", return_value=fake):
            close, open_ = DataLoader(tickers=["SPY"], cache_dir=str(tmp_path)).load(
                "2020-01-01", "2020-12-31"
            )
        assert isinstance(close.index, pd.DatetimeIndex)
        assert isinstance(open_.index, pd.DatetimeIndex)
