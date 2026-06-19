from pathlib import Path
import pandas as pd
import yfinance as yf


class DataLoader:
    """
    yfinance로 과거 시세를 수집하고 parquet으로 로컬 캐싱.
    캐시 히트 시 네트워크 호출 없이 파일에서 로드, 미스 시 전체 히스토리 다운로드 후 저장.
    load()는 (종가 DataFrame, 시가 DataFrame) 튜플 반환.
    """

    def __init__(self, tickers: list[str], cache_dir: str = "data/"):
        self.tickers = tickers
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """DatetimeIndex × ticker의 (종가, 시가) DataFrame 튜플 반환."""
        close_frames, open_frames = [], []

        for ticker in self.tickers:
            df = self._load_ticker(ticker)
            sliced = df.loc[start:end]
            close_frames.append(sliced["Close"].rename(ticker))
            open_frames.append(sliced["Open"].rename(ticker))

        close = pd.concat(close_frames, axis=1).dropna()
        open_ = pd.concat(open_frames, axis=1).dropna()
        return close, open_

    def _load_ticker(self, ticker: str) -> pd.DataFrame:
        cache_path = self.cache_dir / f"{ticker}.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
        else:
            df = yf.download(ticker, start="2000-01-01", auto_adjust=True, progress=False)
            # yfinance 버전에 따라 MultiIndex 컬럼이 반환될 수 있음
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.to_parquet(cache_path)

        df.index = pd.to_datetime(df.index)
        return df
