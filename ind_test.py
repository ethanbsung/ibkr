from ib_insync import IB, Future
import pandas as pd
import numpy as np
import asyncio

class IBKRMarketDataAnalyzer:
    def __init__(self, host='127.0.0.1', port=7497, client_id=2):
        """
        Initialize connection to Interactive Brokers TWS/Gateway
        
        :param host: IB API host
        :param port: IB API port
        :param client_id: Unique client identifier
        """
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        
        # Technical analysis data storage
        self.price_history = []
        self.volume_history = []
        self.last_volume = 0  # Track previous volume for incremental VWAP

    async def connect(self):
        """Establish connection to IBKR API"""
        try:
            await self.ib.connectAsync(self.host, self.port, self.client_id)
            print("Connected to IBKR")
        except Exception as e:
            print(f"Connection error: {e}")

    def create_es_contract(self):
        """Create E-mini S&P 500 Futures contract"""
        return Future('ES', exchange='CME', lastTradeDateOrContractMonth='202503')

    async def stream_market_data(self, contract):
        """Stream real-time market data and perform technical analysis"""
        ticker = self.ib.reqMktData(contract)

        def on_price_change(ticker):
            if ticker.last:
                self.price_history.append(ticker.last)

                # Calculate incremental volume
                incremental_volume = ticker.volume - self.last_volume
                self.last_volume = ticker.volume

                # Only append positive incremental volume
                if incremental_volume > 0:
                    self.volume_history.append(incremental_volume)

                # Ensure enough data for calculations
                if len(self.price_history) > 14 and len(self.volume_history) > 0:
                    rsi = self.calculate_rsi()
                    vwap = self.calculate_vwap()
                    print(f"RSI: {rsi:.2f}, VWAP: {vwap:.2f}")

        ticker.updateEvent += on_price_change
        return ticker

    def calculate_rsi(self, periods=14):
        """Calculate RSI"""
        prices = pd.Series(self.price_history[-(periods + 1):])  # Get the last `periods + 1` prices
        delta = prices.diff()

        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Use Exponential Weighted Moving Average (EWMA) for smoothing
        avg_gain = gain.ewm(span=periods, min_periods=periods).mean()
        avg_loss = loss.ewm(span=periods, min_periods=periods).mean()

        # Calculate RSI
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        return rsi.iloc[-1] if not rsi.empty else np.nan

    def calculate_vwap(self):
        """Calculate Volume Weighted Average Price"""
        prices = np.array(self.price_history)
        volumes = np.array(self.volume_history)
        
        # Ensure volumes are not zero to avoid division by zero
        if np.sum(volumes) == 0:
            return np.nan

        return np.sum(prices * volumes) / np.sum(volumes)

async def main():
    analyzer = IBKRMarketDataAnalyzer()
    await analyzer.connect()
    
    es_contract = analyzer.create_es_contract()
    ticker = await analyzer.stream_market_data(es_contract)
    
    # Keep script running to receive updates
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
        analyzer.ib.disconnect()

if __name__ == "__main__":
    asyncio.run(main())