# Databento ES OHLCV-1s Data Downloader Setup

This guide helps you set up and use the Databento API to download E-mini S&P 500 (ES) futures OHLCV-1s data.

## 🚀 Quick Start

### 1. Get Your Databento API Key

1. Sign up for a free account at [Databento](https://databento.com)
2. Navigate to the [API Keys page](https://databento.com/portal/api-keys)
3. Copy your API key (it should start with `db-`)

### 2. Configure Your Environment

1. Open the `.env` file in this directory
2. Replace `your_databento_api_key_here` with your actual API key:
   ```
   DATABENTO_API_KEY=db-your_actual_api_key_here
   ```

### 3. Install Dependencies

```bash
# Install required packages
pip install -r requirements_databento.txt

# Or install individually:
pip install databento python-dotenv pandas numpy
```

### 4. Run the Downloader

```bash
# Basic usage - downloads recent ES data
python databento_es_downloader.py

# Or run the simple example
python databento_example.py
```

## 📚 Usage Examples

### Basic Download (Continuous Contract)
```python
from databento_es_downloader import DatabentoESDownloader

# Initialize downloader
downloader = DatabentoESDownloader()

# Download ES data for specific date range
df = downloader.download_es_ohlcv_1s(
    start_date='2024-01-15',
    end_date='2024-01-16',
    save_to_csv=True
)
```

### Download Specific Contract
```python
# Download specific ES contract (e.g., March 2024)
df = downloader.download_specific_contract(
    contract_symbol='ESH4',  # March 2024 contract
    start_date='2024-03-01',
    end_date='2024-03-02'
)
```

### Get Cost Estimate First
```python
# Check cost before downloading
cost = downloader.get_cost_estimate(
    symbols=['ES.n.0'],
    start_date='2024-01-15',
    end_date='2024-01-16'
)
```

## 🔧 Configuration Options

### Available Schemas
- `ohlcv-1s` - 1-second OHLCV bars
- `ohlcv-1m` - 1-minute OHLCV bars  
- `trades` - Individual trades
- `mbp-1` - Top of book quotes
- `mbp-10` - 10-level market depth

### Symbol Types
- `ES.n.0` - Continuous front month contract
- `ES.n.1` - Continuous second month contract
- `ESH4` - Specific contract (March 2024)
- `ESM4` - Specific contract (June 2024)

## 💰 Cost Information

- Databento offers $125 in free credits for new accounts
- OHLCV-1s data typically costs a few dollars per trading day
- Use `get_cost_estimate()` to check costs before downloading
- See [Databento Pricing](https://databento.com/pricing) for details

## 📁 Output Files

Downloaded data is saved to the `Data/` directory as CSV files:
- Format: `es_ohlcv_1s_YYYY-MM-DD_to_YYYY-MM-DD.csv`
- Contains columns: open, high, low, close, volume
- Timestamps are in UTC and indexed

## ⚠️ Important Notes

1. **Market Hours**: ES futures trade nearly 24/5, but be aware of maintenance windows
2. **Data Availability**: Historical data may have different availability depending on the date
3. **Rate Limits**: Databento has reasonable rate limits for API calls
4. **Time Zones**: All timestamps are in UTC
5. **File Size**: 1-second data can be quite large - consider your storage needs

## 🛠 Troubleshooting

### Common Issues

**"API key not found" error:**
- Check that your `.env` file contains the correct API key
- Ensure the key starts with `db-`
- Verify the `.env` file is in the same directory as your script

**"No data returned" error:**
- Check if the date range includes trading days
- Verify the symbol format is correct
- Ensure the requested time period has available data

**ImportError for databento:**
```bash
pip install databento
```

**Missing python-dotenv:**
```bash
pip install python-dotenv
```

### Getting Help

- [Databento Documentation](https://databento.com/docs)
- [Databento API Reference](https://databento.com/docs/api-reference-historical)
- [Databento Support](https://databento.com/support)

## 📊 Data Schema Reference

Based on [Databento's documentation](https://databento.com/blog/api-demo-python), here's what you'll get:

### OHLCV-1s Schema
- `ts_recv`: Receive timestamp (nanosecond precision)
- `open`: Opening price for the 1-second interval
- `high`: Highest price during the interval
- `low`: Lowest price during the interval  
- `close`: Closing price for the interval
- `volume`: Total volume traded during the interval

### ES Futures Specifications
- **Contract**: E-mini S&P 500 
- **Exchange**: CME Globex (GLBX.MDP3)
- **Tick Size**: $0.25 (1 point = $50)
- **Trading Hours**: Nearly 24/5 with brief maintenance windows

## 🔗 Useful Links

- [Databento Python Client](https://github.com/databento/databento-python)
- [ES Contract Specifications](https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.html)
- [Market Data Tutorial](https://databento.com/blog/api-demo-python)
- [Futures Introduction](https://databento.com/docs/examples/futures/futures-introduction) 