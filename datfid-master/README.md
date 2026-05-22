---
title: DATFID MASTER
emoji: 🤖
colorFrom: indigo
colorTo: pink
sdk: docker
pinned: false
short_description: DATFID - Secure and powerful forecasting SDK
---

# DATFID MASTER DEMO (Hackathon Proxy)

`datfid-master` is the public proxy layer.
For the hackathon setup, it forwards all model calls to `datfid_api` and does not require the webpage demo flow.

We achieved **state-of-the-art accuracy in the M5 Forecasting Competition**, beated all the benchmarks, forecasting hierarchical Walmart product sales over a 5-year period using advanced exogenous features and rigorous error metrics ([Kaggle M5 page](https://www.kaggle.com/c/m5-forecasting-accuracy)).

---

##  Hackathon Setup

- Set `hf_url` to your public `datfid_api-demo` Space URL.
- No user/API token is required in this demo setup.
- Students call only `datfid-master`; orchestration stays unchanged.
- Backend internals in `datfid_api-demo` are simplified for demo use.

---

##  Getting Started

### 1. Install the SDK

```bash
pip install -i https://test.pypi.org/simple/ datfid
```

### 2. Example Usage

```python
import pandas as pd
from datfid.client import DATFIDClient

# Initialize client (no token needed in demo)
client = DATFIDClient()

# Fit the model
fit_result = client.fit_model(
    df=my_dataframe,
    id_col="SKU_ID",
    time_col="Date",
    y="Sales",
    lagged_features={"Sales": 2, "Promo": 1},
    current_features=["Price", "Holiday"],
    filter_by_significance=True,
    meanvar_test=True
)

# Forecast
forecast_df = client.forecast_model(df_forecast=my_forecast_dataframe)
print(forecast_df.head())

# Check service health
client.secure_ping()
```

---

##  Contact & Info

- Website: [datfid.com](https://datfid.com)  
- Demo setup: no token required

---

Happy forecasting with confidence!  
— The DATFID Team