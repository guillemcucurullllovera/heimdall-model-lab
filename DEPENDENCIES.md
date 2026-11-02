# Third-party boundary

No vendored implementation or third-party dataset is included. `uv.lock` pins the
installed distribution graph; `pyproject.toml` declares direct requirements.

| Dependency | License | Role |
|---|---|---|
| NumPy, SciPy, pandas | BSD-3-Clause | Arrays, numerical/statistical methods, tables |
| arch | NCSA | GARCH/EGARCH estimation and forecasting |
| statsmodels | BSD-3-Clause | Markov switching |
| scikit-learn | BSD-3-Clause | Covariance and Isolation Forest |
| pydantic | MIT | Contracts |
| httpx | BSD-3-Clause | Optional provider connections |
| python-dateutil | Apache-2.0 / BSD-3-Clause | Dates |
| orjson | Apache-2.0 / MIT | Serialization |
| click | BSD-3-Clause | CLI |
| pandas-market-calendars | MIT | Calendars |
| pytest, Hypothesis, ruff, mypy | MIT, MPL-2.0, MIT, MIT | Development only |
| skfolio, timeseriescv | BSD-3-Clause | Optional splitter test oracles |
| hatchling | MIT | Build backend |

Transitive licenses remain with their respective owners. No GPL/AGPL/SSPL runtime
dependency is intended. Network-provider API terms are separate from client-code
licensing and must be checked before downloading or redistributing data.
