# Table 1 - preparation and timestamp checks

| Measure | Value | Note |
|---|---:|---|
| Raw Plant 1 generation rows | 68778 | Inverter-level records |
| Raw Plant 1 sensor rows | 3182 | One sensor record per timestamp |
| Unique generation timestamps | 3158 | After inverter summation |
| Generation-only timestamps | 1 | Excluded before hourly averaging |
| Sensor-only timestamps | 25 | Excluded before hourly averaging |
| Total unmatched timestamps | 26 | Outer-join audit only |
| Matched quarter-hour timestamps | 3157 | Inner join used for all hourly variables |
| Observed inverter identifiers | 22 | Expected coverage benchmark |
| Timestamps below full inverter coverage | 101 | Retained; not scaled up |
| Hourly calendar rows | 816 | 15 May-17 June, inclusive |
| Hours with missing values | 20 | Preserved as NaN; excluded during model fitting |
| Missing hourly numeric cells | 100 | Across five measured variables |
| Complete hourly rows | 796 | Before joining complete weather data |
| Hours containing only 1-3 matched quarters | 14 | Means use available matched measurements |
| Peak hours 2020-05-20 | sensor 12:00; public 12:00 | Public minus sensor +0 h; unresolved |
| Peak hours 2020-06-01 | sensor 10:00; public 11:00 | Public minus sensor +1 h; unresolved |
| Peak hours 2020-06-15 | sensor 10:00; public 13:00 | Public minus sensor +3 h; unresolved |
