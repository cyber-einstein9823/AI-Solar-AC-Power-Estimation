# Location and timestamp investigation

Status: **unresolved; Set B results are provisional**.

| Date | Sensor peak | Public peak | Public minus sensor |
|---|---:|---:|---:|
| 2020-05-20 | 12:00 | 12:00 | +0 h |
| 2020-06-01 | 10:00 | 11:00 | +1 h |
| 2020-06-15 | 10:00 | 13:00 | +3 h |

The sensor is converted from kW/m² to W/m² by multiplying by 1,000. The cached public series contains all 816 local hourly labels. Its requested configuration is 14.82° N, 78.28° E, Asia/Kolkata, 15 May–17 June 2020. The original response was not supplied, so its returned grid cell and timezone metadata cannot be independently authenticated from the CSV. A deliberate `--refresh` preserves a fresh response and its metadata.

The plant file has no timezone metadata. Asia/Kolkata is an explicit assignment assumption, not a discovered timezone. The [Open-Meteo historical documentation](https://open-meteo.com/en/docs/historical-weather-api) describes shortwave radiation as the preceding-hour mean; the sensor aggregation here uses a left-labelled hour. Temperature and cloud cover have different, instantaneous semantics. This interval mismatch is a plausible contributor, but the raw sensor averaging convention is not documented, so relabelling all weather variables would not establish a correct alignment.

A separate diagnostic tests shifts from −3 to +3 hours using only source observations dated before 11 June. Positive shifts move the public timestamps later. Daytime correlation is 0.8875 with no shift; the highest is 0.8875 at a +0-hour shift. Correlation is not proof of geographic or temporal alignment, and these alternatives are not fed to training. The 15 June plot is descriptive, not used to select a correction.

No clock shift or coordinate search was applied to improve the held-out results. Resolving this requires confirmed plant coordinates and sensor interval metadata, followed by a predeclared temporal convention and a fresh untouched evaluation period. Until then, this is a transparent limitation—not a claim that the location-check requirement is fully resolved.
