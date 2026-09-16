# Can public weather predict a solar plants hourly power

Our solar-power regression experiment found a substantial gap between on-site sensors and public weather. On the held-out daytime hours, the normal-equation model using plant sensors had an RMSE of 704.46 kW; the model using public weather had an RMSE of 3,409.50 kW. Public weather produced useful broad daily patterns, but these results do not justify precise operational or rooftop predictions. The public-weather location and time alignment also remain provisional.

## The question and the data

For AML Assignment 1, we asked whether publicly available weather could replace on-site measurements when estimating hourly aggregate AC power. We used Plant 1 from the Kaggle Solar Power Generation dataset, covering 15 May through 17 June 2020. This is a historical regression study of one plant, not a test of real-time forecasting or a comparison of different rooftop systems.

The generation file contains 68,778 inverter-level records and the sensor file contains 3,182 records. Their timestamp strings use different formats, so parsing them explicitly matters. We summed AC and DC power across available inverters at each timestamp, matched generation and sensor observations at quarter-hour resolution, and then calculated hourly means. Matching first ensures that each hourly target and its sensor features summarize the same observations.

The calendar has 816 hours, of which 20 lack complete model inputs or targets. We retained those gaps in the hourly file and excluded them from fitting instead of treating missing measurements as zero. We also recorded 26 unmatched timestamps, 101 generation timestamps without all 22 inverters, and 14 partial hours. These limitations affect what the aggregate power measurements represent.

## What the exploratory plots showed

AC power and measured irradiation are strongly associated, with an hourly correlation of approximately 0.998. Modules are generally hotter than the surrounding air during daylight; the median module-minus-ambient difference is 11.30 degrees Celsius. The mean daily power profile peaks at 11:00. These patterns support using irradiation, temperature and cyclic hour information, although an association alone does not prove a physical mechanism.

![Figure 1 AC power rises with measured irradiation](../results/plot1_ac_vs_irradiation.png)

The AC-versus-DC plot exposed a separate caution. The median raw AC/DC ratio is about 0.0978 for positive-DC hours. We retained the supplied scale and did not describe this number as verified inverter efficiency. Resolving the original units or scaling would require additional source information; multiplying values merely to obtain a familiar efficiency would be unjustified.

## Comparing features fairly

Set A contains on-site irradiation, module temperature and ambient temperature. Set B contains Open-Meteo shortwave radiation, air temperature and cloud cover. Both sets also contain sine and cosine of the local hour, so the model can represent daily timing without treating midnight and 23:00 as unrelated endpoints.

We trained on 15 May through 10 June and evaluated on 11 through 17 June, leaving 628 complete training rows and 168 test rows. Feature means and standard deviations came only from training data. We added the intercept after scaling and used the same 97 daytime test observations, defined by positive on-site irradiation, for both feature sets. This avoids making one model appear better through a different evaluation subset.

## Implementing three solvers in NumPy

We implemented the normal equation, batch gradient descent and ordered stochastic gradient descent without a model-fitting library. The objective is half the sum of squared errors. Consequently, the batch update does not divide the gradient by the sample count. We tested the prescribed learning rates using Set A training cost, selected stable runs, and kept the held-out week out of those decisions.

At a batch learning rate of 0.0001, Set A required 57,846 iterations and Set B required 2,742. Both met a maximum coefficient difference of 0.00001 from the normal-equation solution and actually matched its coefficients when rounded to two decimals. SGD used a fixed 1,000 epochs at 0.01. Its coefficients did not exactly match least squares, which is plausible with a constant step size and a fixed row order.

## Results on the later week

| Feature set | Solver | All hours RMSE kW | Daytime RMSE kW |
|---|---|---:|---:|
| A On site | Normal equation | 539.46 | 704.46 |
| A On site | Batch GD | 539.46 | 704.46 |
| A On site | Ordered SGD | 545.69 | 714.02 |
| B Public weather | Normal equation | 2620.94 | 3409.50 |
| B Public weather | Batch GD | 2620.94 | 3409.50 |
| B Public weather | Ordered SGD | 2485.47 | 3148.71 |

All metrics use predictions clipped at zero. The normal-equation daytime error gap is 2,705.04 kW, equivalent to 9.90% of the observed training peak of 27,325.90 kW. That peak is a reference value, not the plant's rated capacity. All-hours RMSE is lower partly because the test set includes easier night hours. These error-to-peak ratios should not be relabelled as percentage accuracy.

![Figure 2 Actual and predicted power during the held out test week](../results/actual_vs_predicted.png)

Irradiation has the largest absolute non-intercept standardized coefficient in Set A, approximately 8,345.04 kW per training standard deviation. Its positive sign agrees with the basic relationship between sunlight and power, but correlated temperatures and time features prevent a simple causal interpretation of every coefficient. Residuals also show hour-specific structure: the mean actual-minus-predicted error is largest at 10:00 and most negative at 18:00. Each mean uses only seven observations, so we did not turn that pattern into a test-tuned correction.

## Location and timing remain important

The public-weather configuration uses approximate coordinates 14.82 N, 78.28 E and Asia/Kolkata time. Sensor irradiation was converted from kW per square metre to W per square metre for comparison. Public-minus-sensor peak offsets were zero hours on 20 May, one hour on 1 June and three hours on 15 June. These discrepancies remain unresolved rather than being hidden behind a better-looking score.

Open-Meteo describes shortwave radiation as a preceding-hour mean, while our sensor series uses left-labelled hourly bins. The raw sensor interval convention and exact plant coordinates are not confirmed. A training-only shift diagnostic found the highest tested daytime correlation without a shift, but that does not prove correct alignment. We did not use test-week peaks to select new coordinates or adjust the model's timestamps.

## From the model to a usable webpage

The dashboard loads the saved Set B normal-equation weights and training scaler. Users can either type or slide all four inputs, apply a preset, and see predicted AC power prominently in kW and MW. It does not retrain on interaction, invent a confidence interval or silently force every night prediction to zero. A positive night output can reveal a limitation of the linear model.

For this six-column problem, the normal equation is a useful reference. Even with ten million rows, its small sufficient-statistic matrices could be accumulated in chunks, so full in-memory storage is not compulsory. Batch GD needs repeated scans, while SGD can stream updates. The better choice depends on feature count, conditioning and convergence requirements, not just the number of rows.

Our next validation step is to confirm location and interval metadata, then evaluate on a fresh period using a declared alignment rule. Public weather may support rough educational scenarios, but a reliable rooftop application would require local capacity, orientation, losses and longer seasonal validation. The experiment's main lesson is that a working interface and a low training cost cannot substitute for a well-defined dataset and honest evaluation.

## Sources

Kaggle Solar Power Generation Data: https://www.kaggle.com/datasets/anikannal/solar-power-generation-data

Open-Meteo Historical Weather API documentation: https://open-meteo.com/en/docs/historical-weather-api

Experiment evidence: results/analysis.md, results/preparation_table1.md, results/location_check.md and results/rmse_table.csv in the accompanying project.
