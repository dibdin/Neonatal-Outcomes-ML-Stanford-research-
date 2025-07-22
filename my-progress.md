- STABL not used for clinical because too little features
- feature standardization applied before STABL
- using elasticnet for regression after STABL. classifying preterms by hard coding 37 weeks
- Columns with all missing values: Dropped automatically.
Columns with some missing values: Kept, and missing values are imputed with the mean.
Centers the data by subtracting the mean of each feature
Scales the data by dividing by the standard deviation of each feature
Formula: z = (x - μ) / σ
- did a true per-sample frequency for each biomarker in preterm and term groups
- used SHAP for preterm/term biomarker frequency plots: X-axis: Mean absolute SHAP value for each feature in term samples.
Y-axis: Mean absolute SHAP value for each feature in preterm samples.
- lambda_grid="auto"?????? scatter plot for stabl looks weird