import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

def add_features(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	df["Date"] = pd.to_datetime(df["Date"])
	df["Year"] = df["Date"].dt.year
	df["Month"] = df["Date"].dt.month
	df["Day"] = df["Date"].dt.day
	df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

	comp_open = pd.to_datetime(
		dict(year=df["CompetitionOpenSinceYear"],
			 month=df["CompetitionOpenSinceMonth"], day=1),
		errors="coerce",
	)
	months = (df["Date"].dt.year - comp_open.dt.year) * 12 \
			 + (df["Date"].dt.month - comp_open.dt.month)
	df["CompetitionMonthsSince"] = months.clip(lower=0)   # NaN stays NaN → imputed + flagged
	return df



def build_preprocessor() -> ColumnTransformer:
	"""
	Builds a ColumnTransformer for preprocessing the data. It handles numeric imputation, 
	one-hot encoding for low-cardinality categorical features, target encoding for high-cardinality categorical features, 
	and passes through other specified features.
	
	Args:
		None

	Returns:
		ColumnTransformer: A ColumnTransformer object that applies the specified transformations to the input data.
	"""
	# Define the feature groups for preprocessing
	numeric_impute = ["CompetitionDistance", "CompetitionMonthsSince"]   # fixed
	low_card_cat   = ["StoreType", "Assortment", "StateHoliday"]
	high_card_cat  = ["Store"]                                            # the fix
	passthrough    = ["DayOfWeek", "Year", "Month", "Day", "WeekOfYear",
					"Promo", "SchoolHoliday", "Promo2"]

	ct = ColumnTransformer(
		transformers=[
			("impute", SimpleImputer(strategy="median", add_indicator=True), numeric_impute), # median imputation with indicator for missing values
			("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False), low_card_cat), # one-hot encoding for low-cardinality categorical features
			("store", TargetEncoder(target_type="continuous", random_state=42), high_card_cat), # target encoding for high-cardinality categorical features
			("pass",   "passthrough", passthrough), # pass through other specified features without transformation
		],
		remainder="drop", # drop any remaining features not specified in the transformers
	)
	return ct