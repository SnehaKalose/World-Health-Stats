import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv('30-70cancerChdEtc.csv')

# Drop rows with missing values
df.dropna(inplace=True)

# Encode categorical variables
le_location = LabelEncoder()
le_gender = LabelEncoder()

df['Location_enc'] = le_location.fit_transform(df['Location'])
df['Gender_enc'] = le_gender.fit_transform(df['Dim1'])  # 'Dim1' is gender

# Features and target
X = df[['Period', 'Location_enc', 'Gender_enc']]
y = df['First Tooltip']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------ Linear Regression ------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# ------------------ Decision Tree ------------------
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# ------------------ Random Forest ------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ------------------ Evaluation ------------------
def evaluate_model(name, y_true, y_pred):
    print(f"Model: {name}")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("R² Score:", r2_score(y_true, y_pred))
    print("-" * 30)

evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Decision Tree", y_test, y_pred_dt)
evaluate_model("Random Forest", y_test, y_pred_rf)
