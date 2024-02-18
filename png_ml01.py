import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json
import joblib

# Load and preprocess data
penguin_df = pd.read_csv('penguins.csv')
penguin_df.dropna(inplace=True)

# Organize data for model input
output = penguin_df['species']
features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
features = pd.get_dummies(features)

# Note the order of the columns after get_dummies
feature_order = features.columns.tolist()

output, uniques = pd.factorize(output)
x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=0.8) 

rfc = RandomForestClassifier(random_state=15)
rfc.fit(x_train.values, y_train)
y_pred = rfc.predict(x_test.values)

score = accuracy_score(y_pred, y_test)
print('Our accuracy score for this model is {}'.format(score))

# Save the model using joblib
joblib.dump(rfc, 'random_forest_penguin.joblib')

# Save the uniques variable and feature order using JSON
with open('output_penguin.json', 'w') as f:
    json.dump(uniques.tolist(), f)

with open('feature_order.json', 'w') as f:
    json.dump(feature_order, f)
