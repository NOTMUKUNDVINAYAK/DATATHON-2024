import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import make_scorer
from keras.losses import MeanAbsolutePercentageError

train_df = pd.read_csv('Train.csv')
test_df = pd.read_csv('Test.csv')

X_train = train_df.drop(columns=['ENTITY_LENGTH'])
Y_train = train_df['ENTITY_LENGTH']

X_test = test_df.copy()

label_encoder = LabelEncoder()
X_train['CATEGORY_ID'] = label_encoder.fit_transform(X_train['CATEGORY_ID'])

X_test['CATEGORY_ID'] = X_test['CATEGORY_ID'].apply(lambda x: x if x in label_encoder.classes_ else -1)
label_encoder.classes_ = np.append(label_encoder.classes_, -1)
X_test['CATEGORY_ID'] = label_encoder.transform(X_test['CATEGORY_ID'])

tfidf = TfidfVectorizer(max_features=100)
X_train_tfidf = tfidf.fit_transform(X_train['ENTITY_DESCRIPTION'].fillna(''))
X_test_tfidf = tfidf.transform(X_test['ENTITY_DESCRIPTION'].fillna(''))

svd = TruncatedSVD(n_components=50, random_state=42)
X_train_svd = svd.fit_transform(X_train_tfidf)
X_test_svd = svd.transform(X_test_tfidf)

X_train_combined = np.hstack((X_train[['CATEGORY_ID']].values, X_train_svd))
X_test_combined = np.hstack((X_test[['CATEGORY_ID']].values, X_test_svd))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_combined)
X_test_scaled = scaler.transform(X_test_combined)

kf = KFold(n_splits=5, random_state=42, shuffle=True)

xgb = XGBRegressor(random_state=42)

param_dist = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3]
}
mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

random_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_dist, 
                                   n_iter=20, cv=kf, scoring=mape_scorer, 
                                   n_jobs=-1, random_state=42)

random_search.fit(X_train_scaled, Y_train)

best_xgb = random_search.best_estimator_

Y_test_pred = best_xgb.predict(X_test_scaled)

results_df = pd.DataFrame({
    'ENTITY_ID': test_df['ENTITY_ID'],
    'ENTITY_LENGTH': Y_test_pred
})
results_df.to_csv('Predicted_Entity_Lengths_XGB.csv', index=False)

model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Output layer

model.compile(optimizer='adam', loss=MeanAbsolutePercentageError())

history = model.fit(X_train_scaled, Y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

Y_test_nn_pred = model.predict(X_test_scaled)

results_nn_df = pd.DataFrame({
    'ENTITY_ID': test_df['ENTITY_ID'],
    'ENTITY_LENGTH': Y_test_nn_pred.flatten()
})
results_nn_df.to_csv('Predicted_Entity_Lengths_NN.csv', index=False)

final_pred = (Y_test_pred + Y_test_nn_pred.flatten()) / 2

final_results_df = pd.DataFrame({
    'ENTITY_ID': test_df['ENTITY_ID'],
    'ENTITY_LENGTH': final_pred
})
final_results_df.to_csv('Predicted_Entity_Lengths_Ensemble.csv', index=False)

print(f"Best parameters: {random_search.best_params_}")
print('Predictions saved to files.')
