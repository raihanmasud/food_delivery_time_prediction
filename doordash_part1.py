print("loading libraries - pandas, sklearn, etc ...")
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from datetime import date
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.model_selection import KFold

#load data
print("loading data...")
data_df = pd.read_csv("historical_data.csv")

#impute missing values
print("processing data...")
data_df[['total_onshift_dashers', 'total_busy_dashers', 'total_outstanding_orders']] = \
data_df[['total_onshift_dashers', 'total_busy_dashers','total_outstanding_orders']].fillna(value=0)

data_df['store_primary_category'] = data_df['store_primary_category'].fillna(value='other')

data_df['order_protocol'] = data_df['order_protocol'].fillna(data_df['order_protocol'].value_counts().index[0])

data_df = data_df.dropna() 

#process categorial featue
data_df['store_primary_category'] = data_df['store_primary_category'].astype('category')
data_df['store_primary_category'] = data_df['store_primary_category'].cat.codes

#calculate delivery duration from timestamp
print("generating features...")
start = data_df['created_at'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
end = data_df['actual_delivery_time'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
data_df['total_delivery_duration'] = (end - start).apply(lambda x : x.seconds)

#feature engineering and deal with noisy/-ve data
data_df['estimated_order_driving_duration'] = \
data_df['estimated_order_place_duration']+data_df['estimated_store_to_consumer_driving_duration']

data_df['total_onshift_dashers'] = data_df['total_onshift_dashers'].where(data_df['total_onshift_dashers'] > 0, 0)
data_df['total_busy_dashers'] = data_df['total_busy_dashers'].where(data_df['total_busy_dashers'] > 0, 0)

data_df['total_available_dashers'] = data_df['total_onshift_dashers'] - data_df['total_busy_dashers']
data_df['total_available_dashers'] = data_df['total_available_dashers'].where(data_df['total_available_dashers'] > 0, 0)

data_df['total_outstanding_orders'] = data_df['total_outstanding_orders'].where(data_df['total_outstanding_orders'] > 0, 0)

data_df['hour_of_the_day'] = start.apply(lambda x: x.hour)

data_df['total_orders_without_dashers']  = data_df['total_outstanding_orders'] - data_df['total_onshift_dashers']

#dropping timestamp and other features with low feature importance 
y = data_df['total_delivery_duration']
X = data_df.drop(['total_delivery_duration','created_at','estimated_order_place_duration', 
                  'actual_delivery_time', 'total_onshift_dashers','order_protocol',
                  'total_busy_dashers','min_item_price'], axis=1)

#split trainging and test data
print("splitting train-test sets...") 				  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#models
gbr = GradientBoostingRegressor(n_estimators=600, learning_rate=0.1, max_features=0.4, max_depth=7,
                                           min_samples_leaf=3, loss='huber', alpha=0.55)
rfr = RandomForestRegressor(n_estimators=600, max_depth=7, max_features='auto', random_state=1,  n_jobs = -1)

models = {gbr : "GradientBoostedRegressor" , rfr : "RandomForestRegressor" }

#cross validation
no_of_cv_folds = 5
kf = KFold(n_splits=no_of_cv_folds,random_state=42, shuffle=True)
rmses = []
for model in models:
	print("performing cross validation with ", models[model])
	
	for train_index, test_index in kf.split(X_train):
		X_cv_train, X_cv_test = X_train.iloc[train_index], X_train.iloc[test_index]
		y_cv_train, y_cv_test = y_train.iloc[train_index], y_train.iloc[test_index]
		model = model.fit(X_cv_train,y_cv_train)
		y_cv_pred = model.predict(X_cv_test)
		rmse = sqrt(mean_squared_error(y_cv_test, y_cv_pred))
		print("RMSE :",rmse)
		rmses.append(rmse)
		
	print('mean cv rmse :', (sum(rmses)/no_of_cv_folds))

#Final Training and prediction

for model in models:
	print("fitting model...",models[model])
	model = model.fit(X_train,y_train)
	print("predicting ...")
	y_pred = model.predict(X_test)
	print("RMSE :",sqrt(mean_squared_error(y_test, y_pred)))

	for feature, importance in zip(list(X_test.columns),list(model.feature_importances_)):
		print(feature, importance)

#pickle model for future prediction
for model in models:
    print("pickling models with total historic training data")
    model = model.fit(X,y)
    pikle_file = models[model]+'.joblib'
    joblib.dump(model, pikle_file)