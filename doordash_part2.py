import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from datetime import date
from datetime import datetime
from sklearn.externals import joblib
import numpy as np

def predict_delivery_time(file_path):
	data = load_data(file_path)
	processed_data = process_data(data)
	features = generate_features(processed_data)
	predict(features)
	
def load_data(pfile_path):
	#load data
	print("loading data...")
	if len(pfile_path) <= 0:
		pfile_path = "data_to_predict.json"
	prediction_df = pd.read_json(pfile_path, lines=True)
	return prediction_df
	
def process_data(prediction_df):
	print("processing data...")
	prediction_df['store_primary_category'] = prediction_df['store_primary_category'].astype('category')
	prediction_df['store_primary_category'] = prediction_df['store_primary_category'].cat.codes

	#treat missing values
	prediction_df[['total_onshift_dashers', 'total_busy_dashers', 'total_outstanding_orders']] = \
	prediction_df[['total_onshift_dashers', 'total_busy_dashers','total_outstanding_orders']].fillna(value=0)

	prediction_df['store_primary_category'] = prediction_df['store_primary_category'].fillna(value='other')
	prediction_df['order_protocol'] = prediction_df['order_protocol'].fillna(prediction_df['order_protocol'].value_counts().index[0])

	#drop rest of the missing values about 1k
	prediction_df = prediction_df.dropna() 
	prediction_df = prediction_df.replace('NA',0)
	return prediction_df

def generate_features(prediction_df):
	#create features
	print("generating features...")
	prediction_df['created_at'] = prediction_df['created_at'].apply(lambda x : str(x))
	order_time = prediction_df['created_at'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
	prediction_df['hour_of_the_day'] = order_time.apply(lambda x: x.hour)
	
	prediction_df = prediction_df.drop(['platform','created_at'],axis=1)

	prediction_df['estimated_store_to_consumer_driving_duration'] = \
	pd.to_numeric(prediction_df['estimated_store_to_consumer_driving_duration'])

	prediction_df['estimated_order_driving_duration'] = \
	prediction_df['estimated_order_place_duration']+prediction_df['estimated_store_to_consumer_driving_duration']

	prediction_df['total_onshift_dashers'] = pd.to_numeric(prediction_df['total_onshift_dashers'])
	prediction_df['total_busy_dashers'] = pd.to_numeric(prediction_df['total_busy_dashers']) 
	prediction_df['total_onshift_dashers'] = prediction_df['total_onshift_dashers'].where(prediction_df['total_onshift_dashers'] > 0, 0)
	prediction_df['total_busy_dashers'] = prediction_df['total_busy_dashers'].where(prediction_df['total_busy_dashers'] > 0, 0)

	prediction_df['total_available_dashers'] = prediction_df['total_onshift_dashers'] - prediction_df['total_busy_dashers']
	prediction_df['total_available_dashers'] = prediction_df['total_available_dashers'].where(prediction_df['total_available_dashers'] > 0, 0)

	prediction_df['total_outstanding_orders'] = pd.to_numeric(prediction_df['total_outstanding_orders'])
	prediction_df['total_outstanding_orders'] = prediction_df['total_outstanding_orders'].where(prediction_df['total_outstanding_orders'] > 0, 0)

	prediction_df['total_orders_without_dashers']  = prediction_df['total_outstanding_orders'] - prediction_df['total_onshift_dashers']
	
	return prediction_df
	
def predict(prediction_df):
	#load pretrained models
	print("loading models...")
	gbr = joblib.load('GradientBoostedRegressor.joblib')
	rfr = joblib.load('RandomForestRegressor.joblib')
	
	delivery_id = prediction_df['delivery_id']
	prediction_df = prediction_df.drop(['delivery_id'],axis=1)
	
	#perform predictions
	print("performing predictions...")
	py_pred_gbr = gbr.predict(prediction_df)
	py_pred_rfr = rfr.predict(prediction_df)
	py_pred = np.mean([py_pred_gbr, py_pred_rfr], axis=0)

	pred_df = pd.DataFrame(data=py_pred, index=delivery_id.index, columns=['predicted_delivery_seconds'])

	pred_df = pd.concat([delivery_id,pred_df], axis=1) 
	#write to csv
	print("writng to prediction.csv")
	pred_df.to_csv("prediction.csv", sep='\t')
