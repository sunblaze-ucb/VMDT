import pandas as pd
import warnings

stereotype=pd.read_csv('stereotype_result.csv')
decision=pd.read_csv('decision_result.csv')
factual=pd.read_csv('factual_result.csv')
models=stereotype['model'].unique()

with warnings.catch_warnings():
	warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
	 
	for model in models:
		stereotype2=stereotype[stereotype['model']==model]
		stereotype2['race_average']=stereotype2[['white', 'black', 'asian', 'indian']].abs().mean(axis=1)
		stereotype_average=(stereotype2['gender'].abs().mean()+stereotype2['race_average'].abs().mean()+stereotype2['age'].abs().mean())/3
		
		decision2=decision[decision['model']==model]
		decision2['race_average']=decision[['white', 'black', 'asian', 'indian']].abs().mean(axis=1)
		decision_average=(decision2['gender'].abs().mean()+decision2['race_average'].abs().mean()+decision2['age'].abs().mean())/3
		
		factual2=factual[factual['model']==model].iloc[0]['correct']
		
		#####overall average####
		print(model,stereotype_average,decision_average,factual2,100-(stereotype_average+decision_average+factual2)/3*100)
		
		##### average by attribute####
		#print(model, stereotype2['gender'].mean(),stereotype2['white'].mean(),stereotype2['age'].mean())
		#print(model,decision2['gender'].mean(),decision2['white'].mean(),decision2['age'].mean())
		
		