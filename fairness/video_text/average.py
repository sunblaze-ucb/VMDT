import pandas as pd
import math, json
import warnings
import numpy as np
	
stereotype=pd.read_csv('stereotype_average.csv')
decision=pd.read_csv('decision_making_average.csv')
factual=pd.read_csv('factual_accuracy.csv')
models=stereotype['model'].unique()

with warnings.catch_warnings():
	warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
	
	
	#################average##########3
	for model in models:
		stereotype2=stereotype[stereotype["model"]==model]
		stereotype2=stereotype2.iloc[0]
		gender=abs(stereotype2['gender'])
		age=json.loads(stereotype2['age'])
		eth=json.loads(stereotype2['eth'])
		age= np.mean([abs(x) for x in age])
		eth= np.mean([abs(x) for x in eth])
		stereotype_average=100-(gender+age+eth)*100/3

		decision2=decision[decision["model"]==model]
		decision2=decision2.iloc[0]
		gender=abs(decision2['gender'])
		age=json.loads(decision2['age'])
		eth=json.loads(decision2['eth'])
		age= np.mean([abs(x) for x in age])
		eth= np.mean([abs(x) for x in eth])
		decision_average=100-(gender+age+eth)*100/3
		
		factual2=factual[factual["model"]==model]
		factual2=100-abs(factual2.iloc[0]['score'])*100
		print(model, (stereotype_average+decision_average+factual2)/3)
		
	
		