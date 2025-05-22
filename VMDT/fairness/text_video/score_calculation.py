from .analyze import *

def score_calculation(scenarios=['stereotype','decision_making','factual_accuracy']):
	if 'stereotype' in scenarios:
		stereotype()
	if 'decision_making' in scenarios:
		decision()
	if 'factual_accuracy' in scenarios:
		factual()