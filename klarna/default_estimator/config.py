# config.py

from pathlib import Path


data_dir = Path('../data')
data_path = data_dir / 'dataset.csv'

output_data_dir = Path('../data/output')
output_data_path = output_data_dir / 'results.csv'

model_dir = Path('../models')
model_path = model_dir / 'model_rfc.pkl'

means_dir = Path('../obj')
means_path = means_dir / 'mean_values.pkl'

