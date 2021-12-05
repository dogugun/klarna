# Default Probability Estimation
In this work, a simple model is developed and run in batch mode to estimate the probability 
of the default value(1) for a given customer. While doing this estimation, his/her previous financial 
status and behaviours are used as input features. 

The solution is formed of 3 parts

- Notebook: Where exploratory data analysis, model building and training results are observed.
Model and impuation values for missing data exported in this part
- Batch Scoring: This script is developed to run scoring on previously saved model with the test data.
- Flask Service: This is a very simple web service which returns the estimated probability with the queried uuid.

The folder structure is as follows:
```
klarna/
├── data/
│   ├── dataset.csv
│   └── output/
│       └── results.csv
├── default_estimator/
│   └── config.py
├── file_summary.py
├── models/
│   └── model_rfc.pkl
├── notebooks/
│   ├── eda_n_model_building.html
│   └── eda_n_model_building.ipynb
├── obj/
│   └── mean_values.pkl
├── README.md
├── scripts/
│   └── batch_scoring.py
└── service/
    ├── __pycache__/
    │   └── estimations.cpython-37.pyc
    ├── app.py
    └── estimations.py
```
According to this structure:  
Jupyter Notebooks are in notebooks folder
Batch estimator is in scripts folder
Flask service is in service folder.
