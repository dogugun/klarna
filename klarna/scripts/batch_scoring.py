import sys

from klarna.default_estimator import config

sys.path.insert(1, '../industrial_tsp/industrial_tsp')
import pandas as pd
import pickle


# def predict():
#     json_file = open(config.model_path, 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     loaded_model = model_from_json(loaded_model_json)
#     loaded_model.load_weights(config.model_weights_path)
#
#     processed_data = pd.read_csv(config.processed_data_path)
#     processed_data = processed_data[['target','input_1','input_2','input_3','input_4','input_5','input_6','target_t-1','input_1_t-1','input_2_t-1','input_3_t-1','input_4_t-1','input_5_t-1','input_6_t-1','target_t-2','input_1_t-2','input_2_t-2','input_3_t-2','input_4_t-2','input_5_t-2','input_6_t-2','target_t-3','input_1_t-3','input_2_t-3','input_3_t-3','input_4_t-3','input_5_t-3','input_6_t-3','target_t-4','input_1_t-4','input_2_t-4','input_3_t-4','input_4_t-4','input_5_t-4','input_6_t-4','target_t-5','input_1_t-5','input_2_t-5','input_3_t-5','input_4_t-5','input_5_t-5','input_6_t-5','target_t-6','input_1_t-6','input_2_t-6','input_3_t-6','input_4_t-6','input_5_t-6','input_6_t-6']]
#
#     scaler = StandardScaler()
#     processed_data=scaler.fit_transform(processed_data)
#
#     number_of_timesteps = config.vector_length + 1 # from t to t-6
#     number_of_inputs = config.input_sensors+1 #+1 for the target itself at time t
#     processed_data = processed_data.reshape(processed_data.shape[0], number_of_inputs, number_of_timesteps)
#
#     result = loaded_model.predict(processed_data)
#
#     pd.DataFrame(result).to_csv(config.results_data_path, index=False, header=False)

#---------------------------------------------
def batch_score():
    df = pd.read_csv(config.data_path, sep=';')

    merchantgroups = pd.get_dummies(df['merchant_group'])
    df = df.drop(['merchant_group'], axis=1, errors='ignore')
    df = pd.concat([df, merchantgroups], axis=1)

    test = df[df['default'].isna()]
    train = df[df['default'].isna() == False]

    not_nulls_dict = (train.count() / train.shape[0]).to_dict()

    sparse_cols = []
    for k in not_nulls_dict.keys():
        if not_nulls_dict[k] <= 0.5:
            sparse_cols.append(k)

    train = train.drop(sparse_cols, axis=1)
    test = test.drop(sparse_cols, axis=1)

    train_miss_cols = []
    for col in train.columns:
        if train[col].isna().any():
            train_miss_cols.append(col)

    a_file = open(config.means_path, "rb")
    missing_values_dict = pickle.load(a_file)


    for col in train_miss_cols:
        train[col].fillna(missing_values_dict[col], inplace=True)
        test[col].fillna(missing_values_dict[col], inplace=True)

    train.drop(['merchant_category'], axis=1, inplace=True)
    test.drop(['merchant_category'], axis=1, inplace=True)

    train.drop(['name_in_email'], axis=1, inplace=True)
    test.drop(['name_in_email'], axis=1, inplace=True)

    categorical_variables = ['num_arch_written_off_0_12m', 'num_arch_written_off_12_24m', 'status_last_archived_0_24m',
                             'status_2nd_last_archived_0_24m', 'status_3rd_last_archived_0_24m',
                             'status_max_archived_0_6_months', 'status_max_archived_0_12_months',
                             'status_max_archived_0_24_months', 'merchant_group'] + list(merchantgroups.columns)
    numerical_variables = list(set(list(train.columns)) - set(['uuid', 'default']) - set(categorical_variables))

    train_model = train[['default', 'account_amount_added_12_24m', 'account_days_in_dc_12_24m',
                         'account_days_in_rem_12_24m', 'account_days_in_term_12_24m', 'age', 'avg_payment_span_0_3m',
                         'avg_payment_span_0_12m', 'has_paid', 'max_paid_inv_0_24m', 'num_active_div_by_paid_inv_0_12m',
                         'num_active_inv', 'num_arch_dc_0_12m', 'num_arch_dc_12_24m', 'num_arch_ok_0_12m',
                         'num_arch_ok_12_24m', 'num_arch_rem_0_12m', 'num_arch_written_off_0_12m',
                         'num_arch_written_off_12_24m', 'num_unpaid_bills', 'status_last_archived_0_24m',
                         'status_2nd_last_archived_0_24m', 'status_3rd_last_archived_0_24m',
                         'status_max_archived_0_24_months', 'recovery_debt', 'sum_capital_paid_account_0_12m',
                         'sum_capital_paid_account_12_24m', 'sum_paid_inv_0_12m', 'time_hours']
                        + list(merchantgroups.columns)].copy()
    test_model = test[['account_amount_added_12_24m', 'account_days_in_dc_12_24m', 'account_days_in_rem_12_24m',
                       'account_days_in_term_12_24m', 'age', 'avg_payment_span_0_3m','avg_payment_span_0_12m', 'has_paid',
                       'max_paid_inv_0_24m', 'num_active_div_by_paid_inv_0_12m', 'num_active_inv', 'num_arch_dc_0_12m',
                       'num_arch_dc_12_24m', 'num_arch_ok_0_12m', 'num_arch_ok_12_24m', 'num_arch_rem_0_12m',
                       'num_arch_written_off_0_12m', 'num_arch_written_off_12_24m', 'num_unpaid_bills',
                       'status_last_archived_0_24m', 'status_2nd_last_archived_0_24m', 'status_3rd_last_archived_0_24m',
                       'status_max_archived_0_24_months', 'recovery_debt', 'sum_capital_paid_account_0_12m',
                       'sum_capital_paid_account_12_24m', 'sum_paid_inv_0_12m', 'time_hours'] \
                      + list(merchantgroups.columns)].copy()

    train_model['num_arch_dc'] = train_model['num_arch_dc_0_12m'] + train_model['num_arch_dc_12_24m']
    train_model['num_arch_ok'] = train_model['num_arch_ok_0_12m'] + train_model['num_arch_ok_12_24m']
    train_model.drop(['num_arch_dc_0_12m', 'num_arch_dc_12_24m', 'num_arch_ok_0_12m', 'num_arch_ok_12_24m'], axis=1,
                     inplace=True)

    test_model['num_arch_dc'] = test_model['num_arch_dc_0_12m'] + test_model['num_arch_dc_12_24m']
    test_model['num_arch_ok'] = test_model['num_arch_ok_0_12m'] + test_model['num_arch_ok_12_24m']
    test_model.drop(['num_arch_dc_0_12m', 'num_arch_dc_12_24m', 'num_arch_ok_0_12m', 'num_arch_ok_12_24m'], axis=1,
                    inplace=True)

    train_model.drop(['sum_paid_inv_0_12m', 'account_amount_added_12_24m', 'status_2nd_last_archived_0_24m',
                      'status_3rd_last_archived_0_24m'], axis=1, inplace=True)
    test_model.drop(['sum_paid_inv_0_12m', 'account_amount_added_12_24m', 'status_2nd_last_archived_0_24m',
                     'status_3rd_last_archived_0_24m'], axis=1, inplace=True)

    train_model.drop(
        ['Automotive Products', 'Jewelry & Accessories', 'recovery_debt', 'Erotic Materials', 'Food & Beverage',
         'account_days_in_dc_12_24m', 'Children Products', 'Home & Garden', 'Electronics'], axis=1, inplace=True)
    test_model.drop(['Automotive Products', 'Jewelry & Accessories', 'recovery_debt', 'Erotic Materials', 'Food & Beverage',
                     'account_days_in_dc_12_24m', 'Children Products', 'Home & Garden', 'Electronics'], axis=1,
                    inplace=True)

    clf = pickle.load(open(config.model_path, 'rb'))

    y_res = clf.predict(test_model)
    y_res_prob = clf.predict_proba(test_model)


    test_model['estimated_result'] = y_res
    test_model['pd'] = y_res_prob[:,1]
    test_model['uuid'] = df[df['default'].isna()].uuid
    test_model[['uuid', 'pd']].to_csv(config.output_data_path, index=False)

    print("scoring default value: 1 rate: ", y_res.sum() / y_res.shape[0])


batch_score()