import data_import
import data_exploration
import data_preprocessing
import regression_modeling

first_n_timepoints = None

surroundings = data_import.load_surroundings()
sales = data_import.load_sales()
#import json
#print(json.dumps(surroundings[0]))

data_exploration.plot_histogram_of_opening_periods(sales,first_n_timepoints)
data_exploration.plot_boxplot_of_opening_periods(sales)

feature_matrix = data_preprocessing.create_feature_matrix(surroundings,sales)
target_variable = data_preprocessing.create_target_variable(sales,first_n_timepoints)

feature_matrix, target_variable = data_preprocessing.join_dataframes(feature_matrix, 
                                                                      target_variable)

columns = feature_matrix.columns
feature_matrix = data_preprocessing.impute_missing_values(feature_matrix.values)

model, r2_mean, r2_var = regression_modeling.train_regressor(feature_matrix,
                                                                  target_variable.values)
print('r2: {} ({})'.format(r2_mean,r2_var))
regression_modeling.feature_importance(model, columns)
important_features = regression_modeling.feature_importance_details(model, columns)

for feature in important_features:
    data_exploration.correlation_plot(target_variable, feature_matrix, feature, columns[feature])