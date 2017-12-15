import pandas
import numpy
import copy
from sklearn.preprocessing import Imputer

def create_feature_matrix(surroundings,sales):
    surrounding_types = surroundings[0]['surroundings'].keys()
    number_of_surrounding_types= len(surrounding_types)
    number_of_features = 2*len(surrounding_types) + 2
    number_of_examples = len(surroundings)
    feature_matrix = numpy.zeros((number_of_examples, number_of_features))
    store_codes = numpy.zeros(number_of_examples)
    
    for example_index, surrounding in enumerate(surroundings):
        store_codes[example_index] = surrounding['store_code']
        ratingcount=0
        for type_index, surrounding_type in enumerate(surrounding_types):
            surrounding_type_list = surrounding['surroundings'][surrounding_type]
            number_of_surroundings = len(surrounding_type_list)
            feature_matrix[example_index, type_index] = number_of_surroundings
            feature_matrix[example_index, number_of_surrounding_types] += number_of_surroundings
    
            ratings = [d['rating'] for d in surrounding_type_list if 'rating' in d]
            if len(ratings) > 0:
                feature_matrix[example_index, type_index + number_of_surrounding_types + 1] = numpy.mean(ratings)
                feature_matrix[example_index, 2*number_of_surrounding_types+1] += sum(ratings)
                ratingcount+=len(ratings)
            else:
                feature_matrix[example_index, type_index + number_of_surrounding_types + 1] = numpy.nan
        feature_matrix[example_index, 2*number_of_surrounding_types+1] = feature_matrix[example_index, 2*number_of_surrounding_types+1] / ratingcount if ratingcount > 0 else numpy.nan
        

    columns = ['number_of_' + s + 's' for s in surrounding_types] + ['number_of_surroundings'] + ['average_' + s + '_rating' for s in surrounding_types] + ['average_surrounding_rating']
    feature_dataframe = pandas.DataFrame(feature_matrix, 
                                         columns=columns,
                                         index=store_codes)
    
    return feature_dataframe

def first_sales(sales):
    return numpy.isnan(sales.values).argmin(axis=1)

def last_sales(sales):
    number_of_columns = sales.shape[1]
    return number_of_columns - 1 - numpy.fliplr(numpy.isnan(sales.values)).argmin(axis=1)

def fill_nonsales(sales):
    store_open = first_sales(sales)
    store_closed = last_sales(sales)
    for row in range(len(sales)):
        first = store_open[row]
        last = store_closed[row]
        sales.values[row,first:last] = numpy.nan_to_num(sales.values[row,first:last])
        
def change_axis_to_time_since_open(sales):
    store_open = first_sales(sales)
    for row in range(len(sales)):
        first = store_open[row]
        sales.values[row] = numpy.append(sales.values[row,first:len(sales.columns)],
                                         sales.values[row,0:first])
    
def create_target_variable(sales,first_n_timepoints=None):
    copied_sales = copy.deepcopy(sales)
    
    first_n_timepoints = len(copied_sales.columns) if first_n_timepoints is None else first_n_timepoints
    
    fill_nonsales(copied_sales)
    opening_date = first_sales(copied_sales)
    change_axis_to_time_since_open(copied_sales)
    copied_sales = copied_sales[copied_sales.columns[:first_n_timepoints]]
    mean_sales = copied_sales.mean(1).values
        
    target_variable = pandas.DataFrame({'target': mean_sales,
                                        'first_sale': opening_date},
                                       index=sales.index)
    
    return target_variable  

def join_dataframes(features, target):
    joined_dataframe = features.join(target,how='inner')
    joined_dataframe.dropna(inplace=True,axis=1,how='all')
    
    features = joined_dataframe.loc[:, joined_dataframe.columns != 'target']
    target = joined_dataframe['target']
    return features, target

def impute_missing_values(features):
    imputer = Imputer()
    imputed_features = imputer.fit_transform(features)
    return imputed_features