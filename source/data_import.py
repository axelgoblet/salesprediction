import json
import pandas

def load_surroundings():
    with open('../data/Surroundings.json') as json_file:
        surroundings = json.load(json_file)
    return surroundings

def load_sales():
    sales = pandas.read_csv('../data/sales_granular.csv')
    sales.set_index('store_code',inplace=True)
    return sales