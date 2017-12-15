import numpy
import matplotlib.pyplot as plot
import copy

import data_preprocessing

def plot_mean_cumulative_sales_over_time(sales):
    copied_sales= copy.deepcopy(sales)
    data_preprocessing.fill_nonsales(copied_sales)
    cumsales = numpy.cumsum(copied_sales, axis=1)
    mean_cumsales = numpy.nanmean(cumsales, axis=0)
    
    last_index = len(mean_cumsales) -1
    
    plot.figure()
    plot.title("Average Cumulative Sales per Location over Time")
    plot.plot(mean_cumsales)
    plot.plot([0,last_index],[min(mean_cumsales),max(mean_cumsales)])
    
def plot_mean_cumulative_sales_since_first_sale(sales,first_n_timepoints=None):
    first_n_timepoints = len(sales.columns) if first_n_timepoints is None else first_n_timepoints
    
    copied_sales= copy.deepcopy(sales)
    data_preprocessing.fill_nonsales(copied_sales)
    data_preprocessing.change_axis_to_time_since_open(copied_sales)
    cumsales = numpy.cumsum(copied_sales, axis=1)
    mean_cumsales = numpy.nanmean(cumsales, axis=0)[:first_n_timepoints]
    
    last_index = len(mean_cumsales) -1
    
    plot.figure()
    plot.title("Average Cumulative Sales per Location since First Sale")
    plot.plot(mean_cumsales)
    plot.plot([0,last_index],[min(mean_cumsales),max(mean_cumsales)])
    
def plot_histogram_of_opening_periods(sales,first_n_timepoints=None):
    first_n_timepoints = len(sales.columns) if first_n_timepoints is None else first_n_timepoints
    
    openings = data_preprocessing.first_sales(sales)
    closings = data_preprocessing.last_sales(sales)
    opening_periods = closings - openings + 1
    
    print(sum(opening_periods < first_n_timepoints),
          sum(opening_periods >= first_n_timepoints))
    
    plot.figure()
    plot.title("Lengths of Store Opening Periods")
    plot.hist(opening_periods,align="left",bins=10)
    plot.xlabel('Duration (Time Points)')
    plot.ylabel('Number of POSs')
    
def plot_boxplot_of_opening_periods(sales):
    openings = data_preprocessing.first_sales(sales)
    closings = data_preprocessing.last_sales(sales)
    
    plot.figure()
    plot.title("Distribution of First and Last Sales")
    plot.boxplot([openings, closings],0,'')
    plot.xticks(range(1,3),['First Sale','Last Sale'])
    plot.ylabel('Time Point')
    
def correlation_plot(target_variable, feature_matrix,feature,feature_label):
    feature_values = feature_matrix[:,feature]
    target_values = target_variable.values
    
    plot.figure()
    plot.title("")
    plot.scatter(feature_values,target_values)
    plot.xlabel(feature_label)
    plot.ylabel('target')