from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score
import numpy
import matplotlib.pyplot as plot
from scipy.stats import expon

def train_regressor(X,y):
    '''
    tunedModel = RandomForestRegressor(n_estimators=100,
                                       max_features=30,
                                       random_state=6,
                                       bootstrap=True,
                                       min_samples_split=2)
    '''
    tunedModel = AdaBoostRegressor(n_estimators=500,
                                       random_state=18,
                                       loss='linear',
                                       learning_rate=0.01)
    
    
    kfold = KFold(n_splits=10,shuffle=True,random_state=9001)
    performances = numpy.zeros(10)
    
    for fold, (train_index, test_index) in enumerate(kfold.split(X)):
        tunedModel.fit(X[train_index],y[train_index])
        performances[fold] = validate(tunedModel,X[test_index],y[test_index])
    
    tunedModel.fit(X,y)
    
    return tunedModel, performances.mean(), performances.var()


def validate(forest, X, y):
    prediction = forest.predict(X)
    return r2_score(y,prediction)
        
def feature_importance(forest, feature_names):
    importances = forest.feature_importances_
    indices = numpy.argsort(importances)[::-1]

    print("Feature ranking:")
    
    show_top_n_features = 178
    for f in range(show_top_n_features):
        print('{}. {} ({})'.format(f + 1, feature_names[indices[f]], importances[indices[f]]))
    
    plot.figure()
    plot.title("Feature Importance (AdaBoost)")
    plot.bar(range(len(importances)), importances[indices],
            color="r", align="center")
    plot.xlabel('Feature')
    plot.xlim([-1, show_top_n_features])
    plot.ylabel('Importance')
    plot.show()
        
def feature_importance_details(forest, feature_names):
    importances = forest.feature_importances_
    indices = numpy.argsort(importances)[::-1]

    print("Feature ranking:")
    
    show_top_n_features = 10
    for f in range(show_top_n_features):
        print('{}. {} ({})'.format(f + 1, feature_names[indices[f]], importances[indices[f]]))
    
    plot.figure()
    plot.title("Feature Importance (AdaBoost)")
    plot.bar(range(len(importances)), importances[indices],
            color="r", align="center")
    plot.xticks(range(show_top_n_features), feature_names[indices],rotation=60,ha='right')
    plot.xlim([-1, show_top_n_features])
    plot.ylabel('Importance')
    plot.show()
    
    return indices[:show_top_n_features]