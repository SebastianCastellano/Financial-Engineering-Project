def tuning_xgb(x_train, y_train):
    
        # Functions to tune the hyperparameters of the XGB through a random grid 
    # search (tune of learning rate, max depth, min child weight, subsample )
    
    # INPUT: x_train = training set of spreads to be used to tune
    #        y_train = rating associated to each 'x_train'
    
    # OUTPUT: eta = learning rate
    #         depth =  maximum depth of a tree
    #         child =  minimum sum of weights of all observations required in a child
    #         subsample = fraction of observations to be randomly sampled for each tree

    from sklearn.model_selection import RandomizedSearchCV
    import xgboost as xgb
    import numpy as np
    
    import warnings 

    warnings.filterwarnings('ignore') #to suppress warnings related to suggestions for the tuning of XGB 


    param_grid = {'learning_rate': np.logspace(-2,-1,3), 'max_depth': np.linspace(3,10,3).astype(int), 'min_child_weight': np.linspace(0,10,3).astype(int),\
                  'subsample': np.linspace(0.5,1,3), 'objective': ['multi:softmax']}
    grid = RandomizedSearchCV(xgb.XGBClassifier(label_encoder=False), param_distributions = param_grid, n_iter=5) # if greater than 5 runtime is too high
    grid.fit(x_train, y_train)
    best_params = grid.best_params_
    eta = best_params['learning_rate']
    depth = best_params['max_depth']
    child = best_params['min_child_weight']
    #col_sample = best_params['colsample_bytree']
    subsample = best_params['subsample']
        
    return eta, depth, child, subsample

