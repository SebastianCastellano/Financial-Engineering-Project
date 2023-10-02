def tuning_c(x_train, y_train, flag):
    
    # Functions to tune the hyperparameters of the SVM through a random grid 
    # search (tune of C and gamma)
    
    # INPUT: x_train = training set of spreads to be used to tune
    #        y_train = rating associated to each 'x_train'
    #        flag = 0 or 1, selected to tune for a linear kernel or a radial
    
    # OUTPUT: C = penalty parameter of the error term
    #         gamma = control of the precision of  fitting  the  training  set
    
    from sklearn.svm import SVC
    from sklearn.model_selection import RandomizedSearchCV
    import numpy as np

    if flag == 0:
        param_grid = {'C': np.logspace(-1, 2, 100), 'kernel': ['linear']}
        grid = RandomizedSearchCV(SVC(), param_grid, n_iter=10)
        grid.fit(x_train, y_train)
        best_params = grid.best_params_
        c = best_params['C']
        
        return c

    else:
        param_grid = {'C': np.logspace(-1, 2, 100), 'gamma': np.logspace(-4, 1, 100), 'kernel': ['rbf']}
        grid = RandomizedSearchCV(SVC(), param_grid, n_iter=10)
        grid.fit(x_train, y_train)
        best_params = grid.best_params_
        c = best_params['C']
        gamma = best_params['gamma']

        return c, gamma
