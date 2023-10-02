import numpy as np
import pandas as pd
import datetime
from xgboost import XGBClassifier

from FunCleanData import clean_data
from FunReshapeSpreads import reshape_spreads
from FunSplitTrainVal import split_train_val
from FunTuningXGB import tuning_xgb
from FunPlotDecisionSurface import plot_decision_surface
from FunImpliedRating import implied_ratings


# DISCLAIMER: the run of the main script can be avoided since all the meaningful 
# results are stored into  .csv files, so one can run Run_RM3_Performance.py only


# import files
spreads = pd.read_excel("bnd_cds_5y.xlsx")
rating = pd.read_excel("rating_SP.xlsx")
catalogo = pd.read_excel("Catalogo.xlsx")

# reorder data
rating = clean_data(catalogo, spreads, rating)
spreads_sort = reshape_spreads(catalogo, spreads).join(rating['RATING_SP'])

# Params & Initializations
n_days = spreads.shape[0]
timeframe = 66


implied_rating = np.array([])


for day in range(timeframe, n_days):
    
    print('Step', day-timeframe, 'out of', n_days-timeframe-1)  # just to keep track of cycle
    x_train, y_train, x_val, y_val, scaler = split_train_val(day, spreads_sort, catalogo, timeframe)
    
    
    if (day-timeframe) % 10 == 0 or 'C' not in globals() or 'C_RBF' not in globals():
        eta, depth, child, subsample = tuning_xgb(x_train, y_train)  # C tuning for linear
       
    xgbclassifier = XGBClassifier(eta=eta, max_depth=depth, min_child_weight=child, subsample=subsample, objective=['multi:softmax'], label_encoder=False)
    xgbclassifier.fit(x_train, y_train)
    y_pred = xgbclassifier.predict(x_val)
    
    imp_rating, x_today, y_today = implied_ratings(spreads_sort, day, xgbclassifier, scaler)
    implied_rating = np.append(implied_rating, imp_rating)  # add to get imp ratings for all days
    
    # plot decision surfaces (change the if statement to plot different days)
    if spreads_sort['Date'][day] == datetime.datetime.strptime('2018/01/29', "%Y/%m/%d") \
        or spreads_sort['Date'][day] == datetime.datetime.strptime('2012/04/30', "%Y/%m/%d") \
            or spreads_sort['Date'][day] == datetime.datetime.strptime('2014/03/31', "%Y/%m/%d"):
        plot_decision_surface(x_train, x_today, y_today, xgbclassifier, scaler, spreads_sort['Date'][day])


print('end')


# np.savetxt("implied_rating_XGB.csv", implied_rating, delimiter=",",header="Implied Rating")
