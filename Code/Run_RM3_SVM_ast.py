import numpy as np
import pandas as pd
import datetime
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

from FunCleanData import clean_data
from FunReshapeSpreads import reshape_spreads
from FunSplitTrainVal import split_train_val
from FunTuningC import tuning_c
from FunPlotDecisionSurface import plot_decision_surface
from FunImpliedRating import implied_ratings

# DISCLAIMER: the run of the main script can be avoided since all the meaningful 
# results are stored into  .csv files, so one can run Run_RM3_Performance.py only


# import files
spreads = pd.read_excel("bnd_cds_5y.xlsx")
rating = pd.read_excel("rating_SP.xlsx")
catalogo = pd.read_excel("Catalogo.xlsx")
n_countries = catalogo.shape[0]

# reorder data
rating = clean_data(catalogo, spreads, rating)
spreads_sort = reshape_spreads(catalogo, spreads)

# shift ratings (objective is to associate spreads with 22 days ahead ratings)
pred_horizon = 22
rating = rating.sort_values(["Date", "BBG_TICKER"])  # reorder by date first
rating_shift = rating['RATING_SP'].shift(-pred_horizon*n_countries)  # shift ratings
spreads_sort = spreads_sort.sort_values(["Date", "Country"]).join(rating_shift)  # reshape and add shifted ratings col
spreads_sort = spreads_sort.iloc[:-pred_horizon*n_countries].sort_values(["Country", "Date"]).reset_index(drop=True)

# params & initializations
n_days = spreads.shape[0]-pred_horizon
timeframe = 66

implied_rating = np.array([])
implied_rating_RBF = np.array([])

for day in range(timeframe, n_days):
    print('Step', day-timeframe, 'out of', n_days-timeframe-1)
    x_train, y_train, x_val, y_val, scaler = split_train_val(day, spreads_sort, catalogo, timeframe)
    
    # parameter tuning every 10 days
    if (day-timeframe) % 10 == 0 or 'C' not in globals() or 'C_RBF' not in globals():
        C = tuning_c(x_train, y_train, 0)  # C tuning for linear
        C_RBF, gamma = tuning_c(x_train, y_train, 1)  # C and gamma tuning for rbf

    # fit model on train test and check results with validation
    svclassifier = SVC(C=C, kernel='linear', decision_function_shape='ovo')
    svclassifier.fit(x_train, y_train)
    y_pred = svclassifier.predict(x_val)
    
    svclassifier_RBF = SVC(C=C_RBF, kernel='rbf', gamma=gamma, decision_function_shape='ovo')
    svclassifier_RBF.fit(x_train, y_train)
    y_pred_RBF = svclassifier_RBF.predict(x_val)

    # compute daily implied ratings for all countries
    imp_rating, x_today, y_today = implied_ratings(spreads_sort, day, svclassifier, scaler)
    implied_rating = np.append(implied_rating, imp_rating)  # add to get imp ratings for all days
    
    imp_rating_RBF, _, _ = implied_ratings(spreads_sort, day, svclassifier_RBF, scaler)
    implied_rating_RBF = np.append(implied_rating_RBF, imp_rating_RBF)
    
    # plot decision surfaces (change the if statement to plot different days)
    if spreads_sort['Date'][day] == datetime.datetime.strptime('2009/03/30', "%Y/%m/%d") \
        or spreads_sort['Date'][day] == datetime.datetime.strptime('2012/04/30', "%Y/%m/%d") \
            or spreads_sort['Date'][day] == datetime.datetime.strptime('2014/03/31', "%Y/%m/%d"):
        plot_decision_surface(x_train, x_today, y_today, svclassifier, scaler, spreads_sort['Date'][day])
        plot_decision_surface(x_train, x_today, y_today, svclassifier_RBF, scaler, spreads_sort['Date'][day])
        
print('end')

# Save all data files to be used in the second main script

# np.savetxt("implied_rating_asterisco.csv", implied_rating, delimiter=",",header="Implied Rating")
# np.savetxt("implied_rating_RBF_asterisco.csv", implied_rating_RBF, delimiter=",",header="Implied Rating RBF")
