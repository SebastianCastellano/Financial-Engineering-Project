def implied_ratings(spreads_sort, day, svclassifier, scaler):
    # Returns today's implied ratings
    # INPUT: spreads_sort =  dataframe with all spreads
    #        day    = today's index
    #        svclassifier  = svm classifier already fitted on training
    #        scaler = object containing scaling parameters used for training
    #
    # OUTPUT: imp_rating = today's implied ratings
    #         x_today = today's spreads
    #         y_today = today's SP rating

    x_today = spreads_sort[spreads_sort['Date'] == spreads_sort['Date'][day]][['CDS', 'Bond Spread']]\
        .reset_index(drop=True)
    y_today = spreads_sort[spreads_sort['Date'] == spreads_sort['Date'][day]][['RATING_SP']].reset_index(drop=True)

    idx_na = x_today[x_today.isna().any(axis=1)].index  # save index were imp rating will be overwritten
    x_today_wo_na = x_today.fillna(0)  # put zeros in place of na
    x_today_wo_na = scaler.transform(x_today_wo_na)  # normalization

    imp_rating = svclassifier.predict(x_today_wo_na)
    imp_rating[idx_na] = 0  # 0 indicates data na, consider changing 0

    return imp_rating, x_today, y_today
