def compute_v(rating, forecasts):
    # Checks for all migrations in rating if the models correctly anticipate them
    # INPUT: rating    = dataframe with all SP ratings
    #        forecast  =  initialization empty dataframe
    #
    # OUTPUT: forecasts = dataframe that for all migrations and all methods,
    #                     checks if for each time window I1 I2 I3 the implied
    #                     ratings correctly anticipate the migrations: 1=yes 0=no

    import numpy as np
    from FunPlotEarlyWarnings import plot_early_warnings

    plot_count = 0

    for country in rating['BBG_TICKER'].values[0:35]:  # for every country
        rating_country = rating[rating['BBG_TICKER'] == country].reset_index(drop=True)  # select current country data
        positions_of_mig = rating_country['RATING_SP'].diff()  # 0 when no migration, 1 when there is one
        # keep index of migrations, exclude false signals when rating becomes CCC:
        index_of_mig = rating_country.loc[(positions_of_mig[1:] != 0)
                                          & (rating_country['RATING_SP'] != 0)
                                          & (rating_country['RATING_SP'].shift(1) != 0)].index

        for i in index_of_mig:  # go through every migration, check if Vs are positive/negative
            new_row = forecasts.shape[0]
            forecasts.loc[new_row, 'Date'] = str(rating_country['Date'][i])
            forecasts.loc[new_row, 'Country'] = country
            migration = positions_of_mig[i]  # +1 if upgrade, -1 if downgrade
            for imp_col_name in rating.columns[3:]:  # for every method check prediction power

                i1 = rating_country.loc[i - 22:i - 1]
                i2 = rating_country.loc[i - 44: i - 23]
                i3 = rating_country.loc[i - 66: i - 45]

                v1 = (sum(i1[imp_col_name] - i1['RATING_SP'])) / 22
                v2 = (sum(i2[imp_col_name] - i2['RATING_SP'])) / 22
                v3 = (sum(i3[imp_col_name] - i3['RATING_SP'])) / 22

                if np.sign(v1) == np.sign(migration):
                    forecasts.loc[new_row, imp_col_name + ' I1'] = 1
                else:
                    forecasts.loc[new_row, imp_col_name + ' I1'] = 0
                if np.sign(v2) == np.sign(migration):
                    forecasts.loc[new_row, imp_col_name + ' I2'] = 1

                else:
                    forecasts.loc[new_row, imp_col_name + ' I2'] = 0
                if np.sign(v3) == np.sign(migration):
                    forecasts.loc[new_row, imp_col_name + ' I3'] = 1

                else:
                    forecasts.loc[new_row, imp_col_name + ' I3'] = 0

                plot_count += 1
                if plot_count == 10:  # just to select how many plots we want
                    plot_early_warnings(rating_country, i, imp_col_name)
                    plot_count = 0  # reset
    forecasts.sort_values(['Date', 'Country'], inplace=True, ignore_index=True)

    return forecasts
