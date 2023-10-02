def split_train_val(day, spreads_sort, catalogo, timeframe):
    
    # Function to split randomly the samples into training and validation sets 
    # for a given time window (e.g. 'timeframe' days before the current day)
    
    # INPUT: day = current day of analysis
    #        spread_sort = dataframe  with sorted according to country and date
    #                        with CDS and BS of each issuer placed one next to 
    #                        other
    #        catalogo = file.xlsx with the relationship between the name of a
    #                   financial product and a country
    #        timeframe = range of days before'day' to be used for training
    #                    (it defines our training window, e.g. timeframe=66)
        
    # OUTPUT: x_train = training set of spreads 
    #         y_train = rating associated to each 'x_train'
    #         x_val = validation set of spreads 
    #         y_val = rating associated to each 'x_val'
    #         scaler = std scaler to normalize new data following the 
    #                  normalization used for training
        
    from sklearn.preprocessing import StandardScaler

    # select data in training time window
    start_date = spreads_sort['Date'][day-timeframe]
    end_date = spreads_sort['Date'][day]
    data = spreads_sort[spreads_sort['Date'] <= end_date]
    data = data[data['Date'] > start_date]

    # check for nan or CCC
    data_na = data[data.isna().any(axis=1) | ~data['RATING_SP'].isin([1, 2, 3, 4, 5])]
    country_na = data_na['Country'].values
    data = data[~data['Country'].isin(country_na)]  # remove countries with data na from train
    n_na = data_na['Country'].nunique()  # number of countries excluded
    n_countries = catalogo.shape[0] - n_na

    # from our 66 days * 35 countries = 2310 rows we split between train and test set
    rand_train_countries = catalogo['BBG_TICKER'].sample(n=round(0.8*n_countries)).values
    train_dataset = data[data['Country'].isin(rand_train_countries)]
    val_dataset = data[~data['Country'].isin(rand_train_countries)]

    # Separating the data from the target
    target = ['RATING_SP']
    features = ['CDS', 'Bond Spread']

    # Preparing X and y for the training set
    x_train = train_dataset[features]
    y_train = train_dataset[target]
    scaler = StandardScaler().fit(x_train)  # normalization
    x_train = scaler.transform(x_train)

    # Preparing X and y for the test set
    x_val = val_dataset[features]
    y_val = val_dataset[target]
    x_val = scaler.transform(x_val)  # normalization: N.B. no fit, same mean and var as test

    # turn y_train and y_val into arrays to match x_train and x_val
    y_train = y_train.values.ravel().astype('int')
    y_val = y_val.values.ravel().astype('int')

    return x_train, y_train, x_val, y_val, scaler
