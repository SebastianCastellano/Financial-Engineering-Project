def clean_data(catalogo, spreads, rating):
    # Takes rating and corrects all errors and fills missing values
    # INPUT: catalogo =  file.xlsx with the relationship with the name of a
    #                    financial product and a country/company
    #        spreads    = dataframe of CDS and Bond Spreads of an issuer
    #        rating     = dataframe with rating of each country for a set of
    #                    dates (dates are not sorted, each issuer's observations
    #                    are listed for all dates)
    #
    # OUTPUT: rating = complete rating file

    import pandas as pd

    # filter out dates after 2018 and change labels into int
    rating = rating.loc[:, ["Date", "BBG_TICKER", "RATING_SP"]]
    rating = rating[rating["Date"] <= "2018/12/31"]
    rating["RATING_SP"].replace(
        {"AAA": 1, "AAA-": 1, "AA+": 2, "AA": 2, "AA-": 2, "A+": 3, "A": 3, "A-": 3, "BBB+": 4,
         "BBB": 4, "BBB-": 4, "BB+": 5, "BB": 5, "BB-": 5, "B+": 5,
         "B": 5, "B-": 5, "CCC+": 0, "CCC": 0, "CCC-": 0}, inplace=True)

    # replacing NR ratings with the latest available
    num_nr = 1
    r = rating["RATING_SP"]
    outlier = rating["RATING_SP"] == "NR"
    while num_nr > 0:
        r = r.mask(outlier, r.shift())
        num_nr = sum(r == "NR")
    rating["RATING_SP"] = r

    # insert missing dates in rating from spreads, putting the rating_sp=0
    for bbg in catalogo["BBG_TICKER"]:
        rem = set(rating[rating["BBG_TICKER"] == bbg]["Date"]) - set(spreads["Date"])
        rating = rating[~rating["Date"].isin(list(rem))]  # remove dates from rating not in spreads

        c = set(spreads["Date"]) - set(rating[rating["BBG_TICKER"] == bbg]["Date"])
        rating = rating.append(pd.DataFrame({"Date": list(c), "BBG_TICKER": bbg, "RATING_SP": 'New'}))

    # taking out duplicate rows
    rating.drop_duplicates(inplace=True)

    # order by country and then by date
    rating = rating.sort_values(["BBG_TICKER", "Date"])
    rating = rating.reset_index(drop=True)

    # fill rating_sp with the previous date one when it is 'New' (the dates we just added)
    num_new = 1
    r = rating["RATING_SP"]
    outlier = rating["RATING_SP"] == 'New'
    while num_new > 0:
        r = r.mask(outlier, r.shift())
        num_new = sum(r == 'New')
    rating["RATING_SP"] = r

    return rating
