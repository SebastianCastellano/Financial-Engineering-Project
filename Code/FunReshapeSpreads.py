def reshape_spreads(catalogo, spreads):

    #Reorders the spreads data in a more compact dataframe with columns
    #Date, Country, CDS, Bond Spread
    # INPUT: spreads = dataframe of CDS and Bond Spreads of an issuer
    #        catalogo  = file .xlsx with the relationship between the
    #                    name of a financial product and a country
    #
    # OUTPUT: spreads_sort = dataframe of same dim of spreads but with sorted 
    #                        CDS and BS of each issuer placed one next to other

    import pandas as pd

    spreads_sort = pd.DataFrame(columns=["Date", "Country", "CDS", "Bond Spread"])

    for index in catalogo.index:
        bbg_ticker = catalogo["BBG_TICKER"][index]
        cds_name = catalogo["CDS Series Name"][index]
        bs_name = catalogo["Bond Spread Series Name"][index]

        new_row = pd.DataFrame({"Date": spreads.Date, "Country": bbg_ticker, "CDS": spreads[cds_name],
                                "Bond Spread": spreads[bs_name]})

        spreads_sort = spreads_sort.append(new_row, ignore_index=True)

    return spreads_sort
