def compute_ns(catalogo, rating, h, pred_horizon):
    # Computes N matrices for all methods
    # INPUT: catalogo =  dataframe with all spreads
    #        rating    = dataframe with all SP ratings
    #        h  =  horizon for N to evaluate migration
    #
    # OUTPUT: n = matrix N, compares rating today and rating h days from now:
    #             1 if there's a migration, 0 otherwise
    #         n_hat, n_hat_rbf, n_hat_ast, n_hat_rbf_ast, n_hat_xgb =
    #            matrices N_hat for all methods, compares imp ratings and SP ratings:
    #            1 if above, 0 if same, -1 if below

    import pandas as pd
    import numpy as np

    idx_na = rating[
        (rating['Implied Rating'] == 0) | (
                    rating['RATING_SP'] == 0)].index  # select where imp rating is na (0) or rating_sp is CCC(+,-) (0)
    idx_na_shift = rating[rating['RATING_SP'] == 0].index  # select where rating_sp is CCC(+,-) (0)

    n_hat = pd.DataFrame(columns=list(catalogo['BBG_TICKER']))
    n_hat_rbf = pd.DataFrame(columns=list(catalogo['BBG_TICKER']))
    n_hat_ast = pd.DataFrame(columns=list(catalogo['BBG_TICKER']))
    n_hat_rbf_ast = pd.DataFrame(columns=list(catalogo['BBG_TICKER']))
    n_hat_xgb = pd.DataFrame(columns=list(catalogo['BBG_TICKER']))
    n = pd.DataFrame(columns=list(catalogo['BBG_TICKER']), index=list(range(int(rating.shape[0] / catalogo.shape[0]))))

    for country in list(catalogo['BBG_TICKER']):
        n_hat[country] = np.sign(
            rating[rating['BBG_TICKER'] == country]['Implied Rating'] - rating[rating['BBG_TICKER'] == country][
                'RATING_SP']).reset_index(drop=True)
        n_hat_rbf[country] = np.sign(
            rating[rating['BBG_TICKER'] == country]['Implied Rating RBF'] - rating[rating['BBG_TICKER'] == country][
                'RATING_SP']).reset_index(drop=True)
        n_hat_ast[country] = np.sign(
            rating[rating['BBG_TICKER'] == country]['Implied Rating *'] - rating[rating['BBG_TICKER'] == country][
                'RATING_SP']).reset_index(drop=True)
        n_hat_rbf_ast[country] = np.sign(
            rating[rating['BBG_TICKER'] == country]['Implied Rating RBF *'] - rating[rating['BBG_TICKER'] == country][
                'RATING_SP']).reset_index(drop=True)
        n_hat_xgb[country] = np.sign(
            rating[rating['BBG_TICKER'] == country]['Implied Rating XGB'] - rating[rating['BBG_TICKER'] == country][
                'RATING_SP']).reset_index(drop=True)

        n[country] = (np.sign(rating[rating['BBG_TICKER'] == country]['RATING_SP'].iloc[h - 1:].reset_index(drop=True) -
                              rating[rating['BBG_TICKER'] == country]['RATING_SP'].iloc[0:(-h + 1)].reset_index(
                                  drop=True)))  # .shift(h)

    for i in idx_na:
        idx_col = i % 35
        idx_row = i // 35
        n_hat.iloc[idx_row, idx_col] = np.nan
        n_hat_rbf.iloc[idx_row, idx_col] = np.nan
        n_hat_ast.iloc[idx_row, idx_col] = np.nan
        n_hat_rbf_ast.iloc[idx_row, idx_col] = np.nan
        n_hat_xgb.iloc[idx_row, idx_col] = np.nan

    for i in idx_na_shift:
        idx_col = i % 35
        idx_row = i // 35
        n.iloc[idx_row, idx_col] = np.nan  # compare r(t)
        n.iloc[idx_row - h, idx_col] = np.nan  # compare r(t+h)

    return n, n_hat, n_hat_rbf, n_hat_ast, n_hat_rbf_ast, n_hat_xgb
