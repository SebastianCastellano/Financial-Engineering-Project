def prec_rec_f1(N, N_hat, alpha):

    #Compute precision indexes (defined in section 5 of the report)
    #INPUTS:     N = dataframe N
    #        N_hat = dataframe N_hat
    #        alpha = value in {-1, 0, 1}, where -1 indicates an upgrade,
    #                0 no migration and 1 a downgrade
    #OUTPUTS: precision = number of rating movements correctly predicted as alpha over
    #                     the total number of predictions of alpha
    #            recall = number of rating movements correctly predicted as alpha over
    #                     the total number of actual $\alpha$
    #                F1 = performance score defined as function of precision and recall

    precision = ((N == N_hat) & (N == alpha)).sum().sum() / (N_hat == alpha).sum().sum()
    recall = ((N == N_hat) & (N == alpha)).sum().sum() / (N == alpha).sum().sum()
    F1 = 2 * (recall * precision) / (recall + precision)

    return precision, recall, F1