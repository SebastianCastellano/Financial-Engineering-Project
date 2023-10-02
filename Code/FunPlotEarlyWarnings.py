def plot_early_warnings(rating_country, i, imp_col_name):

    #Plot of the difference between actual ratings and implied ratings in
    #the interval between 10 days before and 5 after a migration
    #INPUTS: rating_country = data for a single country
    #                     i = day of migration
    #         imp_col_names = method used to obtain the implied ratings
    #                         (SVM, SVM_RBF, SVM_asterisco, SVM_RBF_asterisco, XGB)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.xlabel('Country: ' + rating_country['BBG_TICKER'][1] + ' Migration date: ' + str(rating_country['Date'][i]))

    ax1 = rating_country['RATING_SP'].loc[i-10:i+5].plot(color='blue', grid=True, label='Rating S&P',  linestyle='--',
                                                         marker='o')
    ax2 = rating_country[imp_col_name].loc[i-10:i+5].plot(color='red', grid=True, secondary_y=True, label=imp_col_name,
                                                          linestyle='', marker='o')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    plt.legend(h1 + h2, l1 + l2, loc=2)
    plt.show()

    return
