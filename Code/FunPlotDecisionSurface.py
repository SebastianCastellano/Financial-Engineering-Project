def plot_decision_surface(x_train, x_today, y_today, classifier, scaler, date):

    #Plot of the decision surface obtained through the classifier used
    #INPUTS: x_train = training set of spreads
    #        x_today =
    #        y_today =
    #        classifier = classifier used for the training
    #                     (between svclassifier and xgbclassifier)
    #        scaler = standard scaler to normalize new data following
    #                 normalization used for training

    import matplotlib.pyplot as plt
    import numpy as np

    # first drop all countries with na spreads or CCC rating
    idx_na = x_today[x_today.isna().any(axis=1)].index  # save index were imp rating will be overwritten
    x_today_wo_na = x_today.drop(idx_na)
    y_today_wo_na = y_today.drop(idx_na)
    idx_ccc = y_today_wo_na[~y_today_wo_na['RATING_SP'].isin([1, 2, 3, 4, 5])].index  # select were  rating is CCC
    x_today_wo_na = x_today_wo_na.drop(idx_ccc)
    y_today_wo_na = y_today_wo_na.drop(idx_ccc)

    # adapt to scaling
    x_today_wo_na = scaler.transform(x_today_wo_na)

    # plot
    h = .02  # step size in the mesh
    x_min, x_max = x_train[:, 1].min() - 1e-2, x_train[:, 1].max() + 1e-2  # Bond Spread
    y_min, y_max = x_train[:, 0].min() - 1e-2, x_train[:, 0].max() + 1e-2  # CDS
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.coolwarm, alpha=0.8)

    col = np.array(y_today_wo_na.RATING_SP)
    markers = ["o", "^", "s", "*", "h"]
    label = ["AAA", "$A A$", "$A$", "$B B B$", "$B B$"]
    colors = ['mediumblue', 'royalblue', 'lightsteelblue', 'salmon', 'firebrick']
    for i, c in enumerate(np.unique(col)):
        plt.scatter(x_today_wo_na[:, 1][col == c], x_today_wo_na[:, 0][col == c], color=colors[c - 1],
                    cmap=plt.cm.coolwarm, marker=markers[i], s=144, label=label[i])
    plt.legend()
    plt.xlabel('Bond Spreads')
    plt.ylabel('CDS')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('Decision Surface ' + str(date)[:10])
    plt.show()

    return
