import pandas as pd
from FunPrecRecF1 import prec_rec_f1
from FunComputeNs import compute_ns
from FunComputeV import compute_v

timeframe = 66

# Import files
catalogo = pd.read_excel("Catalogo.xlsx")
rating = pd.read_csv("rating.csv")
implied_rating = pd.read_csv("implied_rating.csv")
implied_rating_RBF = pd.read_csv("implied_rating_RBF.csv")
implied_rating_XGB = pd.read_csv("implied_rating_XGB.csv")
implied_rating_asterisco = pd.read_csv("implied_rating_asterisco.csv")
implied_rating_RBF_asterisco = pd.read_csv("implied_rating_RBF_asterisco.csv")

# shift asterisco cases so that the days of the implied ratings match with the other cases
extra_rows_1 = pd.DataFrame(list(range(770)), columns=['# Implied Rating'])
implied_rating_asterisco = implied_rating_asterisco.append(extra_rows_1)
implied_rating_asterisco = implied_rating_asterisco.shift(770).reset_index(drop=True)

extra_rows_2 = pd.DataFrame(list(range(770)), columns=['# Implied Rating RBF'])
implied_rating_RBF_asterisco = implied_rating_RBF_asterisco.append(extra_rows_2)
implied_rating_RBF_asterisco = implied_rating_RBF_asterisco.shift(770).reset_index(drop=True)

# Assemble complete rating matrix
start_date = catalogo.shape[0]*timeframe  # for each country drop first 66 days (our timeframe)
rating = rating.sort_values(['Date', 'BBG_TICKER']).reset_index(drop=True)
rating = rating.iloc[start_date:].reset_index(drop=True)
rating['Implied Rating'] = implied_rating
rating['Implied Rating RBF'] = implied_rating_RBF
rating['Implied Rating *'] = implied_rating_asterisco
rating['Implied Rating RBF *'] = implied_rating_RBF_asterisco
rating['Implied Rating XGB'] = implied_rating_XGB

# Compute all Ns and related statistics
h = 22 #change h to compute performance measuresfor different time horizons
pred_horizon = 22
N, N_hat, N_hat_RBF, N_hat_ast, N_hat_RBF_ast, N_hat_XGB = compute_ns(catalogo, rating, h, pred_horizon)

sample_size = rating.shape[0]

# Accuracies
accuracy = (N == N_hat).sum().sum()/sample_size
accuracy_RBF = (N == N_hat_RBF).sum().sum()/sample_size
accuracy_ast = (N == N_hat_ast).sum().sum()/(sample_size-770)
accuracy_RBF_ast = (N == N_hat_RBF_ast).sum().sum()/(sample_size-770)
accuracy_XGB = (N == N_hat_XGB).sum().sum()/sample_size

# Performance Indexes for linear SVM
precision_up, recall_up, F1_up = prec_rec_f1(N, N_hat, -1)
precision_stable, recall_stable, F1_stable = prec_rec_f1(N, N_hat, 0)
precision_down, recall_down, F1_down = prec_rec_f1(N, N_hat, 1)

# Performance Indexes for RBF SVM
precision_up_RBF, recall_up_RBF, F1_up_RBF = prec_rec_f1(N, N_hat_RBF, -1)
precision_stable_RBF, recall_stable_RBF, F1_stable_RBF = prec_rec_f1(N, N_hat_RBF, 0)
precision_down_RBF, recall_down_RBF, F1_down_RBF = prec_rec_f1(N, N_hat_RBF, 1)

# Performance Indexes for linear * SVM
precision_up_Ast, recall_up_Ast, F1_up_Ast = prec_rec_f1(N, N_hat_ast, -1)
precision_stable_Ast, recall_stable_Ast, F1_stable_Ast = prec_rec_f1(N, N_hat_ast, 0)
precision_down_Ast, recall_down_Ast, F1_down_Ast = prec_rec_f1(N, N_hat_ast, 1)

# Performance Indexes for RBF * SVM
precision_up_RBF_ast, recall_up_RBF_ast, F1_up_RBF_ast = prec_rec_f1(N, N_hat_RBF_ast, -1)
precision_stable_RBF_ast, recall_stable_RBF_ast, F1_stable_RBF_ast = prec_rec_f1(N, N_hat_RBF_ast, 0)
precision_down_RBF_ast, recall_down_RBF_ast, F1_down_RBF_ast = prec_rec_f1(N, N_hat_RBF_ast, 1)

# Performance Indexes for XGB
precision_up_XGB, recall_up_XGB, F1_up_XGB = prec_rec_f1(N, N_hat_XGB, -1)
precision_stable_XGB, recall_stable_XGB, F1_stable_XGB = prec_rec_f1(N, N_hat_XGB, 0)
precision_down_XGB, recall_down_XGB, F1_down_XGB = prec_rec_f1(N, N_hat_XGB, 1)

# Compute forecast matrix and plot early warnings around migrations
forecasts = pd.DataFrame()
forecasts = compute_v(rating, forecasts)
