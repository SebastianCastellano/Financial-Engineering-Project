# Financial-Engineering-Project
Repository for the project of the course of Financial Engineering - Polimi 2020/2021

Team members: Sebastian Castellano, Federica Maddaloni, Stefano Tomassetti

Even though the role of credit rating agencies (CRAs) such as Standard & Poor's (S&P) and
Fitch is to provide credit ratings for different institutions, they are often late in signaling a
modification of their credit status. 
Thus, the objective of this project is to deduce a system of implied credit ratings from the bond and credit default swap markets that aims at anticipating
future migrations.
This idea is backed by many studies, which show that market spreads hold valuable information regarding credit ratings and rating migrations. The limit of these studies is that their results come from the analysis of the bond and the CDS markets separately, while this project aims to reach a more accurate implied rating system by joining them in a bi-dimensional framework, extracting meaningful information from each market. 
We resort to modern classification methods such as Support Vector Machines (SVM) and eXtreme Gradient Boosting (XGB) to deduce the market-implied rating which is associated to each couple of daily spreads.
