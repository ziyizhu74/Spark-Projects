
#### (Please use published Databricks links below for better visuals.)



### 1) SF Crime Analysis Using Spark SQL and PySpark DataFrame

[San Francisco Crime Analysis with Apache Spark for Big Data Processing](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2266522961450861/3112813335632301/421022585219207/latest.html)

- Provided residents in SF and tourists with travel safety tips and business insights based on modeling and analysis of 10 most common crimes, top-3 dangerous neighbourhoods, and resolved/unresolved rates for different categories of crimes. Produced predictive models for future crime events by performing two time series analyses.

- Utilized Python Urllib Module to access a large-scale open source database from DataSF.org for San Francisco crime analysis. Gained an in-depth understanding of distributed systems, algorithms, and cloud infrastructure by solving big data issues via PySpark dataframe, Spark SQL, and SQL in Databricks.

- Built Python user-defined functions (UDF) to filter geographical data located at a certain distance from the city’s Financial District, specified by a spatial range as ‘downtown San Francisco’.


### 2) Recommender System

[Movie Recommendations with Spark Collaborative Filtering](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2266522961450861/2354609215160139/421022585219207/latest.html)

- Created a hybrid recommender system using both memory-based and model-based collaborative filtering computed by cosine-similarity and matrix factorization. Trained an Alternating Least Square (ALS) model with 5 latent factors and 5 regularization terms to predict the unknown ratings, respectively.

- Optimized the ALS algorithm with Spark RDD-based MLlib API by measuring root-mean-square error of rating prediction.

- Validated the performance of the ML model by manipulating rating data from MovieLens.org. Implemented the RMSE evaluation metric and output minimum error at 0.891.

- Exposed the underlying patterns and relationships contained within the data through data visualizations using Python Matplotlib and Seaborn libraries. Plotted learning curves to detect overfitting for the Decision Tree Regression estimator,  suggested model complexity reduction techniques for a high variance scenario.


### 3) Youtube User Comments Semantic Analysis

[YouTube Comments Sentiment Analysis and Natural Language Processing](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2266522961450861/1462984688696549/421022585219207/latest.html)

- Parsed and comprehended YouTube comments which classify users as either a cat/dog owner or not an owner. Trained supervised ML models including Logistic Regression, Decision Tree, Random Forests, and Gradient Boosting.

- Preprocessed a 630MB dataset by data collection, cleaning, storage, and labeling. Conducted exploratory data analysis (EDA) to visualize the comment length distribution, dealt with imbalanced labels through resampling methods.

- Architected a ML pipeline to transform categorical variables to numeric variables, vectorized feature columns using Word2Vec.

- Designed four classifiers with the maximum AUC and identified the best model (GBT at 0.962). Evaluated model performance utilizing k-fold cross validation strategy, combined Python sklearn with matplotlib to plot ROC curves and Confusion Matrices. 

- Applied NLP techniques to identify top video creators, provided topic recommendations for the owners via PySpark and SQL.



### Thank you!
