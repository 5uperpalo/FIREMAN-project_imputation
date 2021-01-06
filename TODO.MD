# TODO
* compare my with original results
* add documentation

* nice project with some ideas : https://rmisstastic.netlify.app/
* missing data definitions? https://onlinelibrary.wiley.com/doi/book/10.1002/9781119013563 
* https://www.theanalysisfactor.com/mar-and-mcar-missing-data/
* add simple imputation methods : https://medium.com/@Cambridge_Spark/tutorial-introduction-to-missing-data-imputation-4912b51c34eb 
  * only look at the distribution of the values of the variable with missing entries:
    * Mean
    * Median
    * Mode
    * model-based imputation like regression imputation
    * K-nearest neighbour (KNN) imputation
    * LOCF (Last observation carried forward), in pandas, this can be done using the ffill method in .fillna
    * interpolation - not usable for stream scenario
    * Predictive mean matching, combines the idea of model-based imputation (regression imputation) and neighbour-based (KNN imputer).
    * Multiple Imputation (MI), one of the most popular MICE (multivariate imputation by chained equations)(see [2]) and a python implementation is available in the fancyimpute package.
* convolutional layers to catch data chracteristics from previous N samples
  * the final data stream could be sent in high granularity to catch even slight glitches with a risk of missing samples as they would be imputed by DeepNet
* create data missing scenarios with appropriate functions
  * MCAR (missing completely at random) - There’s no relationship between whether a data point is missing and any values in the data set, missing or observed
  * MAR (missing at random) - probability of a data point to be missing is not related to the missing data, but it is related to some of the observed data.Whether or not someone answered #13 on your survey has nothing to do with the missing values, but it does have to do with the values of some other variable. (A better name would be Missing Conditionally at Random)
  * MNAR (missing not at random) -  there is a relationship between the propensity of a value to be missing and its values
  * variants : random missing(% from DS), random missing with continuous segments(% from DS, min/max size of continuous segment, probability of max missing segment, mechanism of probabilities distribution from min to max segment - linear/exponential)
* add variational autoencoders