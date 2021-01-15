# TODO

## Main
* add example of Raytune usage 
* create data missing scenarios with appropriate functions
  * MCAR 
  * MAR
  * MNAR
  * variants : random missing(% from DS), random missing with continuous segments(% from DS, min/max size of continuous segment, probability of max missing segment, mechanism of probabilities distribution from min to max segment - linear/exponential)
* compare results with paper
* add variational autoencoders
* add simple [imputation methods](https://medium.com/@Cambridge_Spark/tutorial-introduction-to-missing-data-imputation-4912b51c34eb) that only look at the distribution of the values of the variable with missing entries:
  * Mean
  * Median
  * Mode
  * model-based imputation, e.g. regression imputation
  * K-nearest neighbour (KNN) imputation
  * LOCF (Last Observation Carried Forward), in pandas, this can be done using the ffill method in .fillna
  * interpolation(not usable for stream scenario)
  * Predictive mean matching, combines the idea of model-based imputation (regression imputation) and neighbour-based (KNN imputer).
  * Multiple Imputation (MI), one of the most popular MICE (multivariate imputation by chained equations) and a python implementation is available in the fancyimpute package.
