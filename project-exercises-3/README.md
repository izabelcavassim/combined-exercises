Class project: Bayesian linear regresssion
------------------------------------------

Implement a constructor for a `blm` class. One approach, taken from the textbook, is implementing an `update` function and a `blm` function:

``` r
update <- function(model, prior, ...) { ... }
blm <- function(model, ...) {
    # some code here...
    prior <- make_a_prior_distribution_somehow()
    posterior <- update(model, prior, ...)
    # some code that returns an object here...
}
```

To get this version of `blm` to work you need to get the prior in a form you can pass along to `update` but if you did the exercises earlier you should already have a function that does this (although you might want to create a class for these distributions and return them as such so you can manipulate them through an interface if you want to take it a bit further).

``` r
# Importing useful functions

## Prior (Likelihhod Distribution) p(Data|w)
make_prior <- function(model, alpha, ...) {
  
  res = function(n = 1) mvrnorm(n, mu = c(0,0), Sigma = diag(1/alpha, nrow = 2))
  return(res)
}

## Posterior Distribution p(w|Data)p(w)
make_posterior <-function(model, alpha, beta, ...){
  theta_x = model_matrix(model, ...)
  
  S_xy_inv = diag(alpha, nrow = ncol(theta_x)) + beta * t(theta_x) %*% theta_x
  
  # Inverting back the matrix by solving it
  S_xy = solve(S_xy_inv)
  m_xy = beta * S_xy %*% t(theta_x) %*% y
  
  return(list(S_xy = S_xy, m_xy = m_xy))
}

## Sampling from posterior
sample_from_posterior <- function(n, model, alpha, beta, ...) {
  # specialize a function by n
  stat = make_posterior(model, alpha, beta)
  S_xy = stat$S_xy
  m_xy = stat$m_xy

  res <- as.data.frame(mvrnorm(n, mu=m_xy, Sigma=S_xy))
  w_names = sprintf("w%d", 0:(ncol(res)-1)) 
  colnames(res) = w_names
  return(res)
}

## Dealing with models 
model_matrix <- function(model, ...){
  model.matrix(model, ...)
}

## 
noresponse_matrix <- function(model, ...){
  responseless = delete.response(terms(model))
  
  data_frame = model.frame(responseless, ...)
  
  res = model.matrix(responseless, data_frame)
  return(res) 
}

##
predict.blm <- function(object, ...) {
  formula = object$formula
  posterior = object$posterior
  beta = object$beta
  
  # Generalizing theta for old and new data (using '...' method)
  theta_x = noresponse_matrix(formula, ...)

  m_xy = posterior$m_xy
  S_xy = posterior$S_xy

  # Dealing with many data points:
  sds = vector(length = nrow(theta_x))
  means = vector(length = nrow(theta_x))
  var = vector(length = nrow(theta_x))
  
  for(i in seq_along(sds)){      
    S_xxy = 1/beta + (t(theta_x[i,]) %*% S_xy %*% theta_x[i,])
    sds[i] = sqrt(S_xxy)
    var[i] = S_xxy
    means[i] = sum(t(m_xy) %*% theta_x[i,])
  }
  
  return(list(sds = sds, means = means, S_xy = var))
  
}
```

``` r
# Creating the class blm
make_prior <- function(model, alpha, ...){
  arguments <- list(...)
  if (alpha < 0) stop('alpha must be positive!')
  
  model <- model.matrix(model)
  return(list(m_xy = rep(0, ncol(model)), S_xy = diag(1/alpha, nrow = ncol(model))))
}

update <- function(model, prior, alpha, beta, ...){
  arguments <- list(...)

  if(alpha < 0 || beta  < 0) stop('alpha and beta must be positive!')
  
  data <- model.frame(model)
  theta_x = noresponse_matrix(model)
  S_xy <- solve(prior$S_xy + beta * t(theta_x) %*% theta_x)
  m_xy <- beta * S_xy %*% t(theta_x) %*% y
  return(list(m_xy=m_xy, S_xy=S_xy))
  
}

blm <- function(model, ...) {
  arguments <- list(...)
  alpha = arguments$alpha
  beta = arguments$beta
  prior <- make_prior(model, alpha,...) # it returns alpha
  posterior <- update(model, prior, beta, ...)

  # Defining the class blm
  obj <- list(data = model.frame(model),
              variances = posterior$S_xy,
              prior = prior,
              alpha = alpha,
              beta = beta,
              posterior = posterior,
              formula = model,
              sys = sys.call(),
              coefficients = posterior$m_xy)
  class(obj) <- 'blm'
  obj
}
```

### Model methods

There are some polymorphic functions that are generally provided by classes that represent fitted models. Not all models implement all of them, but the more you implement, the more existing code can manipulate your new class; another reason for providing interfaces to objects through functions only.

Below is a list of functions that I think your `blm` class should implement. The functions are listed in alphabetical order, but many of them are easier to implement by using one or more of the others. So read through the list before you start programming. If you think that one function can be implemented simpler by calling one of the others, then implement it that way.

In all cases, read the R documentation for the generic function first. You need the documentation to implement the right interface for each function anyway so you might at least read the whole thing. The description in this note is just an overview of what the functions should do.

#### coefficients

This function should return fitted arguments of the model. It is not entirely straightforward to interpret what that means with our Bayesian models where a fitted model is a distribution and not a single point parameter. We could let the function return the fitted distribution, but the way this function is typically used that would make it useless for existing code to access the fitted arguments for this model as a drop in replacement for the corresponding arguments from a `lm` model, for example. Instead, it is probably better to return the point estimates of the arguments which would be the mean of the posterior you compute when fitting.

Return the result as a numeric vector with the arguments named. That would fit what you get from `lm`.

``` r
x <- runif(30)
y <- rnorm(30, mean=x)

test_blm = blm(y ~ x, alpha = 1.5, beta = 2)
test_lm = lm( y ~ x)
# Creating the polymorphic function and adding an alias to it:
coefficients.blm <- coef.blm <- function(object, ...) t(object$coefficients)

# Test blm
coefficients(test_blm)
```

    ##      (Intercept)        x
    ## [1,]   0.2117622 1.069265

``` r
coef(test_blm)
```

    ##      (Intercept)        x
    ## [1,]   0.2117622 1.069265

``` r
# Test lm
coef(test_lm)
```

    ## (Intercept)           x 
    ##   0.1304033   1.2429709

#### predict

This function should make predictions based on the fitted model. Its interface is

``` r
predict(object, ...)
```

but the convention is that you give it new data in a variable `newdata`. If you do not provide new data, it instead gives you the predictions on the data used to fit the model.

``` r
predict.blm <- function(object, ...) {

  formula = object$formula
  posterior = object$posterior
  beta = object$beta
  
  theta_x = noresponse_matrix(formula, ...)

  # Generalizing theta for old and new data (using '...' method)
  m_xy = posterior$m_xy
  S_xy = posterior$S_xy

  # Dealing with many data points:
  means = vector(length = nrow(theta_x))
  
  for(i in seq_along(means)){      
    means[i] = sum(t(m_xy) %*% theta_x[i,])
  }

  return(list(means = means))
}

# Testing it
newdata <- data.frame(x=runif(3))

# blm method
predict(test_blm, newdata)
```

    ## $means
    ## [1] 0.9685022 0.3226858 0.9634446

``` r
# lm method
predict(test_lm, newdata)
```

    ##         1         2         3 
    ## 1.0100788 0.2593469 1.0041995

``` r
# blm method
predict(test_blm)
```

    ## $means
    ##  [1] 0.9245613 0.4694480 0.5266712 0.6174817 0.9613401 0.6013718 0.8649643
    ##  [8] 0.4857446 0.7931087 0.6369158 0.7042054 0.3270996 1.1327334 0.8061811
    ## [15] 0.4593193 0.7299614 1.2521330 0.7784668 0.4012082 0.7569475 0.4349237
    ## [22] 1.0562333 0.2823214 0.9194080 0.6832079 0.9398539 1.0896919 0.8087405
    ## [29] 1.1298591 0.2376254

``` r
# lm method
predict(test_lm)
```

    ##         1         2         3         4         5         6         7 
    ## 0.9589995 0.4299511 0.4964706 0.6020336 1.0017532 0.5833065 0.8897207 
    ##         8         9        10        11        12        13        14 
    ## 0.4488953 0.8061919 0.6246248 0.7028458 0.2644777 1.2009900 0.8213880 
    ##        15        16        17        18        19        20        21 
    ## 0.4181770 0.7327860 1.3397865 0.7891714 0.3506255 0.7641561 0.3898183 
    ##        22        23        24        25        26        27        28 
    ## 1.1120622 0.2124251 0.9530090 0.6784373 0.9767764 1.1509563 0.8243632 
    ##        29        30 
    ## 1.1976487 0.1604680

``` r
# to make like lm
x <- rnorm(30)
names(x) <- seq(1, length(x))

# to extract the indexes
as.numeric(names(x[x > 0.45])) # or which
```

    ##  [1]  7 10 12 13 14 16 18 20 23 25 29

#### fitted

This function should give you the fitted response variables. This is *not* the response variables in the data you fitted the model to, but instead the predictions that the model makes.

``` r
# You can only put the old data here, the newdata will be ignored in the predict function 
fitted.blm <- function(object, ...) {
  fit <- predict(object)
  fit
}

# Test blm
fitted(test_blm)
```

    ## $means
    ##  [1] -0.38856051 -0.81351870  0.63278260 -0.39515237 -1.70517175
    ##  [6]  0.47936530  1.73049525  0.26355275  0.12810693  0.91840499
    ## [11] -0.43774630  1.33725163  1.93486388  1.37913707 -0.87633088
    ## [16]  0.93391537 -1.19052878  2.20169344 -0.26994506  2.17105292
    ## [21]  0.16860484  0.56460794  1.02923113  0.48660776  1.18642264
    ## [26] -0.08190906 -1.20876192 -1.52937347  0.85936815 -0.15755810

``` r
# Even if I give new data it is not going to do anything with it:
fitted(test_blm, newdata)
```

    ## $means
    ##  [1] -0.38856051 -0.81351870  0.63278260 -0.39515237 -1.70517175
    ##  [6]  0.47936530  1.73049525  0.26355275  0.12810693  0.91840499
    ## [11] -0.43774630  1.33725163  1.93486388  1.37913707 -0.87633088
    ## [16]  0.93391537 -1.19052878  2.20169344 -0.26994506  2.17105292
    ## [21]  0.16860484  0.56460794  1.02923113  0.48660776  1.18642264
    ## [26] -0.08190906 -1.20876192 -1.52937347  0.85936815 -0.15755810

``` r
# Test lm
fitted(test_lm)
```

    ##         1         2         3         4         5         6         7 
    ## 0.9589995 0.4299511 0.4964706 0.6020336 1.0017532 0.5833065 0.8897207 
    ##         8         9        10        11        12        13        14 
    ## 0.4488953 0.8061919 0.6246248 0.7028458 0.2644777 1.2009900 0.8213880 
    ##        15        16        17        18        19        20        21 
    ## 0.4181770 0.7327860 1.3397865 0.7891714 0.3506255 0.7641561 0.3898183 
    ##        22        23        24        25        26        27        28 
    ## 1.1120622 0.2124251 0.9530090 0.6784373 0.9767764 1.1509563 0.8243632 
    ##        29        30 
    ## 1.1976487 0.1604680

``` r
# How does lm deal with it?
fitted(test_lm, newdata)
```

    ##         1         2         3         4         5         6         7 
    ## 0.9589995 0.4299511 0.4964706 0.6020336 1.0017532 0.5833065 0.8897207 
    ##         8         9        10        11        12        13        14 
    ## 0.4488953 0.8061919 0.6246248 0.7028458 0.2644777 1.2009900 0.8213880 
    ##        15        16        17        18        19        20        21 
    ## 0.4181770 0.7327860 1.3397865 0.7891714 0.3506255 0.7641561 0.3898183 
    ##        22        23        24        25        26        27        28 
    ## 1.1120622 0.2124251 0.9530090 0.6784373 0.9767764 1.1509563 0.8243632 
    ##        29        30 
    ## 1.1976487 0.1604680

``` r
# Don't care like I do. :)
```

#### confint

The function `confint` gives you confidence intervals for the fitted arguments. Here we have the same issue as with `coefficients`: we infer an entire distribution and not a parameter (and in any case, our arguments do not have confidence intervals; they have a joint distribution). Nevertheless, we can compute the analogue to confidence intervals from the distribution we have inferred.

If our posterior is distributed as **w** ∼ *N*(**m**, **S**) then component *i* of the weight vector is distributed as *w*<sub>*i*</sub> ∼ *N*(*m*<sub>*i*</sub>, **S**<sub>*i*, *i*</sub>). From this, and the desired fraction of density you want, you can pull out the thresholds that match the quantiles you need.

You take the `level` parameter of the function and get the threshold quantiles by exploiting that a normal distribution is symmetric. So you want the quantiles to be `c(level/2, 1-level/2)`. From that, you can get the thresholds using the function `qnorm`.

``` r
confint.blm <- function(object, parm = NULL, level= 0.95, ...) {
  theta_x = noresponse_matrix(object$formula, ...)
  beta = object$beta
  S_xy = object$posterior$S_xy
  m_xy = object$posterior$m_xy
  
  if(parm == 'y'){
    
  # Calculating Standard deviation to be used in qnorm
  sds = vector(length = nrow(theta_x))  

  for(i in seq_along(sds)){  
    S_xxy = 1/beta + (t(theta_x[i,]) %*% S_xy %*% theta_x[i,])
    sds[i] = sqrt(S_xxy)
  }
  
  fitted <- predict(object, ...)
  quantil_lower <- qnorm(p = (1-level)/2, mean = fitted$means, sd = sds, lower.tail = F)
  quantil_upper <- qnorm(p = (1 - (1 -level)/2), mean = fitted$means, sd = sds, lower.tail = F)
  
  quantiles = cbind(quantil_lower, quantil_upper)
  colnames(quantiles) = c((1-level)/2, (1 - (1 -level)/2))
  
  # it returns the confint of the fitted data
  return(quantiles)
  }
  
  # Mean and variance of the coefficients
  # I will give that as default
  if(is.null(parm)){
    parm = rownames(object$coefficients)
  }
  
  quantil_lower <- qnorm(p = (1-level)/2, mean = m_xy[parm,], sd = sqrt(diag(S_xy)[parm]), lower.tail = F)
  quantil_upper <- qnorm(p = (1 - (1 -level)/2), mean =m_xy[parm,], sd = sqrt(diag(S_xy)[parm]), lower.tail = F)
  
  quantiles = cbind(quantil_lower, quantil_upper)

  colnames(quantiles) = c((1-level)/2, (1 - (1 -level)/2))
  rownames(quantiles) = parm
return(quantiles)  
}

# Test blm
confint(test_blm, parm = '(Intercept)', level = 0.95)
```

    ##                 0.025      0.975
    ## (Intercept) 0.7176081 -0.2940836

``` r
# Test blm
confint(test_blm, parm = 'y', level = 0.95)
```

    ##           0.025       0.975
    ##  [1,] 1.3136292 -2.09075025
    ##  [2,] 1.1179956 -2.74503302
    ##  [3,] 2.0434107 -0.77784553
    ##  [4,] 1.3102306 -2.10053539
    ##  [5,] 0.8146601 -4.22500360
    ##  [6,] 1.9032613 -0.94453073
    ##  [7,] 3.3854878  0.07550267
    ##  [8,] 1.7262424 -1.19913692
    ##  [9,] 1.6264441 -1.37023028
    ## [10,] 2.3371459 -0.50033596
    ## [11,] 1.2885778 -2.16407045
    ## [12,] 2.8423251 -0.16782180
    ## [13,] 3.6886482  0.18107956
    ## [14,] 2.8972925 -0.13901837
    ## [15,] 1.0925852 -2.84524691
    ## [16,] 2.3543150 -0.48648429
    ## [17,] 0.9760780 -3.35713557
    ## [18,] 4.1013400  0.30204689
    ## [19,] 1.3770704 -1.91696056
    ## [20,] 4.0530991  0.28900669
    ## [21,] 1.6554147 -1.31820498
    ## [22,] 1.9796257 -0.85040982
    ## [23,] 2.4625165 -0.40405427
    ## [24,] 1.9096045 -0.93638894
    ## [25,] 2.6507919 -0.27794663
    ## [26,] 1.4873170 -1.65113516
    ## [27,] 0.9697977 -3.38732156
    ## [28,] 0.8664411 -3.92518802
    ## [29,] 2.2729324 -0.55419606
    ## [30,] 1.4414483 -1.75656446

``` r
# Test blm
confint(test_blm, parm = 'x', level = 0.95)
```

    ##      0.025     0.975
    ## x 1.989947 0.1485817

``` r
# Test blm
confint(test_blm, parm = 1, level = 0.95)
```

    ##       0.025      0.975
    ## 1 0.7176081 -0.2940836

``` r
# I want my confint function to accept any case of written 'intercept'

# Test lm
confint(test_lm)
```

    ##                  2.5 %   97.5 %
    ## (Intercept) -0.7770572 1.037864
    ## x           -0.4303242 2.916266

#### deviance

This function just computes the sum of squared distances from the predicted response variables to the observed. This should be easy enough to compute if you could get the squared distances, or even if you only had the distances and had to square them yourself. Perhaps there is a function that gives you that?

``` r
deviance.blm <- function(object, ...){
  observed = object$data[,1]
  fit = fitted(object)$means
  dev = (fit - observed)^2
  sum(dev)
}

# Test blm
deviance(test_blm)  
```

    ## [1] 85.09719

``` r
# Test lm
deviance(test_lm)
```

    ## [1] 34.67814

#### residuals

This function returns the residuals of the fit. That is the difference between predicted values and observed values for the response variable.

``` r
residuals.blm <- function(object, ...){
  observed = object$data[,1]
  x = as.data.frame(object$data[,2])
  predicted = predict(object, x)$means
  residuals = (observed - predicted) 
  residuals
}

# Test it for blm
residuals(test_blm)
```

    ##  [1]  2.60985661  2.54273829 -1.22449976  1.92481067  2.49256007
    ##  [6]  1.70174124 -0.57680084 -0.01629578 -1.04599150 -1.54607888
    ## [11]  2.33719824 -1.46456266  0.11473273 -1.52983265  1.32243484
    ## [16]  0.62275804  4.37103226 -0.71432204  0.43742576 -2.71623790
    ## [21]  0.26338034  0.73861035 -0.15875296  0.25228073 -0.40903872
    ## [26] -1.14058994  3.30167079  0.19682930 -0.87869825  0.72304966

``` r
# Test it for lm
residuals(test_lm)
```

    ##           1           2           3           4           5           6 
    ##  1.26229658  1.29926844 -1.08818771  0.92762472 -0.21436483  1.59780001 
    ##           7           8           9          10          11          12 
    ##  0.26397368 -0.20163828 -1.72407646 -1.25229866  1.19660610 -0.39178877 
    ##          13          14          15          16          17          18 
    ##  0.84860663 -0.97208358  0.02792697  0.82388738  1.84071699  0.69820003 
    ##          19          20          21          22          23          24 
    ## -0.18314484 -1.30934108  0.04216685  0.19115611  0.65805304 -0.21412051 
    ##          25          26          27          28          29          30 
    ##  0.09894662 -2.19927542  0.94195260 -2.15690732 -1.21697879  0.40502352

#### plot

This function plots your model. You are pretty free to decide how you want to plot it, but I could imagine that it would be useful to see an x-y plot with a line going through it for the fit. If there are more than one predictor variable, though, I am not sure what would be a good way to visualise the fitted model. There are no explicit rules for what the `plot` function should do, except for plotting something so you can use your imagination.

``` r
plot.blm <- function(object, ...) {
  data = object$data

  if(is.null(data) == FALSE && ncol(data) == 2){
   # plot(data[,2],data[,1], pch = 20, xlab = 'x', ylab = 'y')
  #  abline(a = object$coefficients[1], b = object$coefficients[2], col="red")
    
    # First plot: fitted values vs residuals
    fit = fitted(object)$means
    resid = residuals(object)
    plot(fit, resid, pch = 20, xlab = 'Fitted values', ylab = 'Residuals', main = 'Residuals vs Fitted') 
    abline(h = 0, col = "gray60", lty = 2)
    # I need to produce the red lines, but i don't know how
  }
  if(is.null(data) == FALSE && ncol(data) > 2){
    # take the m_xy and make eigen decomposition of this matrix and select the variable that explains most of the variance
    
  }
}






plot(test_blm)
```

![](README_files/figure-markdown_github/unnamed-chunk-9-1.png)

``` r
plot(test_lm)
```

![](README_files/figure-markdown_github/unnamed-chunk-9-2.png)![](README_files/figure-markdown_github/unnamed-chunk-9-3.png)![](README_files/figure-markdown_github/unnamed-chunk-9-4.png)![](README_files/figure-markdown_github/unnamed-chunk-9-5.png)

``` r
# 1- residuals versus fitted data
# 2- q-q plot (theoretical quantiles versus standardized residuals)
# 3- third graph i do not understand much
# 4- Residuals versus leverage
```

#### print

This function is what gets called if you explicitly print an object or if you just write an expression that evaluates to an object of the class in the R terminal. Typically it prints a very short description of the object.

For fitted objects, it customarily prints how the fitting function was called and perhaps what the fitted coefficients were or how good the fit was. You can check out how `lm` objects are printed to see an example.

If you want to print how the fitting function was called you need to get that from when you fit the object in the `blm` constructor. It is how the constructor was called that is of interest, after all. Inside that function, you can get the way it was called by using the function `sys.call`.

``` r
# Print gives/print the called function and the coefficients 
print.blm <- function(object, ...){
  cat('\nCall:\n')
  print(object$sys)
  
  cat('\nCoefficients:\n')
  t(object$coefficients) 
}

# Test it
print(test_blm)
```

    ## 
    ## Call:
    ## blm(y ~ x, alpha = 1.5, beta = 2)
    ## 
    ## Coefficients:

    ##      (Intercept)        x
    ## [1,]   0.2117622 1.069265

``` r
print(test_lm)
```

    ## 
    ## Call:
    ## lm(formula = y ~ x)
    ## 
    ## Coefficients:
    ## (Intercept)            x  
    ##      0.1304       1.2430

#### summary

This function is usually used as a longer version of print. It gives you more information about the fitted model.

It does more than this, however. It returns an object with summary information. What that actually means is up to the model implementation so do what you like here.

``` r
# As the name says, the summary function should be summary of everything
summary(test_lm)
```

    ## 
    ## Call:
    ## lm(formula = y ~ x)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -2.19928 -0.82701  0.07056  0.84243  1.84072 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)
    ## (Intercept)   0.1304     0.4430   0.294    0.771
    ## x             1.2430     0.8169   1.522    0.139
    ## 
    ## Residual standard error: 1.113 on 28 degrees of freedom
    ## Multiple R-squared:  0.07637,    Adjusted R-squared:  0.04339 
    ## F-statistic: 2.315 on 1 and 28 DF,  p-value: 0.1393

``` r
summary.blm <- function(object, ...) {
  # Defining the class summary
  obj <- list(data = object$data,
              Call = object$sys,
              Residuals = residuals(object),
              Coefficients = object$coefficients)
  class(obj) <- 'summary.blm'
  obj
}

summary(test_blm)
```

    ## $data
    ##             y          x
    ## 1   2.2212961 0.66662554
    ## 2   1.7292196 0.24099341
    ## 3  -0.5917172 0.29450987
    ## 4   1.5296583 0.37943786
    ## 5   0.7873883 0.70102187
    ## 6   2.1811065 0.36437150
    ## 7   1.1536944 0.61088910
    ## 8   0.2472570 0.25623440
    ## 9  -0.9178846 0.54368814
    ## 10 -0.6276739 0.39761301
    ## 11  1.8994519 0.46054375
    ## 12 -0.1273110 0.10786607
    ## 13  2.0495966 0.86131269
    ## 14 -0.1506956 0.55591377
    ## 15  0.4461040 0.23152082
    ## 16  1.5566734 0.48463136
    ## 17  3.1805035 0.97297782
    ## 18  1.4873714 0.52999472
    ## 19  0.1674807 0.17717405
    ## 20 -0.5451850 0.50986933
    ## 21  0.4319852 0.20870560
    ## 22  1.3032183 0.78976814
    ## 23  0.8704782 0.06598849
    ## 24  0.7388885 0.66180603
    ## 25  0.7773839 0.44090649
    ## 26 -1.2224990 0.68092749
    ## 27  2.0929089 0.82105937
    ## 28 -1.3325442 0.55830735
    ## 29 -0.0193301 0.85862455
    ## 30  0.5654916 0.02418777
    ## 
    ## $Call
    ## blm(y ~ x, alpha = 1.5, beta = 2)
    ## 
    ## $Residuals
    ##  [1]  2.60985661  2.54273829 -1.22449976  1.92481067  2.49256007
    ##  [6]  1.70174124 -0.57680084 -0.01629578 -1.04599150 -1.54607888
    ## [11]  2.33719824 -1.46456266  0.11473273 -1.52983265  1.32243484
    ## [16]  0.62275804  4.37103226 -0.71432204  0.43742576 -2.71623790
    ## [21]  0.26338034  0.73861035 -0.15875296  0.25228073 -0.40903872
    ## [26] -1.14058994  3.30167079  0.19682930 -0.87869825  0.72304966
    ## 
    ## $Coefficients
    ##                  [,1]
    ## (Intercept) 0.2117622
    ## x           1.0692646
    ## 
    ## attr(,"class")
    ## [1] "summary.blm"

``` r
print.summary.blm <- function(object, ...) {
  sapply(object$data, summary)
  cat('\nCall:\n')
  print(object$Call)
  
  cat('\nResiduals:\n')
  print(summary(object$Residuals))
  #sapply(object$residuals, summary) 
  
  cat('\nCoefficients:\n')
  print(object$Coefficients)
  print('Estimate, Std error, t value, Pr(>|t|)')

  cat('\nResidual standard error, numberX, on numberZ degrees of freedom\n')
  cat('\nMultiple R-squared:\n')
  cat('Adjusted R-squared:\n')
  
} 
summary(test_blm)
```

    ## 
    ## Call:
    ## blm(y ~ x, alpha = 1.5, beta = 2)
    ## 
    ## Residuals:
    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ## -2.7160 -0.8376  0.2246  0.4177  1.6070  4.3710 
    ## 
    ## Coefficients:
    ##                  [,1]
    ## (Intercept) 0.2117622
    ## x           1.0692646
    ## [1] "Estimate, Std error, t value, Pr(>|t|)"
    ## 
    ## Residual standard error, numberX, on numberZ degrees of freedom
    ## 
    ## Multiple R-squared:
    ## Adjusted R-squared:

``` r
# The hard and freedom way

## Creating Classes of summary
# Using the same arguments as summary and creating a summary custom class

summary.customclass <- function(object, ...){
  cat('\nCall:\n')
  print(object$sys)
  
  cat('\nCoefficients:\n')
  t(object$coefficients) 
}
```
