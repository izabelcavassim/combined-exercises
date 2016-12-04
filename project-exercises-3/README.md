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
make_prior <- function(alpha) {

  #res = function(n = 1) mvrnorm(n, mu = c(0,0), Sigma = diag(1/alpha, nrow = 2))
  return(alpha)
}

## Posterior Distribution p(w|Data)p(w)
make_posterior <-function(formula, alpha, beta, ...){
  theta_x = model_matrix(formula, ...)
  
  S_xy_inv = diag(alpha, nrow = ncol(theta_x)) + beta * t(theta_x) %*% theta_x
  
  # Inverting back the matrix by solving it
  S_xy = solve(S_xy_inv)
  m_xy = beta * S_xy %*% t(theta_x) %*% y
  
  return(list(S_xy = S_xy, m_xy = m_xy))
}

## Sampling from posterior
sample_from_posterior <- function(n, formula, alpha, beta) {
  # specialize a function by n
  stat = make_posterior(formula, alpha, beta)
  S_xy = stat$S_xy
  m_xy = stat$m_xy

  res <- as.data.frame(mvrnorm(n, mu=m_xy, Sigma=S_xy))
  w_names = sprintf("w%d", 0:(ncol(res)-1)) 
  colnames(res) = w_names
  return(res)
}

## Dealing with formulas 
model_matrix <- function(formula, ...){
  model.matrix(formula, ...)
}

## 
noresponse_matrix <- function(formula, ...){
  responseless = delete.response(terms(formula))
  
  data_frame = model.frame(responseless, ...)
  
  res = model.matrix(responseless, data_frame)
  return(res) 
}

##
predict <- function(formula, posterior, beta, ...) {
  
  theta_x = noresponse_matrix(formula)

  m_xy = posterior$m_xy
  S_xy = posterior$S_xy

  # Dealing with many data points:
  sds = vector(length = nrow(theta_x))
  means = vector(length = nrow(theta_x))
  
  for(i in seq_along(sds)){      
    S_xxy = 1/beta + (t(theta_x[i,]) %*% S_xy %*% theta_x[i,])
    sds[i] = sqrt(S_xxy)
  }
  
  for(i in seq_along(means)){
    means[i] = sum(t(m_xy) %*% theta_x[i,])
  }
  return(list(sds = sds, means = means))
  
}
```

``` r
# Creating the class blm

update <- function(model, prior, ...){
  
  obj = make_posterior(model, prior, beta, ...)
}
blm <- function(model, ...) {
  prior <- make_prior(alpha) # it returns alpha
  posterior <- update(model, prior, ...)
    # some code that returns an object here...
  obj <- list(mean = mean)
  class(obj) <- 'blm'
  obj
}
```

### Model methods

There are some polymorphic functions that are generally provided by classes that represent fitted models. Not all models implement all of them, but the more you implement, the more existing code can manipulate your new class; another reason for providing interfaces to objects through functions only.

Below is a list of functions that I think your `blm` class should implement. The functions are listed in alphabetical order, but many of them are easier to implement by using one or more of the others. So read through the list before you start programming. If you think that one function can be implemented simpler by calling one of the others, then implement it that way.

In all cases, read the R documentation for the generic function first. You need the documentation to implement the right interface for each function anyway so you might at least read the whole thing. The description in this note is just an overview of what the functions should do.

#### coefficients

This function should return fitted parameters of the model. It is not entirely straightforward to interpret what that means with our Bayesian models where a fitted model is a distribution and not a single point parameter. We could let the function return the fitted distribution, but the way this function is typically used that would make it useless for existing code to access the fitted parameters for this model as a drop in replacement for the corresponding parameters from a `lm` model, for example. Instead, it is probably better to return the point estimates of the parameters which would be the mean of the posterior you compute when fitting.

Return the result as a numeric vector with the parameters named. That would fit what you get from `lm`.

``` r
# YOUR IMPLEMENTATION
x <- runif(10)
y <- rnorm(10, mean=x)

# Looking at lm function to get some inspiration
test = lm( x ~ y)
test$coefficients
```

    ## (Intercept)           y 
    ##   0.5767825   0.0395258

#### confint

The function `confint` gives you confidence intervals for the fitted parameters. Here we have the same issue as with `coefficients`: we infer an entire distribution and not a parameter (and in any case, our parameters do not have confidence intervals; they have a joint distribution). Nevertheless, we can compute the analogue to confidence intervals from the distribution we have inferred.

If our posterior is distributed as **w** ∼ *N*(**m**, **S**) then component *i* of the weight vector is distributed as *w*<sub>*i*</sub> ∼ *N*(*m*<sub>*i*</sub>, **S**<sub>*i*, *i*</sub>). From this, and the desired fraction of density you want, you can pull out the thresholds that match the quantiles you need.

You take the `level` parameter of the function and get the threshold quantiles by exploiting that a normal distribution is symmetric. So you want the quantiles to be `c(level/2, 1-level/2)`. From that, you can get the thresholds using the function `qnorm`.

``` r
confint <- function(x, level, posterior, beta) {
  newdist = predict(x, posterior, beta)
  quantil = qnorm(c(level/2, 1-level/2), mean = newdist$mean, sd = newdist$sd_xxy, lower.tail = F)
  return(quantil)
}

# Vectorizing X
confint <- Vectorize(confint, vectorize.args = 'x')
```

#### deviance

This function just computes the sum of squared distances from the predicted response variables to the observed. This should be easy enough to compute if you could get the squared distances, or even if you only had the distances and had to square them yourself. Perhaps there is a function that gives you that?

#### fitted

This function should give you the fitted response variables. This is *not* the response variables in the data you fitted the model to, but instead the predictions that the model makes.

``` r
# YOUR IMPLEMENTATION
```

#### plot

This function plots your model. You are pretty free to decide how you want to plot it, but I could imagine that it would be useful to see an x-y plot with a line going through it for the fit. If there are more than one predictor variable, though, I am not sure what would be a good way to visualise the fitted model. There are no explicit rules for what the `plot` function should do, except for plotting something so you can use your imagination.

``` r
# Plot for lm has a LOT of information
plot(test)
```

![](README_files/figure-markdown_github/unnamed-chunk-7-1.png)![](README_files/figure-markdown_github/unnamed-chunk-7-2.png)![](README_files/figure-markdown_github/unnamed-chunk-7-3.png)![](README_files/figure-markdown_github/unnamed-chunk-7-4.png)

``` r
# 1- residuals versus fitted data
# 2- q-q plot (theoretical quantiles versus standardized residuals)
# 3- third graph i do not understand much
# 4- Residuals versus leverage
```

#### predict

This function should make predictions based on the fitted model. Its interface is

``` r
predict(object, ...)
```

but the convention is that you give it new data in a variable `newdata`. If you do not provide new data, it instead gives you the predictions on the data used to fit the model.

``` r
# Testing it

x <- runif(10)
y <- rnorm(10, mean=x)

posterior <- make_posterior(y ~ x, 1.3, 1)

newdata <- data.frame(x=runif(5))

predict(y ~ x, posterior = posterior, beta = 1, newdata)
```

    ## $sds
    ##  [1] 1.053517 1.055814 1.063738 1.065969 1.051406 1.060754 1.044693
    ##  [8] 1.085832 1.044042 1.099711
    ## 
    ## $means
    ##  [1] 0.4857939 0.4705282 0.4263599 0.8403548 0.7544717 0.4417951 0.6801514
    ##  [8] 0.9202932 0.5899497 0.9657361

#### print

This function is what gets called if you explicitly print an object or if you just write an expression that evaluates to an object of the class in the R terminal. Typically it prints a very short description of the object.

For fitted objects, it customarily prints how the fitting function was called and perhaps what the fitted coefficients were or how good the fit was. You can check out how `lm` objects are printed to see an example.

If you want to print how the fitting function was called you need to get that from when you fit the object in the `blm` constructor. It is how the constructor was called that is of interest, after all. Inside that function, you can get the way it was called by using the function `sys.call`.

``` r
# Print gives the called function and the coefficients 
print(test)
```

    ## 
    ## Call:
    ## lm(formula = x ~ y)
    ## 
    ## Coefficients:
    ## (Intercept)            y  
    ##     0.57678      0.03953

#### residuals

This function returns the residuals of the fit. That is the difference between predicted values and observed values for the response variable.

``` r
predict(y ~ x, posterior = posterior, beta = 1, newdata)
```

    ## $sds
    ##  [1] 1.053517 1.055814 1.063738 1.065969 1.051406 1.060754 1.044693
    ##  [8] 1.085832 1.044042 1.099711
    ## 
    ## $means
    ##  [1] 0.4857939 0.4705282 0.4263599 0.8403548 0.7544717 0.4417951 0.6801514
    ##  [8] 0.9202932 0.5899497 0.9657361

``` r
y
```

    ##  [1]  0.2294591  0.8143193  0.3606876  1.2824453 -0.2689327  0.4590383
    ##  [7]  0.2241737  1.3266842  0.7347939  1.9330505

``` r
x
```

    ##  [1] 0.13917345 0.11434626 0.04251364 0.71581013 0.57613499 0.06761651
    ##  [7] 0.45526494 0.84581719 0.30856617 0.91972275

``` r
test$residuals
```

    ##           1           2           3           4           5           6 
    ## -0.05252498  0.29686291 -0.59100164 -0.25215114  0.01295572  0.27000768 
    ##           7           8           9          10 
    ##  0.04641263  0.09972199  0.20190486 -0.03218803

#### summary

This function is usually used as a longer version of print. It gives you more information about the fitted model.

It does more than this, however. It returns an object with summary information. What that actually means is up to the model implementation so do what you like here.

``` r
# As the name says, the summary function should be summary ofeverything
summary(test)
```

    ## 
    ## Call:
    ## lm(formula = x ~ y)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -0.59100 -0.04744  0.02968  0.17636  0.29686 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)  0.57678    0.09359   6.163  0.00027 ***
    ## y            0.03953    0.07978   0.495  0.63362    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.2808 on 8 degrees of freedom
    ## Multiple R-squared:  0.02977,    Adjusted R-squared:  -0.09151 
    ## F-statistic: 0.2455 on 1 and 8 DF,  p-value: 0.6336
