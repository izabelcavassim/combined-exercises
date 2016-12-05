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
  parameters <- list(...)
  if (alpha < 0) stop('alpha must be positive!')
  
  model <- model.matrix(model)
  return(list(m_xy = rep(0, ncol(model)), S_xy = diag(1/alpha, nrow = ncol(model), alpha)))
}

update <- function(model, prior, alpha, beta, ...){
  parameters <- list(...)

  #if(alpha | beta  < 0) stop('alpha and beta must be positive!')
  
  data <- model.frame(model)
  theta_x = noresponse_matrix(model)
  S_xy <- solve(diag(alpha, nrow = ncol(theta_x)) + beta * t(theta_x) %*% theta_x)
  m_xy <- beta * S_xy %*% t(theta_x) %*% y
  return(list(m_xy=m_xy, S_xy=S_xy))
  
}

blm <- function(model, ...) {
  parameters <- list(...)
  alpha = parameters$alpha
  beta = parameters$beta
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

This function should return fitted parameters of the model. It is not entirely straightforward to interpret what that means with our Bayesian models where a fitted model is a distribution and not a single point parameter. We could let the function return the fitted distribution, but the way this function is typically used that would make it useless for existing code to access the fitted parameters for this model as a drop in replacement for the corresponding parameters from a `lm` model, for example. Instead, it is probably better to return the point estimates of the parameters which would be the mean of the posterior you compute when fitting.

Return the result as a numeric vector with the parameters named. That would fit what you get from `lm`.

``` r
x <- runif(10)
y <- rnorm(10, mean=x)

test = blm(x ~ y, alpha = 1.2, beta = 1.3)
# Creating the polymorphic function and adding an alias to it:
coefficients.blm <- coef.blm <- function(object, ...) object$coefficients

# Test
coefficients.blm(test)
```

    ##                    [,1]
    ## (Intercept) -0.02104039
    ## y            0.89118712

``` r
coef.blm(test)
```

    ##                    [,1]
    ## (Intercept) -0.02104039
    ## y            0.89118712

#### confint

The function `confint` gives you confidence intervals for the fitted parameters. Here we have the same issue as with `coefficients`: we infer an entire distribution and not a parameter (and in any case, our parameters do not have confidence intervals; they have a joint distribution). Nevertheless, we can compute the analogue to confidence intervals from the distribution we have inferred.

If our posterior is distributed as **w** ∼ *N*(**m**, **S**) then component *i* of the weight vector is distributed as *w*<sub>*i*</sub> ∼ *N*(*m*<sub>*i*</sub>, **S**<sub>*i*, *i*</sub>). From this, and the desired fraction of density you want, you can pull out the thresholds that match the quantiles you need.

You take the `level` parameter of the function and get the threshold quantiles by exploiting that a normal distribution is symmetric. So you want the quantiles to be `c(level/2, 1-level/2)`. From that, you can get the thresholds using the function `qnorm`.

``` r
confint.blm <- function(object, level= 50, ...) {
  posterior = object$posterior
  beta = object$beta
  
  newdist <- predict.blm(posterior, beta)
  quantil <- qnorm(c(level/2, 1-level/2), mean = newdist$mean, sd = newdist$sd_xxy, lower.tail = F)
  
  # It returns mean and sd
  return(quantil)
}

# Test
#confint.blm(test)
```

#### deviance

This function just computes the sum of squared distances from the predicted response variables to the observed. This should be easy enough to compute if you could get the squared distances, or even if you only had the distances and had to square them yourself. Perhaps there is a function that gives you that?

``` r
#deviance.blm <- function(object, ...)
```

#### fitted

This function should give you the fitted response variables. This is *not* the response variables in the data you fitted the model to, but instead the predictions that the model makes.

``` r
#fitted.blm <- function(object, ...)
```

#### plot

This function plots your model. You are pretty free to decide how you want to plot it, but I could imagine that it would be useful to see an x-y plot with a line going through it for the fit. If there are more than one predictor variable, though, I am not sure what would be a good way to visualise the fitted model. There are no explicit rules for what the `plot` function should do, except for plotting something so you can use your imagination.

``` r
#plot.blm <- function(object, ...)
# Plot for lm has a LOT of information
#plot(test)
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
predict.blm <- function(object, ...) {
  formula = object$model
  posterior = object$posterior
  beta = object$beta
  
  # Generalizing theta for old and new data (using '...' method)
  theta_x = noresponse_matrix(formula, ...)

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
# testing
newdata <- data.frame(x=runif(3))

#predict.blm(test, newdata)

# Testing it

x <- runif(10)
y <- rnorm(10, mean=x)

posterior <- make_posterior(y ~ x, 1.3, 1)

newdata <- data.frame(x=runif(5))

#predict(y ~ x, posterior = posterior, beta = 1, newdata)
```

#### print

This function is what gets called if you explicitly print an object or if you just write an expression that evaluates to an object of the class in the R terminal. Typically it prints a very short description of the object.

For fitted objects, it customarily prints how the fitting function was called and perhaps what the fitted coefficients were or how good the fit was. You can check out how `lm` objects are printed to see an example.

If you want to print how the fitting function was called you need to get that from when you fit the object in the `blm` constructor. It is how the constructor was called that is of interest, after all. Inside that function, you can get the way it was called by using the function `sys.call`.

``` r
# Print gives the called function and the coefficients 
bla = lm(y ~ x)
print(bla)
```

    ## 
    ## Call:
    ## lm(formula = y ~ x)
    ## 
    ## Coefficients:
    ## (Intercept)            x  
    ##      0.3657       1.5160

``` r
print.blm <- function(object, ...){
  cat('\nCall:\n')
  print(object$sys)
  
  cat('\nCoefficients:\n')
  t(object$coefficients) 
}

print.blm(test)
```

    ## 
    ## Call:
    ## blm(x ~ y, alpha = 1.2, beta = 1.3)
    ## 
    ## Coefficients:

    ##      (Intercept)         y
    ## [1,] -0.02104039 0.8911871

#### residuals

This function returns the residuals of the fit. That is the difference between predicted values and observed values for the response variable.

#### summary

This function is usually used as a longer version of print. It gives you more information about the fitted model.

It does more than this, however. It returns an object with summary information. What that actually means is up to the model implementation so do what you like here.

``` r
# As the name says, the summary function should be summary of everything
getAnywhere(summary.lm)
```

    ## A single object matching 'summary.lm' was found
    ## It was found in the following places
    ##   package:stats
    ##   registered S3 method for summary from namespace stats
    ##   namespace:stats
    ## with value
    ## 
    ## function (object, correlation = FALSE, symbolic.cor = FALSE, 
    ##     ...) 
    ## {
    ##     z <- object
    ##     p <- z$rank
    ##     rdf <- z$df.residual
    ##     if (p == 0) {
    ##         r <- z$residuals
    ##         n <- length(r)
    ##         w <- z$weights
    ##         if (is.null(w)) {
    ##             rss <- sum(r^2)
    ##         }
    ##         else {
    ##             rss <- sum(w * r^2)
    ##             r <- sqrt(w) * r
    ##         }
    ##         resvar <- rss/rdf
    ##         ans <- z[c("call", "terms", if (!is.null(z$weights)) "weights")]
    ##         class(ans) <- "summary.lm"
    ##         ans$aliased <- is.na(coef(object))
    ##         ans$residuals <- r
    ##         ans$df <- c(0L, n, length(ans$aliased))
    ##         ans$coefficients <- matrix(NA, 0L, 4L)
    ##         dimnames(ans$coefficients) <- list(NULL, c("Estimate", 
    ##             "Std. Error", "t value", "Pr(>|t|)"))
    ##         ans$sigma <- sqrt(resvar)
    ##         ans$r.squared <- ans$adj.r.squared <- 0
    ##         return(ans)
    ##     }
    ##     if (is.null(z$terms)) 
    ##         stop("invalid 'lm' object:  no 'terms' component")
    ##     if (!inherits(object, "lm")) 
    ##         warning("calling summary.lm(<fake-lm-object>) ...")
    ##     Qr <- qr.lm(object)
    ##     n <- NROW(Qr$qr)
    ##     if (is.na(z$df.residual) || n - p != z$df.residual) 
    ##         warning("residual degrees of freedom in object suggest this is not an \"lm\" fit")
    ##     r <- z$residuals
    ##     f <- z$fitted.values
    ##     w <- z$weights
    ##     if (is.null(w)) {
    ##         mss <- if (attr(z$terms, "intercept")) 
    ##             sum((f - mean(f))^2)
    ##         else sum(f^2)
    ##         rss <- sum(r^2)
    ##     }
    ##     else {
    ##         mss <- if (attr(z$terms, "intercept")) {
    ##             m <- sum(w * f/sum(w))
    ##             sum(w * (f - m)^2)
    ##         }
    ##         else sum(w * f^2)
    ##         rss <- sum(w * r^2)
    ##         r <- sqrt(w) * r
    ##     }
    ##     resvar <- rss/rdf
    ##     if (is.finite(resvar) && resvar < (mean(f)^2 + var(f)) * 
    ##         1e-30) 
    ##         warning("essentially perfect fit: summary may be unreliable")
    ##     p1 <- 1L:p
    ##     R <- chol2inv(Qr$qr[p1, p1, drop = FALSE])
    ##     se <- sqrt(diag(R) * resvar)
    ##     est <- z$coefficients[Qr$pivot[p1]]
    ##     tval <- est/se
    ##     ans <- z[c("call", "terms", if (!is.null(z$weights)) "weights")]
    ##     ans$residuals <- r
    ##     ans$coefficients <- cbind(est, se, tval, 2 * pt(abs(tval), 
    ##         rdf, lower.tail = FALSE))
    ##     dimnames(ans$coefficients) <- list(names(z$coefficients)[Qr$pivot[p1]], 
    ##         c("Estimate", "Std. Error", "t value", "Pr(>|t|)"))
    ##     ans$aliased <- is.na(coef(object))
    ##     ans$sigma <- sqrt(resvar)
    ##     ans$df <- c(p, rdf, NCOL(Qr$qr))
    ##     if (p != attr(z$terms, "intercept")) {
    ##         df.int <- if (attr(z$terms, "intercept")) 
    ##             1L
    ##         else 0L
    ##         ans$r.squared <- mss/(mss + rss)
    ##         ans$adj.r.squared <- 1 - (1 - ans$r.squared) * ((n - 
    ##             df.int)/rdf)
    ##         ans$fstatistic <- c(value = (mss/(p - df.int))/resvar, 
    ##             numdf = p - df.int, dendf = rdf)
    ##     }
    ##     else ans$r.squared <- ans$adj.r.squared <- 0
    ##     ans$cov.unscaled <- R
    ##     dimnames(ans$cov.unscaled) <- dimnames(ans$coefficients)[c(1, 
    ##         1)]
    ##     if (correlation) {
    ##         ans$correlation <- (R * resvar)/outer(se, se)
    ##         dimnames(ans$correlation) <- dimnames(ans$cov.unscaled)
    ##         ans$symbolic.cor <- symbolic.cor
    ##     }
    ##     if (!is.null(z$na.action)) 
    ##         ans$na.action <- z$na.action
    ##     class(ans) <- "summary.lm"
    ##     ans
    ## }
    ## <bytecode: 0x0000000017b8ef40>
    ## <environment: namespace:stats>

``` r
summary(test)
```

    ##              Length Class      Mode   
    ## data         2      data.frame list   
    ## variances    4      -none-     numeric
    ## prior        2      -none-     list   
    ## alpha        1      -none-     numeric
    ## beta         1      -none-     numeric
    ## posterior    2      -none-     list   
    ## formula      3      formula    call   
    ## sys          4      -none-     call   
    ## coefficients 2      -none-     numeric
