# Trying to understand the POO better


# Creating an object
obj <- "string"

# Structure of the object
str(obj)

# Add one more structure to your object: Class
class(obj) <- 'myClass'

# The structure of the object now is:
str(obj)

# Exemplo of objects S3 (summary)
exClass_glm       <- glm(c(20:1) ~ c(1:20) )
exClass_dataframe <- data.frame(1:20)
str(exClass_glm) # a lot of structure
class(exClass_glm) # ... and two classes

# veja a diferenca: there is no methods for class "myClass", because we haven't defined anything
summary(exClass_glm)
summary(exClass_dataframe)
summary(obj)

# But if you define a method to your class....
summary.myClass <- function (x,y,...) {
  print('Hey, look at this !!')
  paste('My string is:',obj)
}
summary(obj)

# But S3 is like mother's heart: it accepts everything, which can be dangerous and produces errors at some point
x <- 3
summary(x)
class(x) <- 'glm'
summary(x)
