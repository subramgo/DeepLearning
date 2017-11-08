library(mxnet)
library(quantmod, quietly = TRUE)
library(ggplot2)

# Retrieve the ticker data
stock.data <- new.env()
tickers <- ('AAPL')
stock.data <- getSymbols(tickers, src = 'yahoo', from = '2000-01-01', env = FALSE, auto.assign = F)    

data <- stock.data$AAPL.Close
head(data)

# Plot the data
plot.data <- na.omit(data.frame(close_price = data))
names(plot.data) <- c("close_price")
ggplot(plot.data,aes(x = seq_along(close_price))) + geom_line(aes(y = close_price, color ="Close Price"))


# Feature generation
data.ts <- as.ts(data)
data.zoo <- as.zoo(data.ts)

# Last 31 days data is X
# Rolling window of 31 days
x.data <- list()
  for (j in 1:31){
    var.name <- paste("x.lag.",j)
    x.data[[var.name]] <- Lag(data.zoo,j)
  }
  
# 32nd day is Y
final.data <- na.omit(data.frame(x.data, Y = data.zoo))
head(final.data)


set.seed(100)
# Train/test split
train.perc = 0.8
train.indx = 1:as.integer(dim(final.data)[1] * train.perc)

train.data <- final.data[train.indx,]
test.data  <- final.data[-train.indx ,]

train.x.data <- data.matrix(train.data[,-1])
train.y.data <- train.data[,1]

test.x.data <- data.matrix(test.data[,-1])
test.y.data <- test.data[,1]



mx.set.seed(100)
deep.model <- mx.mlp(data = train.x.data, label = train.y.data,
                     hidden_node = c(1000,500,250)
                    ,out_node = 1
                    ,dropout = 0.50
                    ,activation = c("relu", "relu","relu")
                    ,out_activation = "rmse"
                    , array.layout = "rowmajor"
                    , learning.rate = 0.01
                    , array.batch.size = 100
                    , num.round = 100
                    , verbose = TRUE
                    , optimizer = "adam"
                    , eval.metric = mx.metric.mae

                    
                    
)


model.evaluate <- function(deep.model, new.data, actual){
  preds = predict(deep.model, new.data, array.layout = "rowmajor")
  error <- actual - preds
  return(mean(abs(error)))
  
}

print("Train Error")
model.evaluate(deep.model,train.x.data, train.y.data)


print("Test Error")
model.evaluate(deep.model,test.x.data, test.y.data)


preds = predict(deep.model,  train.x.data, array.layout = "rowmajor")
plot.data <- data.frame(actual = train.y.data, predicted = preds[1,])
ggplot(plot.data,aes(x = seq_along(actual))) + geom_line(aes(y = actual, color = "Actual")) + geom_line(aes(y = predicted, color = "Predicted"))

preds = predict(deep.model,  test.x.data, array.layout = "rowmajor")
plot.data <- data.frame(actual = test.y.data, predicted = preds[1,])
ggplot(plot.data,aes(x = seq_along(actual))) + geom_line(aes(y = actual, color ="actual")) + geom_line(aes(y = predicted, color ="predicted"))

graph.viz(deep.model$symbol)

library(Matrix)
# Look at the graph plot for the name of each layer
# alternatively call deep.model$arg.params$  to see the name
weights <- deep.model$arg.params$fullyconnected102_weight
dim(weights)
image(as.matrix(weights))

weights.1 <- deep.model$arg.params$fullyconnected19_weight
dim(weights.1)
image(as.matrix(weights.1))

weights.2 <- deep.model$arg.params$fullyconnected20_weight
dim(weights.2)
image(as.matrix(weights.2))





# Parameter tuning funciton
random.search <- function(){

  # Sample layers
  count.layers <- sample(2:5, 1)
  no.layers <- c()
  activations <- c()
  for (i in 1:count.layers-1){
    # Sample node per layers
    no.nodes <- sample(10:50,1)
    no.layers[i] <- no.nodes
    activations[i] <- "relu"
  }
  
  no.layers <- append(no.layers, 1)
  activations <- append(activations, "relu")
  
  deep.model <- mx.mlp(data = train.x.data, label = train.y.data,
                       hidden_node = no.layers
                       , out_node = 1
                       , dropout = 0.50
                       , activation = activations
                       ,out_activation = "rmse"
                       , array.layout = "rowmajor"
                       , learning.rate = 0.01
                       , array.batch.size = 100
                       , num.round = 10
                       , verbose = TRUE
                       , optimizer = "adam"
                       , eval.metric = mx.metric.mae
                       
                       
                       
  )
  
  
  train.error <- model.evaluate(deep.model,train.x.data, train.y.data)
  test.error <- model.evaluate(deep.model,test.x.data, test.y.data)
  
  output <- list(layers = no.layers, activations <- activations
                 , train.error = train.error, test.error = test.error)
  return(output)
}

final.output = list()
for (i in 1:2){
  out <- random.search()
  final.output[[i]] <- out
}

