################################################################
#
#   Definindo diretórios de trabalho
#
################################################################
setwd("./")
#setwd("D:/2022/projetos/FapitecIA/reunioes/26_abr_2022/R")
dados.dir    <- "../data/"

################################################################
#
#   Lista de pacotes R a serem carregados
#
################################################################

#Carregando a base de dados

dado <- read.csv(file=paste(dados.dir,  "raw-data-entropy-diversity-indices_v2.csv", sep=""), header=TRUE, sep=";") 

#Selecionando as vari?veis
colnames(dado)

efetivo <- 13:23
val.prodani <- 24:29
val.temp <- 97:127
val.perm <- 128:163
area.plantada.temp <- 164:194
aquicultura <- 231:254
extrativismo <- 255:297
silvicultura <- 298:312

dado.autoencoder <- dado[c(1,2, efetivo, val.prodani, val.temp, val.perm, 
                   aquicultura, extrativismo, silvicultura  )]

#sum(is.na(dado.dado.autoencoder))

#colSums(dado.dado.autoencoder==0) #Dado esparso - com muitos zeros

dado.autoencoder.t <- dado.autoencoder
for (i in 3:168) {
  max_i <- max( dado.autoencoder[,i])
  min_i <- min( dado.autoencoder[,i])
  dado.autoencoder.t[,i] <- (dado.autoencoder[,i] - min_i)/(max_i - min_i)
}
#summary(dado.autoencoder.t$EFETIVO_01)

#data.wide <- reshape(dado.autoencoder.t, idvar = "CODIBGE", timevar = "Ano", direction = "wide")

#install.packages("keras")
library(keras)
#install_keras(tensorflow = "cpu")

train.data <- as.matrix(sapply(dado.autoencoder.t[,c(3:168)], as.numeric))  

############################################################################################
#
#     SIMPLE UNDERCOMPLETE LINEAR AUTOENCODER
#
############################################################################################
model <- keras_model_sequential() %>%
  layer_dense(units = 50, activation = "linear", input_shape = ncol(train.data)) %>%
  layer_dense(units = 166, activation = "linear")
#Compile the model
model %>% compile(
  optimizer = "rmsprop", 
  loss = "mse",
  metrics = c('mse')
)
#Train the model
model %>% fit(x = train.data, y = train.data, batch = 20, epochs = 15, verbose = 2)

model1 <- keras_model_sequential() %>%
  layer_dense(units = 50, activation = "linear", input_shape = ncol(train.data)) %>%
  layer_dense(units = 166, activation = "linear")
#Compile the model
model1 %>% compile(
  optimizer = "adam", 
  loss = "mse",
  metrics = c('mse')
)
#Train the model
model1 %>% fit(x = train.data, y = train.data, batch = 20, epochs = 15, verbose = 2)

model2 <- keras_model_sequential() %>%
  layer_dense(units = 50, activation = "tanh", input_shape = ncol(train.data)) %>%
  layer_dense(units = 166, activation = "linear")
#Compile the model
model2 %>% compile(
  optimizer = "adam", 
  loss = "mse",
  metrics = c('mse')
)
#Train the model
model2 %>% fit(x = train.data, y = train.data, batch = 20, epochs = 15, verbose = 2)
############################################################################################
#
#     STACKED LINEAR AUTOENCODER - optimizer = "rmsprop"
#
############################################################################################

model3 <- keras_model_sequential() %>%
  layer_dense(units = 80, activation = "tanh", input_shape = ncol(train.data)) %>%
  layer_dense(units = 40, activation = "tanh") %>%
  layer_dense(units = 20, activation = "tanh") %>%
  layer_dense(units = 40, activation = "tanh") %>%
  layer_dense(units = 80, activation = "tanh") %>%
  layer_dense(units = 166, activation = "linear")
#Compile the model
model3 %>% compile(
  optimizer = "adam", 
  loss = "mse",
  metrics = c('mse')
)
#Train the model
model3 %>% fit(x = train.data, y = train.data, batch = 20, epochs = 15, verbose = 2)



############################################################################################
#
#     Extracting Latent representation  - One Hidden layer
#
############################################################################################
Weights_bias <- model2$get_weights()
number_of_hidden_layers <- 1
Latent <- Weights_bias[[number_of_hidden_layers]]
Latent_bias   <- Weights_bias[[number_of_hidden_layers+1]]

#Dataset            111400x166
#Latent weights     166x50
#Latent_bias        50


Transformed.data <- train.data %*% Latent
for (i in 1:nrow(Transformed.data))
      Transformed.data[i,] + Latent_bias
dim(Transformed.data)

############################################################################################
#
#     Extracting Latent representation  - More than one Hidden layer
#
############################################################################################


get.Latent.Representation <- function( model, data ) {
  Weights_bias <- model$get_weights()
  number_of_hidden_layers <- length( Weights_bias ) / 2 - 1
  tmp <- 1
  X <- data
  while (tmp <= number_of_hidden_layers ) {
    Latent <- Weights_bias[[tmp]]
    Latent_bias   <- Weights_bias[[tmp+1]]
    X <- X %*% Latent
    for (i in 1:nrow(X))
      X[i,] + Latent_bias
    tmp <- tmp + 2
  }
  return(X)
}

Transformed.data <-get.Latent.Representation( model3, train.data ) 

############################################################################################
#
#     Data exploration
#     Correlogram: https://towardsdatascience.com/customizable-correlation-plots-in-r-b1d2856a4b05
#
############################################################################################
library(ggcorrplot)

png(filename=paste("CorrelogramaDadosOriginais.png",  sep=""), width = 12, height = 10, units = 'in', res = 600)
ggcorrplot::ggcorrplot(cor(dado.autoencoder[,c(3:168)]))
dev.off()

png(filename=paste("CorrelogramaDadosTransformados.png",  sep=""), width = 12, height = 10, units = 'in', res = 600)
ggcorrplot::ggcorrplot(cor(dado.autoencoder.t[,c(3:168)]))
dev.off()

png(filename=paste("CorrelogramaDadosTransformadosSimpleUndercompleteLinearAutoencoder.png",  sep=""), width = 12, height = 10, units = 'in', res = 600)
ggcorrplot::ggcorrplot(cor(Transformed.data))
dev.off()

png(filename=paste("CorrelogramaDadosTransformadosStackedUndercompleteLinearAutoencoder.png",  sep=""), width = 12, height = 10, units = 'in', res = 600)
ggcorrplot::ggcorrplot(cor(t.data8))
dev.off()

png(filename=paste("CorrelogramaDadosTransformadosStackedUndercompleteNonLinearAutoencoder.png",  sep=""), width = 12, height = 10, units = 'in', res = 600)
ggcorrplot::ggcorrplot(cor(t.data11))
dev.off()


png(filename=paste("VarianciaDadosOriginais.png",  sep=""), width = 12, height = 8, units = 'in', res = 600)
var <- apply(dado.autoencoder[,c(3:168)], 2, var)
barplot( var, names=c(1:166), cex.names = 0.2 )
dev.off()

png(filename=paste("VarianciaDadosTransformados.png",  sep=""), width = 12, height = 8, units = 'in', res = 600)
var <- apply(dado.autoencoder.t[,c(3:168)], 2, var)
barplot( var, names=c(1:166), cex.names = 0.2 )
dev.off()

png(filename=paste("VarianciaDadosTransformadosLatentRepresentation.png",  sep=""), width = 12, height = 8, units = 'in', res = 600)
var <- apply(Transformed.data, 2, var)
barplot( var, cex.names = 0.2 )
dev.off()

############################################################################################
#
#     Comparing Autoencoders results
#     
#
############################################################################################

losses <- c(model$history$history$loss[ length(model$history$history$loss)])

#Losses
model3$history$history$loss

#Metrics
model3$history$history$mse
