# load wine data

wine=read.table("data/WINE.data",
			header=F,
			sep=",",
			col.names=c("Wine Type","Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids",
			"Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline ")
		)

# Look at the data frame
str(wine)

library(ggplot2)
# scatter plot
qplot(Alcohol,Malic.acid,data=wine,color=Wine.Type,alpha=I(0.7))



