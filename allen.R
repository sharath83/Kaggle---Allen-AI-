setwd("/Users/homw/Documents/petp/AllenAI/")
library(tm)
library(lsa)
library(slam)
train <- read.delim("training_set.tsv", sep= "\t", header = TRUE)
test <- read.delim("validation_set.tsv", sep= "\t", header = TRUE)
sub <- read.csv("sample_submission-2.csv")
docs <- readLines('Concepts - CK-12 Foundation.txt', encoding = "utf-8")
docs <- docs[docs != ""]
#--------create a corpus with docs + qst + (qst+optionX)
create.corp <- function(docs,train){}
docs <- as.data.frame(docs)
docs$type <- "s" #source
docs$id <- 0
# add qsts and qst+optionX to corpus
qst <- train[,"question"]
optA <- paste(qst, train[,"answerA"], sep = " ")
optB <- paste(qst, train[,"answerB"], sep = " ")
optC <- paste(qst, train[,"answerC"], sep = " ")
optD <- paste(qst, train[,"answerD"], sep = " ")

qst <- as.data.frame(qst)
qst$id <- train$id
qst$type <- "q"
names(qst)[1] <- "docs"

optA <- as.data.frame(optA)
optA$id <- train$id
optA$type <- "a"
names(optA)[1] <- "docs"

optB <- as.data.frame(optB)
optB$id <- train$id
optB$type <- "b"
names(optB)[1] <- "docs"

optC <- as.data.frame(optC)
optC$id <- train$id
optC$type <- "c"
names(optC)[1] <- "docs"

optD <- as.data.frame(optD)
optD$id <- train$id
optD$type <- "d"
names(optD)[1] <- "docs"
corp <- rbind.data.frame(docs,qst,optA,optB,optC,optD)


#Create a TDM
corpus <- VCorpus(DataframeSource(as.data.frame(corp[,"docs"])))
# clean and compute tfidf
corpus.clean = tm_map(corpus, stripWhitespace)                
corpus.clean = tm_map(corpus.clean, removeNumbers)                   
corpus.clean = tm_map(corpus.clean, removePunctuation)          
corpus.clean = tm_map(corpus.clean, content_transformer(tolower))  
corpus.clean = tm_map(corpus.clean, removeWords, stopwords("english"))
corpus.clean = tm_map(corpus.clean, stemDocument)                     
corpus.clean = DocumentTermMatrix(corpus.clean, control = list(weighting = weightTfIdf))

#inspect(corpus.clean[1:2])
#Remove empty documents
row.sum <- apply(corpus.clean,1, "sum")
corpus.clean.tf <- corpus.clean.tf[row.sum>5,]
corpus.clean <- corpus.clean[row.sum>0]
#tweets_full <- tweets_full[row.sum>0,]

#Idea tdm on corpus + qst + (qst+optionX). Get sentences closer to qst first then find answers
x <- corpus.clean[1:10,]
length(corpus.clean$i[corpus.clean$i == 10000])
y <- corpus.clean[3:4,]
sum(x[1,])
(x[1,] * y[2,])
# Get top matched sentences with atleast 5 words for every question 
docs <- corp[corp$type=='s',]
n <- nrow(docs)
q
for (q in n:(n+nrow(qst)-1)){
  for (s in 1:nrow(docs)){
    sim.c[s] = cosine(cosine(corpus.clean[s,],corpus.clean[q,]))
  }
  
    
}
# Calculates cosine similarity between a and b
cos.sim=function(a, b){
  return(sum(a*b)/sqrt(sum(a^2))*sqrt(sum(b^2)))
}
q = 1+n
sim = docs$id
tm <- proc.time()
for (s in 1:100){
  if(length(corpus.clean$i[corpus.clean$i == s]) >= 5){
    sim[s] = cos.sim(as.vector(corpus.clean[s,]),as.vector(corpus.clean[q,]))
  }
}

tm <- proc.time()
cosine_sim_mat <- crossprod_simple_triplet_matrix(corpus.clean)/(sqrt(row_sums(corpus.clean^2) %*% t(row_sums(corpus.clean^2))))
proc.time()-tm
y <- row_sums(corpus.clean^2)
