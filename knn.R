" -------------------------------------------------------------
  K-Nearest Neighbors Algorithm
  
  @author Cui, Elliott, Lai, Steuer
  
  This program is our implementation of the k-nearest 
  neghbors algorithm for classification.
  
  We only used the first two features 
  of the iris data so that we can plot it
  ------------------------------------------------------------- "

#bind a matrix and a vector vertically
vert_bind <- function(matrix1, vector){ 
  result = matrix(0, nrow(matrix1), ncol(matrix1) + 1)
  for(i in 1 : nrow(matrix1)){
    for (j in 1 : ncol(matrix1)){
      result[i, j] = matrix1[i,j]
    }
  }
  for(i in 1 : length(vector)){
    result[i, ncol(result)] = vector[i]
  }
  return (result)
}


#compute distances between the feature vectors and return a distance matrix
distance <- function(train_set,test_set){
  dis_matrix = matrix(0, nrow(test_set), nrow(train_set))
  for(i in 1 : nrow(test_set)){
    n = ncol(test_set)
    l_1 = vector(length = n)
    for (k in 1: n) {
      l_1[k] = test_set[i,k];
    }
    #x1 = test_set[i,1]
    #y1 = test_set[i,2]
    for(j in 1 : nrow(train_set)){
      
      n = ncol(train_set)
      l_2 = vector(length = n)
      for (t in 1: n) {
        l_2[t] = train_set[j,t];
      }
      
      #x2 = train_set[j,1]
      #y2 = train_set[j,2]
      #d = sqrt((x2-x1)^2+(y2-y1)^2)
      d = dist(rbind(l_1, l_2))
      dis_matrix[i, j] = d
    }
  }
  return (dis_matrix)
}


#make a copy of a matrix
copy <- function(matx){  
  copy_matx = matrix(0, nrow(matx), ncol(matx))
  for(i in 1 : nrow(matx)){
    for(j in 1 : ncol(matx)){
      copy_matx[i,j] = matx[i,j]
    }  
  }
  return (copy_matx)
} 


#find the K nearest neighbors of each point
neighbor <- function(distance_matrix, K, train_set){ 
  neiB = matrix(0, nrow(distance_matrix), K)
  dis_copy = copy(distance_matrix)
  for(i in 1 : nrow(distance_matrix)){
    dis_copy[i,] = sort(dis_copy[i, ])
    for(j in 1 : K){
      neiB[i,j] = dis_copy[i,j]
    }
  }  
  
  for(i in 1 : nrow(distance_matrix)){
    for(j in 1 : K){
      temp = match(neiB[i,j], distance_matrix[i,])
      neiB[i,j] = train_set[temp, 3]
    }
  }
  return (neiB)
}


#classify each point based on the labels of its neighbors 
#this classify function is for the iris data, which has 3 categories"
classify <- function(neiB, K, n){
  classification = rep(0, nrow(neiB))
  for(i in 1 : nrow(neiB)){
    counts = rep(0, n)
    for(j in 1 : K){
      ncat = neiB[i,j]
      counts[ncat] = counts[ncat] + 1
    }
    
    
    #classify points based of counts, different for each data set
    max_cat = max(counts)
    temp = which(counts == max_cat)
    index = 0
    if(length(temp) > 1){
      rand = sample(1:length(temp), 1)
      index = temp[rand]
    }else{
      index = temp
    }
    
    classification[i] = index
  }
  return (classification)
}


#calculates the misclassification rate of our model
miss_rate <- function(classification, train_set){
  miss_total = 0
  for (i in 1: length(classification)){
    if (classification[i] != train_set[i, 3]){
      miss_total = miss_total + 1
    }
  }
  
  miss_Rate = miss_total / length(classification)
  return (miss_Rate)
}


#performs 10-fold cross-validation (N: number of folds)
cross_validation <- function(data_set, N, K, n){
  shuffle = data_set[sample.int(nrow(data_set)),]
  validation_size = nrow(shuffle) %/% N
  total = 0
  for (i in 1:N){
    validation_set = shuffle[((i-1) * validation_size + 1) : (i * validation_size),]
    front = shuffle[ 1 : ((i-1) * validation_size), ]
    if (i < 10){
      back = shuffle[ (i * validation_size + 1) : nrow(shuffle), ]
      train_set = rbind(front, back)
    }else{
      train_set = front
    }
    
    dist = distance(train_set[,1:2], validation_set[,1:2]) #find the distance matrix 
    neighbor_matrix = neighbor(dist, K, train_set)    #find the neighbor matrix
    classification = classify(neighbor_matrix, K, n)     #find the classification matrix
    misclassification_rate = miss_rate(classification, validation_set) #find the misclassification rate
    total = total +  misclassification_rate
  }
  avg = total / N
  return (avg)
  
}

read_input <- function(){
  print("Welcome to our K-Nearest Neighbors program!")
  print("We have 5 data sets you can run with this program! Type the number corresponding with the data set you wish to run.")
  print("1 - Iris data set")
  print("2 - Banknote Authentication")
  n = readline(prompt="Which do you want to load: ")
  return(as.integer(n))
}


main <- function(){
  input = read_input()
  if(input == 1){
    set.seed(20) #fix random initialization
    
    data = read.csv("Iris.csv")  #the data frame
    dataset = data.matrix(data)  #convert the data frame to a matrix
    features = dataset[,c(2,3)]  #extract the features
    labels = dataset[,6]         #extract the labels
    num_labels = 3               #number of categories
    K = 10                       #consider 3 neighbors
    train_set = vert_bind(features, labels) #bind the features and labels together
    
    dist = distance(train_set[,1:2], train_set[,1:2]) #find the distance matrix 
    neighbor_matrix = neighbor(dist, K, train_set)    #find the neighbor matrix
    classification = classify(neighbor_matrix, K, num_labels)     #find the classification matrix
    #print(classification)
    misclassification_rate = miss_rate(classification, train_set) #find the misclassification rate
    
    avg_missrates = vector(length = K) #create a vector of average misclassification rates (over K)
    for (i in 1:K){
      miss = cross_validation(train_set, 10, i, num_labels)
      avg_missrates[i] = miss
    }
    
    min = which.min(avg_missrates) #find the number of neighbors that gives the best performance
   
    plot(avg_missrates, type = "l", main = "10-Fold Cross Validation Average Misclassification Rates", xlab = "Number of Neighbors", ylab = "Misclassification Rate")
    
    plot(train_set[,1:2], pch = c(15,16,17)[as.numeric(labels)], main = "Ground Truth", xlab = "Sepal Length", ylab = "Sepal Width", col= c("red", "blue", "orange")[as.numeric(labels)])
    plot(train_set[,1:2], pch = c(15,16,17)[as.numeric(classification)], main = "Classification by KNN", xlab = "Sepal Length", ylab = "Sepal Width", col= c("red", "blue", "orange")[as.numeric(classification)])
  
  #Banknote Authentication
  }else if(input == 2){  
    set.seed(20) #fix random initialization
    
    data = read.csv("Iris.csv")  #the data frame
    dataset = data.matrix(data)  #convert the data frame to a matrix
    features = dataset[,c(2,3)]  #extract the features
    labels = dataset[,6]         #extract the labels
    num_labels = 3               #number of categories
    K = 10                       #consider 3 neighbors
    train_set = vert_bind(features, labels) #bind the features and labels together
    
    dist = distance(train_set[,1:2], train_set[,1:2]) #find the distance matrix 
    neighbor_matrix = neighbor(dist, K, train_set)    #find the neighbor matrix
    classification = classify(neighbor_matrix, K, num_labels)     #find the classification matrix
    #print(classification)
    misclassification_rate = miss_rate(classification, train_set) #find the misclassification rate
    
    avg_missrates = vector(length = K) #create a vector of average misclassification rates (over K)
    for (i in 1:K){
      miss = cross_validation(train_set, 10, i, num_labels)
      avg_missrates[i] = miss
    }
    
    min = which.min(avg_missrates) #find the number of neighbors that gives the best performance
    
    plot(avg_missrates, type = "l", main = "10-Fold Cross Validation Average Misclassification Rates", xlab = "Number of Neighbors", ylab = "Misclassification Rate")
    
    plot(train_set[,1:2], pch = c(15,16,17)[as.numeric(labels)], main = "Ground Truth", xlab = "Sepal Length", ylab = "Sepal Width", col= c("red", "blue", "orange")[as.numeric(labels)])
    plot(train_set[,1:2], pch = c(15,16,17)[as.numeric(classification)], main = "Classification by KNN", xlab = "Sepal Length", ylab = "Sepal Width", col= c("red", "blue", "orange")[as.numeric(classification)])
    
  }
}
main()


