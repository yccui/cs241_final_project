" ------------------------------------------------------
  K-Means Algorithm
  
  @author Cui, Elliott, Lai, Steuer 
  
  This program is our implementation of the k-means
  algorithm for clustering. 
  ------------------------------------------------------"

#bind a matrix and a vector vertically
vert_bind <- function(matrix1, vector){ 
  result = matrix(0, nrow(matrix1), ncol(matrix1) + 1) #create a matrix with an extra column
  for(i in 1 : nrow(matrix1)){ #transfer data from matrix1
    for (j in 1 : ncol(matrix1)){
      result[i, j] = matrix1[i,j]
    }
  }
  for(i in 1 : length(vector)){ #transfer data from vector
    result[i, ncol(result)] = vector[i]
  }
  return (result)
}


#randomly initialize each pointer to a cluster 
initialize <- function(features, K){
  rand_init = vector(length = nrow(features)) #create a vector of length "number of points"
  for(i in 1:nrow(features)){ 
    rand_temp = sample(1:K, 1) 
    rand_init[i] = rand_temp #assign each point a randome label (1 - K)
  }
  
  init = vert_bind(features, rand_init) #bind features and labels into one matrix
  return(init)
}


#extract clusters from a matrix
extract <- function(matrix, label){
  count = 0
  end = ncol(matrix) #index of the last column of matrix
  
  for (i in 1:nrow(matrix)){
    if (matrix[i, end] == label){
      count = count + 1
    }
  }
  
  cluster = matrix(0, count, ncol(matrix))
  index = 1
  for(i in 1:nrow(matrix)){
    if(matrix[i, end] == label){
      cluster[index, ] = matrix[i,]
      index = index + 1
    }
  }
  return(cluster)
}


#find the center of a cluster
find_center <- function(matrix, num_attributes){
  sum = vector(length = num_attributes)
  for(i in 1:nrow(matrix)){
    sum = sum + matrix[i, 1:num_attributes] #add all the points componentwise
  }
  sum = sum / nrow(matrix) #find the average
  col_labels = num_attributes + 1
  center = c(sum, matrix[1, col_labels]) #bind the average feature with its label
  return(center)
}

#compute distances between each point in the features and the centers
distance <- function(features, centers){ #needs generalization!!!!
  dis_matrix = matrix(0, nrow(features), nrow(centers))
  print(ncol(features))
  print(ncol(centers))
  for(i in 1 : nrow(features)){
    n = ncol(features) - 1     #delete the column of labels
    l_1 = vector(length = n)
    for (k in 1: n) {
      l_1[k] = features[i,k];
    }
    #x1 = features[i,1]
    #y1 = features[i,2]
    for(j in 1 : nrow(centers)){
      n = ncol(centers) - 1      #delete the column of labels
      l_2 = vector(length = n)
      for (t in 1: n) {
        l_2[t] = centers[j,t];
      }
      #x2 = centers[j,1]
      #y2 = centers[j,2]
      #d = sqrt((x2-x1)^2+(y2-y1)^2)
      d = dist(rbind(l_1, l_2))
      dis_matrix[i, j] = d
    }
  }
  return (dis_matrix)
}

#reassign points to closest center
reassign <- function(init_features, centers){
  dis = distance(init_features, centers)
  for(i in 1:nrow(init_features)){
    min = which.min(dis[i,]) #find the label of its nearest center
    init_features[i,3] = min #label that point with the label of its nearest center
  }
  return(init_features)
}


#test to see if reassigning makes any changes
convergence <- function(previous, current, col_labels){
  for( i in 1:nrow(previous)){
    if(current[i, col_labels] != previous[i, col_labels]){ #if the labels of any point are different in the previous iteration from the current iteration
      return(FALSE)                    #it does not converge; return false
    }
  }
  return(TRUE) #else, return true
}


#metric on how well the model performs
metric <- function(final_it, cluster_centers){
  D = 0
  for(i in 1:nrow(final_it)){
    cluster = final_it[i,3]
    x1 = final_it[i,1]
    y1 = final_it[i,2]
    
    x2 = cluster_centers[cluster, 1]
    y2 = cluster_centers[cluster, 2]
    
    d = (x2-x1)^2 + (y2-y1)^2
    
    D = D + d
  }
  return (D)
}


main <- function(){
  set.seed(20) #fix random initialization
  
  data = read.csv("iris.csv") #obtain the data frame
  dataset = data.matrix(data) #convert the data frame to a matrix
  
  start_col_of_features = 2                #the index of the first column of feaures is always 2 (index 1 is the row numbers)
  last_col_of_features = ncol(dataset) - 1 #the index of the last column of features
  col_of_labels = ncol(dataset)            #the index of the column of labels
  
  features = dataset[,c(start_col_of_features, last_col_of_features)] #extract the features
  labels = dataset[ ,col_of_labels]               #extract the labels
  
  K = 3   #the number of clusters
  num_attributes = 2   #the number of attributes to be considered
  col_labels = num_attributes + 1  #the index of the column of labels once combined
  
  init_features = initialize(features, K)   #randomly initialize labels (K clusters)

  cluster_centers = matrix(0, K, ncol(init_features)) #create a matrix to store the K cluster centers

  for (i in 1:K){
    temp = extract(init_features, i)
    cluster_centers[i, ] = find_center(temp, num_attributes)
  }
  
  first_iteration = reassign(init_features, cluster_centers) #result from the 1st iteration
  
  starter = convergence(init_features, first_iteration, col_labels) #initial convergence check
  
  while (!starter){
    cluster_centers = matrix(0, K, ncol(first_iteration))
    for (i in 1:K){
      temp = extract(first_iteration, i)
      cluster_centers[i, ] = find_center(temp, num_attributes)
    }
    
    next_iteration = reassign(first_iteration, cluster_centers) #result from the next iteration
    
    starter = convergence(first_iteration, next_iteration, col_labels) #update convergence 
    
    first_iteration = next_iteration #update starting configuration
  
  }
  
  final_iteration = first_iteration #obtain result from the final iteration

  labels_kmeans = first_iteration[ , col_labels] #obtain labels of the final iteration

  D = metric(final_iteration, cluster_centers) #computer the metric
  
  print(D)
  
  shapes = vector(length = K) #for plotting: make an array of shapes
  for (i in 15:(15 + K - 1)){
    shapes[i- 14] = i
  }
  
  colors = c(1:K) #for plotting: make an array of colors

  plot(features, pch = shapes[as.numeric(labels)], main = "Ground Truth", xlab = "Sepal Length", ylab = "Sepal Width", col = colors[as.numeric(labels)])
  
  plot(features, pch = shapes[as.numeric(labels_kmeans)], main = "Clustering by K-Means", xlab = "Sepal Length", ylab = "Sepal Width", col = colors[as.numeric(labels_kmeans)])
  
}

main()