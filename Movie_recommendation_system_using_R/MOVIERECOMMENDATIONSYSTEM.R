# Load necessary libraries
library(recommenderlab)  # For building recommendation systems
library(ggplot2)         # For data visualization
library(data.table)      # For efficient data manipulation
library(reshape2)        # For reshaping data
library(dplyr)           # For data manipulation using the %>% operator

# Load movie and rating data
movie_data <- read.csv("~/Desktop/movies-(1).csv", stringsAsFactors = FALSE)
rating_data <- read.csv("~/Desktop/ratings-(1).csv")
head(movie_data)
head(rating_data)

# Extract genres from movie_data into a data frame
movie_genre <- as.data.frame(movie_data$genres, stringsAsFactors = FALSE)

# Split genres into separate columns using '|' delimiter
movie_genre2 <- as.data.frame(tstrsplit(movie_genre[, 1], '[|]', type.convert = TRUE), 
                              stringsAsFactors = FALSE)

# Assign column names to the genre matrix
colnames(movie_genre2) <- c(1:10)

# Define list of genres
list_genre <- c("Action", "Adventure", "Animation", "Children", "Comedy", "Crime", 
                "Documentary", 
                "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", 
                "Romance", 
                "Sci-Fi", "Thriller", "War", "Western")

# Initialize genre matrix with zeros
genre_mat <- matrix(0, nrow(movie_data), length(list_genre))

# Assign column names to the genre matrix
colnames(genre_mat) <- list_genre

# Iterate through each movie and its genres
for (i in 1:nrow(movie_genre2)) {
  for (j in 1:ncol(movie_genre2)) {
    # Find the column index for the genre
    genre_col <- which(colnames(genre_mat) == movie_genre2[i, j])
    # Mark the corresponding genre as 1 in the genre matrix
    genre_mat[i, genre_col] <- 1
  }
}

# Convert genre matrix to data frame and ensure integer type
genre_mat <- as.data.frame(genre_mat, stringsAsFactors = FALSE)
genre_mat <- sapply(genre_mat, as.integer)

# Print structure of genre matrix
str(genre_mat)

# Combine movie_data, movie_id, and genre information into SearchMatrix
SearchMatrix <- cbind(movie_data[, 1:2], genre_mat)

# Display the first few rows of SearchMatrix
head(SearchMatrix)

# Create a histogram of rating distribution using ggplot2
ggplot(rating_data, aes(x = rating)) +  
  ggtitle("Rating Distribution") +  # Add plot title
  xlab("Rating") +  # Label for x-axis
  ylab("Count")  # Label for y-axis

# Calculate average rating and count of ratings for each movieId
top_rated_movies <- rating_data %>%
  group_by(movieId) %>%
  summarize(avg_rating = mean(rating), count = n()) %>%
  # Filter movies with more than 50 ratings
  filter(count > 50) %>%
  # Arrange movies by average rating in descending order
  arrange(desc(avg_rating)) %>%
  # Select top 10 movies by average rating
  top_n(10, wt = avg_rating)

# Merge top_rated_movies with movie_data to get movie titles
top_rated_movies <- merge(top_rated_movies, movie_data, by = "movieId")

# Create a bar plot of top rated movies
ggplot(top_rated_movies, aes(x = reorder(title, avg_rating), y = avg_rating)) +  
  geom_bar(stat = "identity", fill = "lightgreen", color = "black") +  
  coord_flip() +  # Flip coordinates to make horizontal bar plot
  ggtitle("Top 10 Rated Movies") +  # Add plot title
  xlab("Movie Title") +  # Label for x-axis
  ylab("Average Rating")  # Label for y-axis

ratingMatrix <- dcast(rating_data, userId ~ movieId, value.var = "rating", na.rm = FALSE)
ratingMatrix <- as.matrix(ratingMatrix[,-1])
ratingMatrix <- as(ratingMatrix, "realRatingMatrix")
ratingMatrix

# Build Item-Based Collaborative Filtering (IBCF) model using ratingMatrix data
recommen_model <- Recommender(data = ratingMatrix, method = "IBCF", 
                              parameter = list(k = 30))

# Get model information
model_info <- getModel(recommen_model)

# Display heatmap of similarity matrix for the first 20 rows and columns
image(model_info$sim[1:20, 1:20], main = "Heatmap of the first rows and columns")

# Build IBCF model
recommen_model <- Recommender(data = ratingMatrix, method = "IBCF", 
                              parameter = list(k = 30))
recommen_model

# Inspect model
model_info <- getModel(recommen_model)
class(model_info$sim)
dim(model_info$sim)

# Heatmap of similarities
top_items <- 20
image(model_info$sim[1:top_items, 1:top_items], 
      main = "Heatmap of the first rows and columns")


# Set seed for reproducibility
set.seed(123)

# Sample data for training and testing
sampled_data <- sample(x = c(TRUE, FALSE), size = nrow(ratingMatrix), 
                       replace = TRUE, prob = c(0.8, 0.2))

# Split ratingMatrix into training and testing data
training_data <- ratingMatrix[sampled_data, ]
testing_data <- ratingMatrix[!sampled_data, ]

# Define the number of top recommendations to predict
top_recommendations <- 10 

# Predict recommendations for testing data using the recommen_model
predicted_recommendations <- predict(object = recommen_model, 
                                     newdata = testing_data, n = top_recommendations)

# Extract recommendations for the first user in the testing set
user1_recommendations <- predicted_recommendations@items[[1]]
user1_movies <- predicted_recommendations@itemLabels[user1_recommendations]

# Retrieve movie titles for the recommended movies
user1_movie_titles <- sapply(user1_movies, function(x) as.character(subset(movie_data, 
                                                                           movieId == x)$title))

# Print recommended movie titles for the first user
user1_movie_titles

# Create an evaluation scheme with given parameters
scheme <- evaluationScheme(ratingMatrix, method = "split", train = 0.8, 
                           given = 15, goodRating = 4)

# Define the model using the evaluation scheme
model <- Recommender(getData(scheme,"train"), method = "IBCF", parameter = list(k = 30))

# Predict ratings for known data in the evaluation scheme
pred <- predict(model, getData(scheme, "known"), type = "ratings")

# Calculate prediction accuracy using unknown data in the scheme
error <- calcPredictionAccuracy(pred, getData(scheme, "unknown"))

# Print prediction accuracy error
error
