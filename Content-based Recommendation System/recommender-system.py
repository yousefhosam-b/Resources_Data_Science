# Import required libraries
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

'''
# For downloading the datasets
!wget -O moviedataset.zip https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%205/data/moviedataset.zip
print('unziping ...')
!unzip -o -j moviedataset.zip 
'''

# Read the data
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

def RecommenderSystem(inputMovies):
    # Create year column and remove year from title
    movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
    movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
    movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
    movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
    
    # Split the genres
    movies_df['genres'] = movies_df.genres.str.split('|')
    
    # One hot encoding  
    moviesWithGenres_df = movies_df.copy()
    for index, row in movies_df.iterrows():
        for genre in row['genres']:
            moviesWithGenres_df.at[index, genre] = 1
    # Filling nan values with 0
    moviesWithGenres_df = moviesWithGenres_df.fillna(0)
    
    # Getting the movie id
    inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
    inputMovies = pd.merge(inputId, inputMovies)
    inputMovies = inputMovies.drop('genres', 1).drop('year', 1)
    
    # Filter movies from the input
    userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
    
    # Get the actual genre table
    userMovies = userMovies.reset_index(drop=True)
    userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
    
    # Creating the user profile
    userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
    
    # Creating the genre table
    genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
    genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
    
    # Get the recommendations 
    recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
    recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
    recommendations = movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())]
    return recommendations
    
    
# User movies
userInput = [
            {'title': 'Air Force One', 'rating': 5},
            {'title': 'Taken 2', 'rating': 5},
            {'title': 'Enemy at the Gates', 'rating': 5},
            {'title': 'Jurassic Park', 'rating': 5},
            {'title': 'Wanted', 'rating': 5}
         ] 
inputMovies = pd.DataFrame(userInput)

# Call the function
recommendations = RecommenderSystem(inputMovies)
