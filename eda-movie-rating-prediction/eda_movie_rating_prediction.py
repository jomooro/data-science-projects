import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def read_dataset(file_path):
    """
    Read the IMDb dataset from a CSV file.

    Args:
    file_path (str): Path to the CSV file containing the dataset.

    Returns:
    pandas.DataFrame: DataFrame containing the dataset.
    """
    movies_df = pd.read_csv(file_path)
    return movies_df

def clean_data(movies_df):
    """
    Clean the dataset by removing rows with missing 'Rating' values,
    cleaning 'Votes' and 'Year' columns, and filling missing values in 'Duration' column.

    Args:
    movies_df (pandas.DataFrame): DataFrame containing the dataset.

    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """
    movies_df.dropna(subset=['Rating'], inplace=True)
    movies_df['Votes'] = movies_df['Votes'].str.replace(',', '').astype(int)
    movies_df['Year'] = movies_df['Year'].str.strip('()').astype(int)
    movies_df['Duration'] = movies_df['Duration'].str.replace(' min', '').astype(float)
    mask = movies_df['Duration'].isnull()
    random_values = np.random.randint(90, 181, size=mask.sum())
    movies_df.loc[mask, 'Duration'] = random_values
    return movies_df

def plot_distribution(movies_df):
    """
    Plot the distribution of movie durations.

    Args:
    movies_df (pandas.DataFrame): DataFrame containing the dataset.
    """
    sns.histplot(data=movies_df['Duration'].fillna(movies_df['Duration'].mean()), bins=20, kde=True)
    plt.xlabel('Duration in Minutes')
    plt.title('Distribution of movie durations')
    plt.show()

def select_top_movies(movies_df):
    """
    Select the top 10 movies based on ratings.

    Args:
    movies_df (pandas.DataFrame): DataFrame containing the dataset.

    Returns:
    pandas.DataFrame: DataFrame containing the top 10 movies.
    """
    top_10_movies = movies_df.sort_values(by='Rating', ascending=False).head(10)
    return top_10_movies

def plot_average_ratings_by_year(movies_df):
    """
    Plot average movie ratings by year.

    Args:
    movies_df (pandas.DataFrame): DataFrame containing the dataset.
    """
    highest_rated_by_year = movies_df.loc[movies_df.groupby('Year')['Rating'].idxmax()]
    sns.set_style('darkgrid')
    highest_rated_by_year.groupby('Year')['Rating'].mean().plot(figsize=(15, 5))
    plt.xlabel('Year')
    plt.ylabel('Average Rating')
    plt.title('Average movie ratings by year')
    plt.xticks(np.arange(1917,2023,5))
    plt.xlim(1917,2023)
    plt.show()

def plot_duration_and_rating_by_year(movies_df):
    """
    Plot average duration and rating by year.

    Args:
    movies_df (pandas.DataFrame): DataFrame containing the dataset.
    """
    # Compute highest rated movie by year
    highest_rated_by_year = movies_df.loc[movies_df.groupby('Year')['Rating'].idxmax()]

    # Plot average duration and rating by year
    fig, ax1 = plt.subplots(figsize=(15, 5))
    sns.lineplot(data=highest_rated_by_year, x='Year', y='Duration', ax=ax1, label='Average Duration')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Average Duration')
    plt.title('Average duration and rating by year')
    ax1.set_xlim(1917, 2023)
    ax1.set_xticks(np.arange(1917, 2023, 5))
    ax2 = ax1.twinx()
    sns.lineplot(data=highest_rated_by_year, x='Year', y='Rating', ax=ax2, color='red', label='Average Rating')
    ax2.set_ylabel('Average Rating')

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.show()

if __name__ == "__main__":

    movies_df = read_dataset('IMDb_Movies_India.csv')
    movies_df = clean_data(movies_df)
    plot_distribution(movies_df)
    top_10_movies = select_top_movies(movies_df)
    print(top_10_movies)
    plot_average_ratings_by_year(movies_df)
    plot_duration_and_rating_by_year(movies_df)
