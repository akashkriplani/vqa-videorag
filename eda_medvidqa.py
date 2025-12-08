"""
eda_medvidqa.py - Exploratory Data Analysis for MedVidQA Dataset

This module provides comprehensive exploratory data analysis (EDA) for the MedVidQA cleaned
train, test, and validation datasets. It generates a variety of visualizations and statistical
summaries to understand the dataset characteristics and identify patterns.

The analysis is divided into three main categories:
1. Univariate Analysis: Examines individual variables in isolation
2. Bivariate Analysis: Explores relationships between pairs of variables
3. Multivariate Analysis: Analyzes interactions between multiple variables

Key Features:
-------------
- Video Analysis: Distribution of questions across videos, identification of top videos
- Temporal Analysis: Distribution of answer timestamps within videos
- Question Analysis: Length distribution, duplicate detection
- Correlation Analysis: Relationship between video length and question count
- Missing Data Detection: Identifies incomplete or missing data fields

Visualizations Generated:
------------------------
- Bar plots: Top 20 videos by question count
- Histograms: Video lengths, question lengths, questions per video
- Heatmaps: Missing data patterns, 2D distribution of questions vs video length
- Scatter plots: Video length vs number of questions correlation
- Temporal distributions: Answer start and end times

Output:
-------
All visualizations are saved as PNG files in the 'EDA/' directory with descriptive filenames
indicating the analysis type and dataset split (train/test/validation).

Directory Structure:
-------------------
- Input: MedVidQA_cleaned/ (cleaned JSON files for train, test, val)
- Output: EDA/ (generated visualization PNG files)

Dependencies:
------------
- pandas: Data manipulation and analysis
- matplotlib: Base plotting library
- seaborn: Statistical visualization
- json: Loading dataset files

Usage:
------
Run this script directly to generate all EDA visualizations:
    $ python eda_medvidqa.py

The script will process all three dataset splits (train, test, validation) and generate
comprehensive visualizations and statistics for each.
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

CLEANED_DIR = "MedVidQA_cleaned"
TRAIN_PATH = os.path.join(CLEANED_DIR, "train_cleaned.json")
TEST_PATH = os.path.join(CLEANED_DIR, "test_cleaned.json")
VAL_PATH = os.path.join(CLEANED_DIR, "val_cleaned.json")

EDA_DIR = "EDA"
os.makedirs(EDA_DIR, exist_ok=True)

def load_df(path):
    """
    Load a cleaned MedVidQA dataset from JSON file into a pandas DataFrame.

    Args:
        path (str): Path to the JSON file containing the cleaned dataset.

    Returns:
        pd.DataFrame: DataFrame containing the loaded dataset with columns such as
                      video_id, question, answer, video_length, answer_start_second, etc.

    Raises:
        FileNotFoundError: If the specified path does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)

train_df = load_df(TRAIN_PATH)
test_df = load_df(TEST_PATH)
val_df = load_df(VAL_PATH)

# --- Visualization Functions ---
def plot_top_videos_by_questions(df, set_name):
    """
    Generate a bar plot showing the top 20 videos by number of questions.

    Creates a horizontal bar plot displaying which videos have the most questions associated
    with them, helping identify videos with rich Q&A content. Also prints summary statistics
    including total questions, unique videos, and the top 10 videos by question count.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset with 'video_id' column.
        set_name (str): Name of the dataset split (e.g., 'Train', 'Test', 'Validation').

    Output:
        - Saves a PNG file: 'EDA/eda_questions_per_video_{set_name}.png'
        - Prints summary statistics to console

    Returns:
        None
    """
    plt.figure(figsize=(12,6))
    q_per_video = df['video_id'].value_counts()
    sns.barplot(x=q_per_video.index[:20], y=q_per_video.values[:20])
    plt.xticks(rotation=90)
    plt.title(f'Top 20 Videos by Number of Questions ({set_name} Set)')
    plt.xlabel('Video ID')
    plt.ylabel('Number of Questions')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, f'eda_questions_per_video_{set_name.lower()}.png'))
    plt.close()
    print(f'\n{set_name} set:')
    print('Total questions:', len(df))
    print('Unique videos:', df["video_id"].nunique())
    print('Questions per video (top 10):')
    print(q_per_video.head(10))


def plot_video_length_distribution(df, set_name):
    """
    Generate a histogram showing the distribution of video lengths.

    Creates a histogram of video durations (in seconds) using only unique videos to avoid
    counting the same video multiple times. Provides insights into the typical length of
    videos in the dataset and prints descriptive statistics.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset with 'video_id' and 'video_length' columns.
        set_name (str): Name of the dataset split (e.g., 'Train', 'Test', 'Validation').

    Output:
        - Saves a PNG file: 'EDA/eda_video_length_distribution_{set_name}.png'
        - Prints descriptive statistics (mean, std, min, max, quartiles)

    Returns:
        None

    Note:
        Only processes data if 'video_length' column is present in the DataFrame.
    """
    if 'video_length' in df.columns:
        unique_videos = df.drop_duplicates(subset=['video_id'])
        plt.figure(figsize=(8,5))
        sns.histplot(unique_videos['video_length'], bins=30)
        plt.title(f'Distribution of Video Lengths (seconds) - {set_name} Set (Unique Videos)')
        plt.xlabel('Video Length (s)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, f'eda_video_length_distribution_{set_name.lower()}.png'))
        plt.close()
        print('Video length stats (unique videos):')
        print(unique_videos['video_length'].describe())


def print_summary_stats(df, set_name):
    """
    Print comprehensive summary statistics for the dataset.

    Displays key metrics including total question count, unique video count, top videos
    by question count, and video length statistics if available.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        set_name (str): Name of the dataset split (e.g., 'Train', 'Test', 'Validation').

    Output:
        Prints to console:
        - Total number of questions
        - Number of unique videos
        - Top 10 videos by question count
        - Video length statistics (if 'video_length' column exists)

    Returns:
        None
    """
    print(f'\n{set_name} set summary:')
    print('Total questions:', len(df))
    print('Unique videos:', df["video_id"].nunique())
    q_per_video = df['video_id'].value_counts()
    print('Questions per video (top 10):')
    print(q_per_video.head(10))
    if 'video_length' in df.columns:
        print('Video length stats:')
        print(df['video_length'].describe())


def plot_question_length_distribution(df, set_name):
    """
    Generate a histogram showing the distribution of question lengths in words.

    Analyzes and visualizes how long questions typically are by counting the number of words
    in each question. This helps understand question complexity and verbosity patterns.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset with 'question' column.
        set_name (str): Name of the dataset split (e.g., 'Train', 'Test', 'Validation').

    Output:
        - Saves a PNG file: 'EDA/eda_question_length_distribution_{set_name}.png'
        - Prints descriptive statistics for question lengths

    Returns:
        None

    Note:
        - Only processes data if 'question' column is present in the DataFrame.
        - Question length is measured in number of words (whitespace-separated tokens).
    """
    if 'question' in df.columns:
        lengths = df['question'].apply(lambda x: len(str(x).split()))
        plt.figure(figsize=(8,5))
        sns.histplot(lengths, bins=30)
        plt.title(f'Distribution of Question Lengths ({set_name} Set)')
        plt.xlabel('Question Length (words)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, f'eda_question_length_distribution_{set_name.lower()}.png'))
        plt.close()
        print(f'Question length stats ({set_name}):')
        print(lengths.describe())


def plot_questions_per_video_hist(df, set_name):
    """
    Generate a histogram showing the distribution of questions per video.

    Creates a histogram that shows how many questions each video typically has, revealing
    whether questions are distributed evenly or concentrated on certain videos.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset with 'video_id' column.
        set_name (str): Name of the dataset split (e.g., 'Train', 'Test', 'Validation').

    Output:
        - Saves a PNG file: 'EDA/eda_questions_per_video_hist_{set_name}.png'
        - Prints descriptive statistics for questions per video distribution

    Returns:
        None

    Note:
        This visualization helps identify if some videos are over-represented in the dataset.
    """
    q_per_video = df['video_id'].value_counts()
    plt.figure(figsize=(8,5))
    sns.histplot(q_per_video, bins=30)
    plt.title(f'Questions per Video Distribution ({set_name} Set)')
    plt.xlabel('Questions per Video')
    plt.ylabel('Number of Videos')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, f'eda_questions_per_video_hist_{set_name.lower()}.png'))
    plt.close()
    print(f'Questions per video stats ({set_name}):')
    print(q_per_video.describe())


def plot_missing_data(df, set_name):
    """
    Generate a heatmap visualization of missing data patterns in the dataset.

    Creates a binary heatmap where missing values are highlighted, making it easy to identify
    columns with missing data and patterns in missingness across the dataset.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        set_name (str): Name of the dataset split (e.g., 'Train', 'Test', 'Validation').

    Output:
        - Saves a PNG file: 'EDA/eda_missing_data_{set_name}.png'
        - Prints count of missing values per column

    Returns:
        None

    Note:
        White areas in the heatmap indicate missing values, colored areas indicate present values.
    """
    plt.figure(figsize=(10,5))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title(f'Missing Data Heatmap ({set_name} Set)')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, f'eda_missing_data_{set_name.lower()}.png'))
    plt.close()
    print(f'Missing data summary ({set_name}):')
    print(df.isnull().sum())


def plot_temporal_distribution(df, set_name):
    """
    Generate overlapping histograms showing temporal distribution of answer timestamps.

    Visualizes when answers occur within videos by plotting the distribution of answer start
    and end times. This helps understand if answers are concentrated at certain time points
    in the videos (e.g., beginning, middle, or end).

    Args:
        df (pd.DataFrame): DataFrame containing the dataset with 'answer_start_second' and
                          'answer_end_second' columns.
        set_name (str): Name of the dataset split (e.g., 'Train', 'Test', 'Validation').

    Output:
        - Saves a PNG file: 'EDA/eda_temporal_distribution_{set_name}.png'
        - Prints descriptive statistics for answer start and end times

    Returns:
        None

    Note:
        - Only processes data if 'answer_start_second' and 'answer_end_second' columns exist.
        - If string-based 'answer_start' and 'answer_end' columns exist instead, prints sample values.
    """
    if 'answer_start_second' in df.columns and 'answer_end_second' in df.columns:
        plt.figure(figsize=(8,5))
        sns.histplot(df['answer_start_second'], bins=30, color='blue', label='Start')
        sns.histplot(df['answer_end_second'], bins=30, color='orange', label='End', alpha=0.6)
        plt.title(f'Temporal Distribution of Answers ({set_name} Set)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Count')
        plt.legend(['Answer Start', 'Answer End'])
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, f'eda_temporal_distribution_{set_name.lower()}.png'))
        plt.close()
        print(f'Answer start/end stats ({set_name}):')
        print('Start:', df['answer_start_second'].describe())
        print('End:', df['answer_end_second'].describe())
    elif 'answer_start' in df.columns and 'answer_end' in df.columns:
        print(f'Answer start/end columns present as strings in {set_name} set.')
        print('Sample start:', df['answer_start'].head())
        print('Sample end:', df['answer_end'].head())


def plot_correlation_video_length_questions(df, set_name):
    """
    Generate a scatter plot showing the correlation between video length and question count.

    Creates a scatter plot to visualize the relationship between how long a video is and how
    many questions are associated with it. Computes and displays the correlation coefficient
    to quantify the strength of this relationship.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset with 'video_id' and 'video_length' columns.
        set_name (str): Name of the dataset split (e.g., 'Train', 'Test', 'Validation').

    Output:
        - Saves a PNG file: 'EDA/eda_video_length_vs_questions_{set_name}.png'
        - Prints correlation matrix between video length and number of questions

    Returns:
        None

    Note:
        - Only processes data if 'video_length' column is present.
        - Positive correlation suggests longer videos tend to have more questions.
    """
    if 'video_length' in df.columns:
        q_per_video = df.groupby('video_id').size().reset_index(name='num_questions')
        video_lengths = df.drop_duplicates('video_id')[['video_id', 'video_length']]
        merged = pd.merge(q_per_video, video_lengths, on='video_id')
        plt.figure(figsize=(8,5))
        sns.scatterplot(x='video_length', y='num_questions', data=merged)
        plt.title(f'Video Length vs. Number of Questions ({set_name} Set)')
        plt.xlabel('Video Length (s)')
        plt.ylabel('Number of Questions')
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, f'eda_video_length_vs_questions_{set_name.lower()}.png'))
        plt.close()
        print(f'Correlation (video length vs. questions) ({set_name}):')
        print(merged[['video_length', 'num_questions']].corr())


def plot_duplicate_questions(df, set_name):
    """
    Generate a bar plot showing the proportion of unique vs. duplicate questions.

    Identifies and visualizes duplicate questions within the same video, helping assess
    data quality and identify potential issues with question redundancy.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset with 'question' and 'video_id' columns.
        set_name (str): Name of the dataset split (e.g., 'Train', 'Test', 'Validation').

    Output:
        - Saves a PNG file: 'EDA/eda_duplicate_questions_{set_name}.png'
        - Prints count and percentage of duplicate questions

    Returns:
        None

    Note:
        - Only processes data if 'question' column is present.
        - Duplicates are identified by matching both question text and video_id.
        - High duplication rates may indicate data quality issues.
    """
    if 'question' in df.columns:
        dupes = df.duplicated(subset=['question', 'video_id'], keep=False)
        num_dupes = dupes.sum()
        total = len(df)
        plt.figure(figsize=(6,4))
        sns.barplot(x=['Unique', 'Duplicate'], y=[total-num_dupes, num_dupes])
        plt.title(f'Unique vs. Duplicate Questions ({set_name} Set)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, f'eda_duplicate_questions_{set_name.lower()}.png'))
        plt.close()
        print(f'Duplicate questions ({set_name}): {num_dupes} / {total} ({num_dupes/total:.2%})')


def plot_heatmap_questions_vs_video_length(df, set_name):
    """
    Generate a 2D histogram heatmap of questions per video vs. video length.

    Creates a two-dimensional heatmap that shows the joint distribution of video length and
    question count, revealing patterns in how these two variables interact. Darker/brighter
    areas indicate higher concentration of videos.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset with 'video_id' and 'video_length' columns.
        set_name (str): Name of the dataset split (e.g., 'Train', 'Test', 'Validation').

    Output:
        Saves a PNG file: 'EDA/eda_heatmap_questions_vs_video_length_{set_name}.png'

    Returns:
        None

    Note:
        - Only processes data if 'video_length' column is present.
        - Uses viridis colormap with probability threshold for better visualization.
        - Helps identify if certain video length ranges tend to have specific question counts.
    """
    if 'video_length' in df.columns:
        q_per_video = df['video_id'].value_counts().reset_index()
        q_per_video.columns = ['video_id', 'num_questions']
        video_lengths = df.drop_duplicates('video_id')[['video_id', 'video_length']]
        merged = pd.merge(q_per_video, video_lengths, on='video_id')
        plt.figure(figsize=(8,6))
        sns.histplot(data=merged, x='video_length', y='num_questions', bins=30, pthresh=.1, cmap='viridis')
        plt.title(f'Heatmap: Questions per Video vs. Video Length ({set_name} Set)')
        plt.xlabel('Video Length (s)')
        plt.ylabel('Questions per Video')
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, f'eda_heatmap_questions_vs_video_length_{set_name.lower()}.png'))
        plt.close()

# --- Univariate Analysis ---
def univariate_analysis(df, set_name):
    """
    Perform comprehensive univariate analysis on the dataset.

    Executes all single-variable analyses to understand the distribution and characteristics
    of individual features in isolation. This includes analyzing video distributions, question
    characteristics, and basic dataset statistics.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset to analyze.
        set_name (str): Name of the dataset split (e.g., 'Train', 'Test', 'Validation').

    Output:
        Generates multiple visualizations and prints summary statistics:
        - Top videos by question count
        - Video length distribution
        - Question length distribution
        - Questions per video histogram
        - Summary statistics

    Returns:
        None
    """
    plot_top_videos_by_questions(df, set_name)
    plot_video_length_distribution(df, set_name)
    plot_question_length_distribution(df, set_name)
    plot_questions_per_video_hist(df, set_name)
    print_summary_stats(df, set_name)

# --- Bivariate Analysis ---
def bivariate_analysis(df, set_name):
    """
    Perform comprehensive bivariate analysis on the dataset.

    Examines relationships between pairs of variables to understand how different features
    interact with each other. This includes temporal patterns, correlations, and data quality
    issues like duplicates.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset to analyze.
        set_name (str): Name of the dataset split (e.g., 'Train', 'Test', 'Validation').

    Output:
        Generates multiple visualizations and prints statistics:
        - Temporal distribution of answers within videos
        - Correlation between video length and question count
        - Duplicate question analysis

    Returns:
        None
    """
    plot_temporal_distribution(df, set_name)
    plot_correlation_video_length_questions(df, set_name)
    plot_duplicate_questions(df, set_name)

# --- Multivariate Analysis ---
def multivariate_analysis(df, set_name):
    """
    Perform comprehensive multivariate analysis on the dataset.

    Analyzes complex interactions between multiple variables simultaneously to reveal patterns
    that may not be apparent in univariate or bivariate analyses. Currently focuses on the
    joint distribution of video length and question count.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset to analyze.
        set_name (str): Name of the dataset split (e.g., 'Train', 'Test', 'Validation').

    Output:
        Generates 2D heatmap visualization showing the relationship between multiple variables:
        - Questions per video vs. video length heatmap

    Returns:
        None
    """
    plot_heatmap_questions_vs_video_length(df, set_name)

def main():
    """
    Main entry point for the EDA script.

    Orchestrates the complete exploratory data analysis workflow by loading all three dataset
    splits (train, test, validation) and performing univariate, bivariate, and multivariate
    analyses on each. Results are saved to the EDA directory.

    Workflow:
        1. Load cleaned datasets (train, test, validation)
        2. For each dataset split:
           a. Perform univariate analysis
           b. Perform bivariate analysis
           c. Perform multivariate analysis
        3. Save all visualizations to EDA/ directory

    Args:
        None

    Returns:
        None

    Note:
        This function assumes that cleaned JSON files exist in the MedVidQA_cleaned/ directory.
    """
    # --- Run EDA for all sets ---
    for df, name in zip([train_df, test_df, val_df], ['Train', 'Test', 'Validation']):
        print(f"\n{'='*20} {name} Set: Univariate Analysis {'='*20}")
        univariate_analysis(df, name)
        print(f"\n{'='*20} {name} Set: Bivariate Analysis {'='*20}")
        bivariate_analysis(df, name)
        print(f"\n{'='*20} {name} Set: Multivariate Analysis {'='*20}")
        multivariate_analysis(df, name)

if __name__ == "__main__":
    main()