"""
eda_medvidqa.py
Exploratory Data Analysis (EDA) for MedVidQA cleaned train, test, and validation datasets.

This script loads the cleaned train, test, and validation datasets and generates visualizations and summary statistics:
- Top 20 videos by number of questions
- Distribution of video lengths (unique videos)
- Distribution of question lengths
- Histogram of questions per video
- Missing data heatmap
- Temporal distribution of answers (using answer_start_second/answer_end_second)
- Correlation: video length vs. number of questions
- Duplicate questions analysis
- 2D heatmap: questions per video vs. video length
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

# Load cleaned train, test, and validation datasets
def load_df(path):
    with open(path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)

train_df = load_df(TRAIN_PATH)
test_df = load_df(TEST_PATH)
val_df = load_df(VAL_PATH)

# --- Visualization Functions ---
def plot_top_videos_by_questions(df, set_name):
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
    plt.figure(figsize=(10,5))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title(f'Missing Data Heatmap ({set_name} Set)')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, f'eda_missing_data_{set_name.lower()}.png'))
    plt.close()
    print(f'Missing data summary ({set_name}):')
    print(df.isnull().sum())


def plot_temporal_distribution(df, set_name):
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
    plot_top_videos_by_questions(df, set_name)
    plot_video_length_distribution(df, set_name)
    plot_question_length_distribution(df, set_name)
    plot_questions_per_video_hist(df, set_name)
    print_summary_stats(df, set_name)

# --- Bivariate Analysis ---
def bivariate_analysis(df, set_name):
    plot_temporal_distribution(df, set_name)
    plot_correlation_video_length_questions(df, set_name)
    plot_duplicate_questions(df, set_name)

# --- Multivariate Analysis ---
def multivariate_analysis(df, set_name):
    plot_heatmap_questions_vs_video_length(df, set_name)

# --- Run EDA for all sets ---
for df, name in zip([train_df, test_df, val_df], ['Train', 'Test', 'Validation']):
    print(f"\n{'='*20} {name} Set: Univariate Analysis {'='*20}")
    univariate_analysis(df, name)
    print(f"\n{'='*20} {name} Set: Bivariate Analysis {'='*20}")
    bivariate_analysis(df, name)
    print(f"\n{'='*20} {name} Set: Multivariate Analysis {'='*20}")
    multivariate_analysis(df, name)
