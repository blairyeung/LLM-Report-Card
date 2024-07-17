import pandas as pd
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_jitter_plots_multiple(file_names):
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)

    num_files = len(file_names)
    if num_files <= 2:
        num_rows = 1
        num_cols = num_files
    elif num_files <= 4:
        num_rows = 2
        num_cols = 2
    elif num_files <= 6:
        num_rows = 2
        num_cols = 3
    else:
        num_rows = 3
        num_cols = 3

    x_lim = (0.45, 1.05)
    y_lim = (0, 1.05)
    # y_lim = (5, 10)
    point_size = 150

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(9 * num_cols, 6 * num_rows))

    for i, file_name in enumerate(file_names):
        df = pd.read_csv(f'exp_rslt/{file_name}')
        merged = df

        row = i // num_cols
        col = i % num_cols

        if num_files > 1:
            if num_rows > 1:
                ax = axs[row, col]
            else:
                ax = axs[col]
            
        else:
            ax = axs
        # all jitter around 0.5
        # sns.stripplot(ax=ax, data=merged, x='Oracle_Accuracy', y='Predictive_Accuracy', hue='Topic', jitter=0.1, s=10)
        # violin
        sns.violinplot(ax=ax, data=merged, x='Oracle_Accuracy', y='Predictive_Accuracy', hue='Topic')
        # label average for each violin
        for violin in ax.collections:
            # get the x and y data for the violins
            x = violin.get_offsets()[:, 0]
            y = violin.get_offsets()[:, 1]
            # calculate the average y value for the violins
            average = np.mean(y)
            stdev = np.std(y)
            # plot the average value at the bottom of the violin
            # ax.text(x.mean(), average, f'{average:.2f}', color='black', ha='center', va='top')
            # move down the average value
            ax.text(x.mean(), average + 2 * stdev, f'{average:.2f}', color='white', ha='center', va='top')

        
       # no line
        # ax.plot((0, 1), (0, 1), linestyle='--', color='gray')
        # ax.set_xlim(0.45, 0.55)
        ax.set_ylim(0, 1.0)
        average_diff = np.mean(merged['Predictive_Accuracy'])
        beating_ratio = (merged['Predictive_Accuracy'] > merged['Oracle_Accuracy']).mean()
        ax.set_xlabel('Oracle Accuracy')
        ax.set_ylabel('GPT-4 Accuracy')
        ax.set_title(f'Evaluation Card Performance avg {average_diff:.2f}\n{file_name}')
        # ax.set_title(f'Evaluation Card Performance beating {beating_ratio:.2f}\n{file_name}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


    plt.tight_layout()
    plt.savefig('exp_rslt/combined_plots.svg')
    plt.show()

def create_plots_multiple(file_names):
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)

    num_files = len(file_names)
    if num_files <= 2:
        num_rows = 1
        num_cols = num_files
    elif num_files <= 4:
        num_rows = 2
        num_cols = 2
    elif num_files <= 6:
        num_rows = 2
        num_cols = 3
    else:
        num_rows = 3
        num_cols = 3

    x_lim = (0.45, 1.05)
    y_lim = (0.45, 1.05)
    # y_lim = (5, 10)
    point_size = 150

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(9 * num_cols, 6 * num_rows))

    for i, file_name in enumerate(file_names):
        df = pd.read_csv(f'exp_rslt/{file_name}')
        merged = df

        row = i // num_cols
        col = i % num_cols

        if num_files > 1:
            if num_rows > 1:
                ax = axs[row, col]
            else:
                ax = axs[col]
            
        else:
            ax = axs
        avg = np.mean(merged['Predictive_Accuracy'])
        scatter = sns.scatterplot(ax=ax, data=merged, x='Oracle_Accuracy', y='Predictive_Accuracy',
                                  hue='Topic', style='Student_Model', s=point_size)
        # x = y line
        ax.plot((0, 1), (0, 1), linestyle='--', color='gray')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        average_diff = np.mean(merged['Predictive_Accuracy'] - merged['Oracle_Accuracy'])
        beating_ratio = (merged['Predictive_Accuracy'] > merged['Oracle_Accuracy']).mean()
        ax.set_xlabel('Oracle Accuracy')
        ax.set_ylabel('GPT-4 Accuracy')
        ax.set_title(f'Evaluation Card Performance avg higher {average_diff:.2f}\n{file_name}')
        # ax.set_title(f'Evaluation Card Performance beating {beating_ratio:.2f}\n{file_name}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('exp_rslt/combined_plots.svg')
    plt.show()

def create_plot(method):
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)

    df1 = pd.read_csv(f'exp_rslt/{method.capitalize()}_mmlu_arxiv.csv')

    x_lim = (0.45, 1.05)
    # y_lim = (6, 9)
    y_lim = (0.35, 0.9)
    point_size = 150
    merged = df1

    fig, axs = plt.subplots(figsize=(9, 6))
    scatter = sns.scatterplot(ax=axs, data=merged, x='Oracle_Accuracy', y='Predictive_Accuracy', hue='Topic', style='Student_Model', s=point_size)
    # axs.plot((0, 1), (0, 1), linestyle='--', color='gray')
    axs.set_xlim(x_lim)
    axs.set_ylim(y_lim)
    average_diff = np.mean(merged['Predictive_Accuracy'] - merged['Oracle_Accuracy'])
    axs.set_xlabel('Oracle Accuracy')
    axs.set_ylabel('GPT-4 Accuracy')
    axs.set_title('Evaluation Card Performance, avg likert: {:.2f}'.format(np.mean(merged['Predictive_Accuracy'])))

    # Move the legend to the right side of the plot
    axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    plt.savefig(f'exp_rslt/{method.capitalize()}_mmlu_arxiv.svg')
    plt.show()


def create_plot2(file_name):
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)
    df = pd.read_csv(f'exp_rslt/{file_name}')
    
    # Correlation plot
    plt.figure(figsize=(8, 6))
    sns.regplot(data=df, x='Oracle_Accuracy', y='Predictive_Accuracy')
    corr = df['Oracle_Accuracy'].corr(df['Predictive_Accuracy'])
    plt.title(f'Correlation between Likert Rating and Model Accuracy (r={corr:.2f})')
    plt.savefig('exp_rslt/correlation_plot.png')
    plt.close()
    
    # Performance difference plot
    df['Diff'] = df['Predictive_Accuracy'] - df['Oracle_Accuracy']
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Oracle_Accuracy', y='Diff', hue='Topic', style='Student_Model')
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.title('Difference between Likert Rating and Model Accuracy')
    plt.savefig('exp_rslt/performance_diff_plot.png')
    plt.close()
    
    # Topic-wise analysis
    g = sns.FacetGrid(df, col='Topic', col_wrap=3)
    g.map(sns.scatterplot, 'Oracle_Accuracy', 'Predictive_Accuracy', 'Student_Model')
    g.set_titles(col_template='{col_name}')
    g.fig.suptitle('Likert Rating vs. Oracle Accuracy by Topic')
    # g.savefig('exp_rslt/topic_wise_plot.png')
    plt.close()
    
    # Student model comparison
    models = df['Student_Model'].unique()
    for model in models:
        model_df = df[df['Student_Model'] == model]
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=model_df, x='Oracle_Accuracy', y='Predictive_Accuracy', hue='Topic')
        plt.title(f'Likert Rating vs. Oracle Accuracy for {model}')
        # plt.savefig(f'exp_rslt/{model}_plot.png')
        plt.close()
    
    # Average Likert rating by model
    avg_ratings = df.groupby('Student_Model')['Predictive_Accuracy'].mean()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=avg_ratings.index, y=avg_ratings.values)
    plt.title('Average Likert Rating by Student Model')
    plt.xticks(rotation=45)
    # plt.savefig('exp_rslt/avg_rating_plot.png')
    plt.close()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_heatmap(file_name):
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)
    df = pd.read_csv(f'exp_rslt/{file_name}')
    
    # Calculate the mean Likert rating for each topic
    topic_means = df.groupby('Topic')['Predictive_Accuracy'].mean()
    
    # Sort the topics based on the mean Likert rating in descending order
    sorted_topics = topic_means.sort_values(ascending=False).index
    
    # Pivot the data to create a matrix with models as columns and topics as rows
    pivot_df = df.pivot_table(index='Topic', columns='Student_Model', values='Predictive_Accuracy')
    
    # Reorder the rows based on the sorted topics
    pivot_df = pivot_df.reindex(sorted_topics)
    
    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, cmap='coolwarm', annot=True, fmt='.2f', cbar_kws={'label': 'Likert Rating'})
    plt.title('Likert Rating Heatmap')
    plt.xlabel('Student Model')
    plt.ylabel('Topic')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('exp_rslt/likert_rating_heatmap.png')
    plt.show()

#


# create_plot(method='likert')
if __name__ == '__main__':
    file_names = ['Predictive_mmlu_arxiv copy.csv']
    # create_plots_multiple(file_names)
    create_jitter_plots_multiple(file_names)
    # create_plots_multiple
    # plot_heatmap('likert_mmlu_arxiv.csv')
    
