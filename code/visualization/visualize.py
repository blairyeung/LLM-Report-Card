import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_heatmaps(topics, df):
    # Create a pandas dataframe from the provided data

    # only topics that are in the provided list
    df = df[df['Topic'].isin(topics)]

    # only include guesser of meta-llama/Meta-Llama-3-70B-Instruct
    df = df[df['Guesser'] == 'gpt-4o']

    # reflect the df (concat with student and student 2 swapped)
    df_reflected = df.copy()
    df_reflected['Student_Model_1'] = df['Student_Model_2']
    df_reflected['Student_Model_2'] = df['Student_Model_1']
    df = pd.concat([df, df_reflected])

    contrastive_data = df[df['Card_type'] == 'few_shot']
    generative_data = df[df['Card_type'] == 'card']

    # Get unique models from both generation methods
    all_models = pd.concat([contrastive_data[['Student_Model_1', 'Student_Model_2']], 
                            generative_data[['Student_Model_1', 'Student_Model_2']]]).stack().unique()

    # Convert all_models to a pandas Series
    all_models = pd.Series(all_models)


    # Create subplots for each topic and generation method
    fig, axes = plt.subplots(2, len(topics), figsize=(6 * len(topics), 12))

    for i, topic in enumerate(topics):
        for j, (data, method) in enumerate(zip([contrastive_data, generative_data], ['Few_shot', 'Generative'])):
            topic_data = data[data['Topic'] == topic]

            # Pivot table for accuracy by student models and average entries with the same pair
            pivot = pd.pivot_table(topic_data, values='Contrastive_Accuracy', index='Student_Model_1', columns='Student_Model_2', aggfunc='mean')

            missing_models = all_models[~all_models.isin(pivot.index)]
            for model in missing_models:
                # if the whole pivot table is empty, fill it with 0
                if pivot.empty:
                    pivot = pd.DataFrame(0, index=[model], columns=[model])
                else:
                    pivot.loc[model] = [0] * len(pivot.columns)

            missing_models = all_models[~all_models.isin(pivot.columns)]
            for model in missing_models:
                pivot[model] = [0] * len(pivot.index)

            # Fill NaN values with 0
            pivot = pivot.fillna(0)

            # Make the matrix symmetrical by combining with the transpose
            pivot = pivot.combine_first(pivot.T)

            # Sort the pivot table based on index and columns
            pivot = pivot.reindex(index=sorted(pivot.index), columns=sorted(pivot.columns))

            # Heatmap
            # use blues
            cmap = 'Blues' if method == 'Contrastive' else 'Reds'
            im = axes[j, i].imshow(pivot, cmap=cmap, vmin=0, vmax=1)
            axes[j, i].set_xticks(np.arange(len(pivot.columns)))
            axes[j, i].set_yticks(np.arange(len(pivot.index)))
            axes[j, i].set_xticklabels([])
            axes[j, i].set_yticklabels([])
            avg_acc = pivot.sum().sum() / (len(pivot.index) * len(pivot.columns) - len(pivot.index))
            axes[j, i].set_title(f'{method} on {topic} (avg. acc: {avg_acc:.2f})')

            # Add text labels to each block in the heatmap
            for x in range(len(pivot.index)):
                for y in range(len(pivot.columns)):
                    axes[j, i].text(y, x, f'{pivot.iloc[x, y]:.2f}', ha='center', va='center', color='w')

            # Add model names at the left and bottom of the plot
            if i == 0:
                axes[j, i].set_yticklabels(pivot.index, rotation=45, ha='right')
            if j == 1:
                axes[j, i].set_xticklabels(pivot.columns, rotation=45, ha='right')

    # Add colorbars and adjust the layout
    # fig.colorbar(im, ax=axes.ravel().tolist())
    plt.tight_layout()
    plt.savefig('heatmap_contrastive.png', dpi=400)
    plt.show()


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_model_accuracies(topics):
    # Create a pandas dataframe from the provided data
    df = pd.read_csv('ExpRslts/contrastive/contrastive_mmlu.csv')

    # only topics that are in the provided list
    df = df[df['Topic'].isin(topics)]

    # only include guesser of meta-llama/Meta-Llama-3-70B-Instruct
    df = df[df['Guesser'] == 'meta-llama/Meta-Llama-3-70B-Instruct']

    # reflect the df (concat with student and student 2 swapped)
    df_reflected = df.copy()
    df_reflected['Student_Model_1'] = df['Student_Model_2']
    df_reflected['Student_Model_2'] = df['Student_Model_1']
    df = pd.concat([df, df_reflected])

    contrastive_data = df[df['Card_type'] == 'few_shot']
    generative_data = df[df['Card_type'] == 'card']

    # Create subplots for each topic and generation method
    fig, axes = plt.subplots(2, len(topics), figsize=(6 * len(topics), 12))

    for i, topic in enumerate(topics):
        for j, (data, method) in enumerate(zip([contrastive_data, generative_data], ['Few_shot', 'Generative'])):
            topic_data = data[data['Topic'] == topic]

            # Pivot table for accuracy by student models and average entries with the same pair
            pivot = pd.pivot_table(topic_data, values='Contrastive_Accuracy', index='Student_Model_1', columns='Student_Model_2', aggfunc='mean')

            # Compute model accuracies
            model_accuracies = {}
            for model in pivot.index:
                row_sum = pivot.loc[model].sum()
                col_sum = pivot[model].sum()
                total_sum = row_sum + col_sum
                avg_accuracy = total_sum / (2 * len(pivot) - 2)
                model_accuracies[model] = avg_accuracy

            models = list(model_accuracies.keys())
            accuracies = list(model_accuracies.values())

            # Bar plot
            axes[j, i].bar(models, accuracies)
            # a h line of average, label it

            average = sum(accuracies) / len(accuracies) if len(accuracies) > 0 else 0
            axes[j, i].axhline(average, color='r', linestyle='--', label='Average')
            axes[j, i].set_xlabel('Model')
            axes[j, i].set_ylabel('Average Accuracy')

            axes[j, i].set_title(f'{method} on {topic} avg is {average:.2f}')
            axes[j, i].set_xticklabels(models, rotation=45, ha='right')
            axes[j, i].set_ylim(0, 1)

            for x, accuracy in enumerate(accuracies):
                axes[j, i].text(x, accuracy + 0.01, f'{accuracy:.2f}', ha='center')

    plt.tight_layout()
    plt.savefig('model_accuracies.png', dpi=400)
    plt.show()



if __name__ == '__main__':

    # topics = ['dataset_Roleplaying as a fictional character', 'power-seeking-inclination', 'high_school_mathematics', 'high_school_physics']
    # topics = ['power-seeking-inclination', 'self-awareness-general-ai']
    topics = ['high_school_mathematics', 'high_school_physics', 'high_school_chemistry']
    plot_heatmaps(topics)

    # plot_model_accuracies(topics)