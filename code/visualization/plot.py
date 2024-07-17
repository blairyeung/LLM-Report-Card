import pandas as pd
import numpy as np

df1 = pd.read_csv('ExpRslts/Predictive_ALL.csv')
df2 = pd.read_csv('results.csv')


merged = pd.merge(df1, df2, how='inner', left_on=['Topic', 'Student_Model'], right_on=['topic', 'model'])

import seaborn as sns
from matplotlib import pyplot as plt

fig, axs = plt.subplots(2, 2, figsize=(12, 9))

# New limits for the plots
x_lim = y_lim = (0.5, 0.9)

# Adjusting point size
point_size = 150
#  Could not interpret value `topic` for parameter `hue`
# change the hue to model


# Scatter plot for model_acc ("GPT-4")
sns.scatterplot(ax=axs[0, 0], data=merged, x='Oracle_Accuracy', y='Predictive_Accuracy', hue='topic', style='model', s=point_size)
axs[0, 0].plot(x_lim, y_lim, linestyle='--', color='gray')  # y=x line
axs[0, 0].set_title('Evaluation Card Performance')
axs[0, 0].set_xlim(x_lim)
axs[0, 0].set_ylim(y_lim)
axs[0, 0].set_xlabel('Oracle Accuracy')
axs[0, 0].set_ylabel('GPT-4 Accuracy')

# Scatter plot for Logistic Regression
sns.scatterplot(ax=axs[0, 1], data=merged, x='Oracle_Accuracy', y='logistic_reg_acc', hue='topic', style='model', s=point_size)
axs[0, 1].plot(x_lim, y_lim, linestyle='--', color='gray')  # y=x line
axs[0, 1].set_title('Logistic Regression Performance')
axs[0, 1].set_xlim(x_lim)
axs[0, 1].set_ylim(y_lim)
axs[0, 1].set_xlabel('Oracle Accuracy')
axs[0, 1].set_ylabel('Logistic Regression Accuracy')

# Scatter plot for Few-shot
sns.scatterplot(ax=axs[1, 0], data=merged, x='Oracle_Accuracy', y='knn_accuracy', hue='topic', style='model', s=point_size)
axs[1, 0].plot(x_lim, y_lim, linestyle='--', color='gray')  # y=x line
axs[1, 0].set_title('KNN Performance')
axs[1, 0].set_xlim(x_lim)
axs[1, 0].set_ylim(y_lim)
axs[1, 0].set_xlabel('Oracle Accuracy')
axs[1, 0].set_ylabel('Few-shot Accuracy')

# Removing the fourth subplot (as per the updated request)
# fig.delaxes(axs[1][1])  # Delete the unused fourth subplot
trend_data = pd.concat([
    merged[['Oracle_Accuracy', 'Predictive_Accuracy']].rename(columns={'Predictive_Accuracy': 'Accuracy'}).assign(Method='Our_card'),
    merged[['Oracle_Accuracy', 'logistic_reg_acc']].rename(columns={'logistic_reg_acc': 'Accuracy'}).assign(Method='Logistic Regression'),
    merged[['Oracle_Accuracy', 'knn_accuracy']].rename(columns={'knn_accuracy': 'Accuracy'}).assign(Method='KNN')
])

sns.scatterplot(ax=axs[1, 1], data=trend_data, x='Oracle_Accuracy', y='Accuracy', hue='Method', style='Method', s=100)
axs[1, 1].set_title('General Trend Across Methods')
axs[1, 1].set_xlabel('Oracle Accuracy')
axs[1, 1].set_ylabel('Method Accuracy')
axs[1, 1].plot(x_lim, y_lim, linestyle='--', color='gray')  # y=x line


plt.tight_layout()
plt.savefig('ExpRslts/meta_plot.png')
# plt.show()



fig, axs = plt.subplots(1, 1, figsize=(7, 5))

# New limits for the plots
x_lim = y_lim = (0.5, 0.9)

# Adjusting point size
point_size = 150
#  Could not interpret value `topic` for parameter `hue`
# change the hue to model


# Scatter plot for model_acc ("GPT-4")
sns.scatterplot(ax=axs, data=merged, x='Oracle_Accuracy', y='Predictive_Accuracy', hue='topic', style='model', s=point_size)
axs.plot(x_lim, y_lim, linestyle='--', color='gray')  # y=x line
axs.set_title('Evaluation Card Performance')
axs.set_xlim(x_lim)
axs.set_ylim(y_lim)
axs.set_xlabel('Oracle Accuracy')
axs.set_ylabel('GPT-4 Accuracy')


plt.tight_layout()
plt.savefig('ExpRslts/single_plot.png')
plt.show()


