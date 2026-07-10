import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# numeric features distribution plots
def univariate_numeric_distribution(df:pd.DataFrame, col:str):
    fig, ax = plt.subplots(ncols=2, figsize=(15, 5))

    sns.histplot(data=df, x = col, ax=ax[0], bins=30, kde=True, stat='density', color='lightskyblue')
    ax[0].lines[0].set_color('crimson')
    ax[0].set_title(f'{col.capitalize()} Distribution with Density Curve', fontsize=14, fontweight='bold')
    
    sns.boxplot(data=df, x = col, ax=ax[1], fill=False, color='k', flierprops=dict(marker='o', markerfacecolor='r'))
    ax[1].set_title(f'Box Plot for {col.capitalize()}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()
    

    # categorical features distribution plots
def univariate_categorical_distribution(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots(ncols=2, figsize=(20, 8))

    # Bar plot (frequency counts)
    sns.countplot(data=df, x=col, order=df[col].value_counts().index, ax=ax[0], palette="pastel")
    ax[0].set_title(f'{col.capitalize()} Distribution (Bar Plot)', fontsize=14, fontweight='bold', y=1.02)
    ax[0].tick_params(axis='x', rotation=45)

    # Pie plot (proportions)
    counts = df[col].value_counts()
    ax[1].pie(
        counts.values, labels=counts.index, 
        autopct='%1.1f%%', startangle=45, 
        colors=sns.color_palette("pastel"), 
        textprops={'fontsize': 12, 'weight':'bold', 'color':'k'}
    )
    ax[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax[1].set_title(f'{col.capitalize()} Distribution (Pie Chart)', fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()

# plots for numeric columns and target column
def plot_numeric_columns(df:pd.DataFrame, column:str, target:str='term_deposit', kde:bool = True,edgecolor:str = 'black',multiple:str ='stack') ->None:
    ''' 
    Plots histogram and normalized histogram of specified numeric columns with a target column in hue.

    Parameters:


    Returns:
    None: Displays the histogram and normalized histogram plots for numeric columns.
    '''
    fig, ax = plt.subplots(ncols=2, figsize=(15,5),sharey=True)

    # histogram plot for each type of term_deposit customer
    sns.histplot(data=df, y=column, hue=target ,kde=kde, edgecolor=edgecolor,linewidth=0.8, legend=False, alpha =0.6,multiple=multiple ,ax=ax[0])

    # Normalized histogram plot for each type of term_deposit customer
    sns.histplot(data=df, y=column, hue=target,kde=kde ,stat='percent', edgecolor=edgecolor,linewidth=0.8, multiple=multiple ,ax=ax[1])

    # set titles and labels
    ax[0].set_ylabel(column.capitalize(), fontsize=12, fontweight='bold')
    ax[0].set_xlabel('Term Deposit Count', fontsize=12, fontweight='bold')
    ax[1].set_xlabel('Term Deposit Percentage', fontsize=12, fontweight='bold')

    # Hide all spines for both subplots
    sns.despine(left=True,bottom=True)
    # Display settings
    plt.suptitle(f'Distribution of Counts & Percentages of Normalized Term Deposit for {column.capitalize()}', fontsize=14, fontweight='bold')
    plt.subplots_adjust(top=0.96)
    plt.tight_layout()
    plt.show()

# plots for categorical columns and target column
def plot_categorical_columns(df:pd.DataFrame, column: str, target:str='term_deposit'):
    """
    Plot counts and percentages of a categorical column with respect to a target column.

    Parameters:

    Returns:
    None: Displays the bar plot of counts and percentages of the categorical column with respect to the target column.
    """
    # count DataFrame for the specified column and target
    count_df = pd.crosstab(df[column], df[target]).stack().reset_index().rename(columns={0:'count'})
                                                                                
    # percentage DataFrame for the specified column and target
    pct_df = (pd.crosstab(df[column], df[target], normalize='index').round(4)*100).stack().reset_index().rename(columns={0:'percentage'})

    # Display the counts DataFrame
    display(count_df.head(10))

    # Create subplots for count and percentage
    fig, ax = plt.subplots(ncols=2, figsize=(15, 5), sharey=True)

    #plot using seaborn barplot for count and percentage
    sns.barplot(data=count_df, x='count', y=column, hue=target, ax=ax[0], palette='Set2', edgecolor='black', linewidth=0.8)
    sns.barplot(data=pct_df, x='percentage', y=column, hue=target, ax=ax[1], palette='Set2', edgecolor='black', linewidth=0.8,legend=False)

    # Set titles and labels
    ax[0].set_ylabel(column.capitalize(), fontsize=12, fontweight='bold')
    ax[0].set_xlabel('Count', fontsize=12, fontweight='bold')
    ax[1].set_xlabel('Percentage', fontsize=12, fontweight='bold')
    

    # Hide all spines for both subplots
    sns.despine(left=True, bottom=True)
    
    # Display settings
    plt.suptitle(f'Counts & Percentages of Term Deposit for each category in {column.capitalize()}', fontsize=14, fontweight='bold')
    plt.subplots_adjust(top=0.96)   
    plt.tight_layout()

    #add legend to the first subplot
    ax[0].legend(labelcolor='linecolor', loc='lower right', title=target.capitalize(),edgecolor='black', fontsize=10, title_fontsize='12', frameon=True, framealpha=0.8)
    plt.show()

                                                                                                                                        