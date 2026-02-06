import src.my_library as my_library

pd, np, plt, sns = my_library.make_inital_imports()

def plot_numeric_yes_proportion(df=None, feature_var=None, ycol=None, num_grp=10, widths=float, ax=None, fig=None):
    """"
    Splits the numeric features into bins and 
    computes the probability of 'yes' (y_binary == 1) in each bin.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe containing the feature and target column 'y'.
    feature_name : str
        The name of the numeric feature column.
    num_gr : int
        Desired number of groups (bins).
    """
    # Count unique values
    n_unique = df[feature_var].nunique()

    # Adjust number of group if needed
    if num_grp > n_unique:
        num_grp = n_unique
        print(f"WARNING:\n"
              f"The number of groups cannot be greater than number of unique values ({n_unique}).\n"
              f"It has been reduced automatically.")
    else:
        num_grp += 1

    # Create evenly spaced bins
    bins = np.linspace(df[feature_var].min(),
                       df[feature_var].max(),
                       num=num_grp)
    
    # Assign each values to a bins
    bins_series = pd.cut(df[feature_var], bins=bins, include_lowest=True)
    bins_series.name = f"{feature_var}_bins"

    # Combine feature, bins and target columns
    df_bins = df.assign(bin=bins_series)

    # Compute total counts and yes responses per bin
    yes_count = df_bins.groupby("bin")[ycol].apply(lambda y : (y==1).sum())
    total_count = df_bins.groupby("bin")[ycol].count()

    # Probability of "yes" per bin
    df_yes_prob = (yes_count / total_count).to_frame(name="mean").reset_index()

    # Plot
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(15, 4))
    sns.barplot(x='bin', y='mean',data=df_yes_prob,
                 color='skyblue', ci=None, width=widths, ax=ax)
    # Annotate probablitites above bar
    for i, row in df_yes_prob.iterrows():
        ax.text(i, row['mean']+0.02, f"{row['mean']:.2f}", ha='center', va='bottom', fontsize=10, color='k')

    # Titles and labels 
    ax.set_title(f"{feature_var.capitalize()} YES-proportion", fontsize=16, fontweight='bold', y=1.02) 
    ax.set_ylabel("Proportion")
    ax.set_xlabel(f"{feature_var}_bins")

    # Absoulute and relative counts
    abs_values = df_bins.groupby(['bin'])[ycol].count()
    rel_values = [(f"{i:.0f}" if i >= 0.5 else "<0.5") for i in (abs_values / df.shape[0] * 100)]
    
    # Format x-axis
    bin_type = 'int' if df[feature_var].dtype == 'int' else 'float'
    if bin_type == 'int':
        x_labl = [f"({int(i.left)}...{int(i.right)}]\n-----\n{j} ({k}%)" 
                  for i, j, k in zip(df_yes_prob['bin'], abs_values, rel_values)]
    else:
        x_labl = [f"({i.left}...{i.right}]\n-----\n{j} ({k}%)]" for i, j, k in zip(df_yes_prob['bin'], abs_values, rel_values)]
    
    ax.set_xticklabels(x_labl)

    return df_yes_prob, fig, ax


def plot_numeric_distribution(df, feature_var, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(15, 3), gridspec_kw={'height_ratios':[3, 0.5]}, sharex=True)
    sns.histplot(df[feature_var], stat='density', kde=True, lw=0, ax=ax[0]);
    sns.kdeplot(df[feature_var], color='#ffb500', lw=3, ax=ax[0]);

    # Add text for features descriptive statistics
    _, mean_val, _, min_val, q25, median, q75, max_val = np.round(df[feature_var].describe(), 2)
    features_stats_text = f"min : {min_val}\nmax : {max_val}\nmean : {mean_val}\nq25 : {q25}\nq50 : {median}\nq75 : {q75}"
    box = dict(boxstyle='round', alpha=0.5, fc='white', ec='k')
    x_coor, y_coor = np.quantile(ax[0].get_xlim(), 0.85), np.quantile(ax[0].get_ylim(), 0.2)

    ax[0].text(x_coor, y_coor, features_stats_text, fontsize=14, color='k', bbox=box)
    ax[0].set_title(f"{feature_var.capitalize()} distribution", fontsize=16, fontweight='bold', y=1.02)

    sns.boxplot(data = df, x =feature_var, ax=ax[1])
    
    plt.tight_layout()

    return fig, ax


# Univariate analysis function for categorical variables with bar plots
def plot_univariate_analysis_categorical_var(df=None, categorical_var=None, ycol=None, display_name=None):
    """Compute total count of customers contacted and percentage of customers who subscribed to a term deposit."""
    # Count of campaign for each category
    total_counts = df.groupby([categorical_var])[ycol].value_counts().unstack().fillna(0)   
    total_counts['total'] = total_counts.sum(axis='columns')
    total_counts = total_counts.sort_values(by='total', ascending=False)
    total_counts['conversion_rate'] = (total_counts['yes'] / total_counts['total']).round(2)
    total_counts = total_counts.assign(
        cumsum_total_proportion = lambda x: (x['total'].cumsum() / x['total'].sum())*100,
        cumsum_yes_proportion = lambda x: (x['yes'].cumsum() / x['yes'].sum())*100
    )
    
    # Visualization
    fig, ax = plt.subplots(2, 1, sharex=True,figsize=(15, 8), dpi=100)
    colors = [(0.3, 0.8, 1), (0.6, 0.6, 1), '#0000FF']
    # plot for total campaigns
    x_pos = range(len(total_counts.index))
    ax[0].bar(x_pos, total_counts['total'], color=colors)
    ax[0].set_xticks(x_pos)
    ax[0].set_xticklabels(total_counts.index)
    ax[0].set_ylabel("Count")
    ax[0].set_title(display_name + ' Distribution', fontsize=14, fontweight='bold', y=1.04)
    for i, v in enumerate(total_counts['total']):
        ax[0].text(i, v + 200, f"{int(v)} ({(v/(total_counts['total'].sum(axis=0))):.0%})", ha='center', fontsize=10)

    # plot for successful subscriptions
    ax[1].bar(x_pos, total_counts['conversion_rate'], color=colors)
    ax[1].set_xticks(x_pos)
    ax[1].set_xticklabels(total_counts.index)
    ax[1].set_ylabel('Proportion')
    ax[1].set_title(display_name +" " + 'YES-proportion', fontsize=14, fontweight='bold', y=1.04)
    for i, v in enumerate(total_counts['conversion_rate']):
        ax[1].text(i, v , f"({total_counts['conversion_rate'].iloc[i]:.2f})", ha='center', fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()

    return total_counts, fig

