import src.my_library as my_library

pd, np, plt, sns = my_library.make_inital_imports()

def plot_numeric_yes_proportion(df, feature_var, num_grp, ax=None, fig=None):
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
    yes_count = df_bins.groupby("bin")['y'].apply(lambda y : (y=="yes").sum())
    total_count = df_bins.groupby("bin")['y'].count()

    # Probability of "yes" per bin
    df_yes_prob = (yes_count / total_count).to_frame(name="mean").reset_index()

    # Plot
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(15, 4), gridspec_kw={'height_ratios':[3, 0.5]})
    sns.barplot(x=df_bins['bin'], y=df_bins['y_binary'],
                 color='skyblue', ci=95, errwidth=2, ax=ax)

    # Annotate probablitites above bar
    for i, row in df_yes_prob.iterrows():
        ax.text(i-0.09, row['mean']+0.02, f"{row['mean']:.2f}", ha='center', va='bottom', fontsize=10, color='k')

    # Titles and labels 
    ax.set_title(f"{feature_var.capitalize()} YES-proportion", fontsize=16, fontweight='bold', y=1.02) 
    ax.set_ylabel("Proportion")

    # Absoulute and relative counts
    abs_values = df_bins.groupby(['bin'])['y'].count()
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

