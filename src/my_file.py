import src.my_library as my_library

pd, np, plt, sns = my_library.make_inital_imports()

from scipy.stats import stats

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
        ax.text(i, row['mean']+0.02, f"{row['mean']:.2f}", ha='center', va='bottom', fontsize=14, fontweight='bold', color='k')

    # Titles and labels 
    ax.set_title(f"{feature_var.capitalize()} YES-proportion", fontsize=16, fontweight='bold', y=1.02) 
    ax.set_ylabel("Proportion", fontsize=14, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, fontweight='bold')
    ax.set_xlabel(f"{feature_var}_bins", fontsize=14, fontweight='bold')

    # Absoulute and relative counts
    abs_values = df_bins.groupby(['bin'])[ycol].count()
    rel_values = [(f"{i:.0f}" if i >= 0.5 else "<0.5") for i in (abs_values / df.shape[0] * 100)]
    
    # Format x-axis
    bin_type = 'int' if df[feature_var].dtype == 'int' else 'float'
    if bin_type == 'int':
        x_labl = [
            f"({int(i.left)}...{int(i.right)}]\n-----\n{j} ({k}%)" 
                  for i, j, k in zip(df_yes_prob['bin'], abs_values, rel_values)
        ]
    if bin_type == 'float':
        x_labl = [
            f"({i.left:.2f}...{i.right:.2f}]\n-----\n{j} ({k}%)" 
            for i, j, k in zip(df_yes_prob['bin'], abs_values, rel_values)
        ]
    
    ax.set_xticklabels(x_labl, fontsize=12, fontweight='bold')

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


# Bivariate Analysis with Heatmaps
def plot_heatmap_bivariate_analysis(df=None, feature_var=None, seasonal_var=None, y_binary=None, fig=None, ax=None):
    """Plot heatmap for total observed and yes counts for given variables."""
    # order for reindex
    values_count = df[feature_var].value_counts()
    # Total counts
    total_counts = pd.crosstab(df[feature_var], df[seasonal_var], values=df[y_binary], aggfunc='count').reindex(index=values_count.index)
    # Yes counts
    yes_counts = pd.crosstab(df[feature_var], df[seasonal_var], values=df[y_binary], aggfunc='mean').round(2).reindex(index=values_count.index)

    # Plot heatmap
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    # Total observed heatmap
    scatter = sns.heatmap(total_counts, annot=True, fmt='.0f', cmap='Blues', cbar=True, ax=ax[0])
    colorbar = scatter.collections[0].colorbar
    colorbar.set_ticks([])
    ax[0].set_title(f'Total Observed Counts by\n{feature_var.capitalize()} and {seasonal_var}', fontsize=14, fontweight='bold', y=1.04)
    ax[0].tick_params(axis='y', rotation=25)
    ax[0].set_ylabel('')
    ax[0].set_xlabel('')

    # yes counts heamap
    heat_map = sns.heatmap(yes_counts, annot=True, fmt='.2f', cmap='Blues', cbar=True, ax=ax[1])
    colorbar = heat_map.collections[0].colorbar
    colorbar.set_ticks([])
    ax[1].set_title(f'Conversion Rate by\n{feature_var.capitalize()} and {seasonal_var}', fontsize=14, fontweight='bold', y=1.04)
    ax[1].tick_params(axis='y', rotation=25)
    ax[1].set_ylabel('')
    ax[1].set_xlabel('')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    return fig, ax


# Cumulative proportion of total contact and yes responses
def plot_cumsum_total_yes_response(df=None, feature_var=None, cumsum_total_var=None, cumsum_yes_var=None, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(14,6))
    new_df = df.reset_index()

    # Left: Cumulative contribution of total contact
    ax[0].plot(new_df[feature_var], new_df[cumsum_total_var], marker='o', color="skyblue")
    ax[0].set_title("Cumulative Sum of Total Contacts", fontsize=14, fontweight='bold', y=1.05)
    ax[0].set_ylabel("Cumulative Proportion of Total")
    ax[0].tick_params(axis="x", rotation=45)


    # Right: Cumulative contribution of yes responses
    ax[1].plot(new_df[feature_var], new_df[cumsum_yes_var], marker="o", color="orange")
    ax[1].set_title("Cumulative Contribution of Yes Responses", fontsize=14, fontweight='bold', y=1.05)
    ax[1].set_ylabel("Cumulative Proportion of Yes")
    ax[1].tick_params(axis="x", rotation=45)

    for i, row in new_df.iterrows():
        ax[0].text(row[feature_var], row[cumsum_total_var]*1.01,
                    f"{row[cumsum_total_var]:.2f}%", ha='center', va='bottom', color='k', fontsize=10)
        ax[1].text(row[feature_var], row[cumsum_yes_var]*1.01,
                    f"{row[cumsum_yes_var]:.2f}%", ha='center', va='bottom', color='k', fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()


# Statistical test for relationship between categorical variables and y(term deposit)
def chi_square_categorical_y(data, feature_var, target_var):
    # Contingency table
    table = pd.crosstab(data[feature_var], data[target_var])

    # Chi-square test
    chi2, p, dof, expected = stats.chi2_contingency(table)

    # Compute Cramér's V
    n = table.sum().sum()
    k = min(table.shape)
    cramers_v = np.sqrt(chi2 / (n * (k - 1)))
    significance = 'small' if cramers_v < 0.1 else 'medium' if cramers_v < 0.3 else 'large' if cramers_v < 0.5 else 'very large'
    df = pd.DataFrame(
        [chi2, p, cramers_v, significance], index=['Chi-square', 'pvalue', 'cramers_v', 'effect_size'], columns=['statistical_values'])

    df = df.round(4)
    return df


def plot_odd_ratio_table(data, feature_var, target_var, baseline):
    """Compute odd ratios for categories of a given feature relative to a baseline."""

    # Contingency table
    table = pd.crosstab(data[feature_var], data[target_var])

    # Compute odds for each job
    table["odds"] = table["yes"] / table["no"]

    # Choose baseline
    if baseline is None:
        baseline = table.index[0]
    baseline_odds = table.loc[baseline, "odds"]

    # Compute odds ratio relative to baseline
    table[f"odds_ratio_vs_{baseline}"] = table["odds"] / baseline_odds

    # Create side-by-side figure
    fig, ax = plt.subplots(1, 2, figsize=(16,6))

    # Left side: table displayed as text
    ax[0].axis("off")  # hide axes
    table_display = table[["yes", "no", "odds", f"odds_ratio_vs_{baseline}"]]
    ax[0].table(cellText=table_display.round(2).values,
                rowLabels=table_display.index,
                colLabels=table_display.columns,
                loc="center")

    ax[0].set_title(f"Odds Ratio Table (vs {baseline})")

    # Right side: bar plot of odds ratios
    ax[1].bar(table.index, table[f"odds_ratio_vs_{baseline}"], color="skyblue")
    ax[1].axhline(1.0, color="red", linestyle="--", label=f"Baseline ({baseline})")
    ax[1].set_ylabel(f"Odds Ratio vs {baseline.capitalize()}")
    ax[1].set_title(f"Odds Ratios of YES-answers by {feature_var} Category")
    ax[1].tick_params(axis="x", rotation=45)
    ax[1].legend()

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()


def plot_numeric_heatmap_bivariate_analysis(df=None, month_var=None, year_var=None, feature_var=None, display_name=None, fig=None, ax=None):
    """Plot numeric variables median value for seasonal trend with overall and yes-proportion data"""  

    # Overall data
    total_df = pd.crosstab(
        df[month_var], df[year_var],
        df[feature_var], aggfunc='median'
    )
    
    # yes-proportion data
    new_df = df[df['y']=='yes']
    yes_df = pd.crosstab(
        new_df[month_var], new_df[year_var],
        new_df[feature_var], aggfunc='median'
    )
    # Visualization
    if fig is None or ax is None:
        fig, ax = plt.subplots(1,2, figsize=(15, 5))
    sns.heatmap(total_df, annot=True, fmt='.2f', cmap='Blues', cbar=True, ax = ax[0])
    ax[0].set_yticklabels(ax[0].get_yticklabels(), rotation=25)
    ax[0].set_title(f"Overall median '{feature_var}' by {display_name.capitalize()} and {year_var.capitalize()}", fontsize=14,fontweight='bold', y=1.02)

    sns.heatmap(yes_df, annot=True, fmt='.2f', cmap='Blues', cbar=True, ax = ax[1])
    ax[1].set_yticklabels(ax[0].get_yticklabels(), rotation=25)
    ax[1].set_title(f"YES-proportion median '{feature_var}' by {display_name.capitalize()} and {year_var.capitalize()}", fontsize=14, fontweight='bold', y=1.02)

    
    plt.show()

    return fig, ax


def plot_boxplot_yes_no_distribution(df=None, xcol=None, feature_name=None, fig=None, ax=None):
    """Plot distribution of numeric feature by yes-no answer"""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x=xcol, y=feature_name, color='b',
                medianprops={'ls':'-', 'lw':2.5, 'color':'r',
                'marker': 'o', 'markersize':6,'markerfacecolor':'white', 'markeredgecolor':'k'})
    
    bbox_props = dict(boxstyle='round', edgecolor='k', facecolor='white', alpha=0.9)
    median_no, median_yes = df.groupby([xcol])[feature_name].median()
    plt.text(1.6, ax.get_ylim()[1], f"No Meidan = {median_no:.2f}\nYes Median = {median_yes:.2f}",
            ha='left', va='center', fontsize=14, color='k', bbox=bbox_props)
    
    ax.set_title(f"Distribution of {feature_name} by {xcol}", fontsize=14, fontweight='bold', y=1.02)
    plt.show()
    return fig, ax


# Compute Variance by year of numeric variable
def var_by_year(df=None, seasonal_var=None, feature_var=None):
    """Compute Variance by year of numeric variable"""
    results = []
    yearly_var = df.groupby([seasonal_var])[feature_var].var().round(2)
    yearly_observ = round(df[seasonal_var].value_counts(normalize=True)*100, 1)
    df = pd.concat([yearly_var, yearly_observ], axis='columns')    
    df.columns = [f'variation of `{feature_var}`', 'Number of observatios(%)']

    return df
