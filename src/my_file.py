import src.my_library as my_library

pd, np, plt, sns = my_library.make_inital_imports()


def feature_analysis(df_, feature_name, feature_type, num_gr=10):
    # Check if feature exists
    if feature_name not in df_.columns:
        print(f"Error : {feature_name} is not found in DataFrame columns.")
        return None

    try:
        if feature_type == 'categorical':
            # Two subplots: distribution + yes proportion
            fig, ax = plt.subplots(1, 2, 
                                   figsize=(15, 10),
                                   gridspec_kw = {'height_ratios': [3, 2]}, 
                                   sharey = True)
            
            # Distribution
            plot_distribution(df_= df_, feature_name = feature_name, feature_type = feature_type, 
                              plot_options={'fig': fig, 'ax': ax[0]})
            ax[0].set_title(f"{feature_name} Distribution",
            fontsize=15, fontweight='bold', pad=20)
            # Yes Proportion
            plot_yes_proportion(df_ = df_, feature_name = feature_name, feature_type = feature_type,
                                plot_options = {'fig': fig, 'ax': ax[1]})
            
            plt.tight_layout()
            return fig
        
        elif feature_type == 'numeric':
            # Two subplots: distribution + yes proportion(with grouping)
            fig, ax = plt.subplots(1, 2,
                                   figsize=(15, 10),
                                   gridspec_kw = {'height_ratios':[3,2]},
                                   sharey=True)
            
            # Distribution
            plot_distribution(df_ = df_, feature_name = feature_name, feature_type = feature_type,
                              plot_options = {'fig': fig, 'ax': ax[0]})
            ax[0].set_title(f"{feature_name} Distribution", fontsize=15, fontweight='bold', pad=20)
            # Yes Proportion
            plot_yes_proportion(df_ = df_, feature_name = feature_name, feature_type = feature_type,
                                plot_options = {'fig': fig, 'ax': ax[1]})
            
            plt.tight_layout()
            return fig
        else:
            print(f"Error: feature_type `{feature_type}` must be either 'categorical' or 'numeric'.")

    except Exception as e:
        print(f"An error occured while analysing `{feature_name}` : {e}")
        return None
    
def print_error_msg(msg, title="ERROR"):
    print("=" * 40)
    print(f"{title}:")
    print(msg)
    print("=" * 40)

def make_round(dataset, roundby=2):
    if roundby != 0:
        # Round to given decimal places
        return dataset.apply(lambda col : [np.round(val, roundby) if not pd.isna(val) else np.nan for val in col])
    else:
        return dataset.apply(lambda col : [np.round(val, roundby) if not pd.isna(val) else np.nan for val in col])

def plot_heatmap(df_, title_, annot_, fmt_, ax_, show_cbar=True):
    if ax_ is None:
        fig, ax_ = plt.subplots(figsize=(8, 6))
        sns.heatmap(df = df_, annot=annot_, fmt=fmt_, cbar = show_cbar, linewidth=1, ax=ax_)

        ax_.set_title(title_, fontsize=16, fontweight='bold', pad=20)
        ax_.set_ylabel("")
        ax_.set_yticklabels(ax_.get_yticklabels, rotation=25)
        ax_.set_xticklabels(ax_.get_xticklabels, rotation=45, ha='right')

        return ax_


def error_handling(func_name, by, df_, feature_name, fig, ax):
    """
    Handles error checking for plotting functions.

    Args:
        func_name (str): Name of the calling function.
        by (str): Grouping parameter ('month', 'year', or 'month and year').
        df_ (pd.DataFrame): Input DataFrame.
        feature_name (str): Name of the feature to plot.
        fig (matplotlib.figure.Figure): Matplotlib figure object.
        ax (matplotlib.axes.Axes): Matplotlib axes object.

    Returns:
        int: -1 if an error is detected, None otherwise.
    """
    # Check if the DataFrame is None
    if df_ is None:
        print_error_msg(f'{func_name}(): The dataset is None.')
        return -1

    # Check if the feature_name exists in the DataFrame
    if feature_name not in df_.columns:
        if feature_name:
            print_error_msg(f'{func_name}(): There is no such feature in dataset.')
            fig.delaxes(ax)
        else:
            print_error_msg(f'{func_name}(): Please, specify the feature_name.')
            fig.delaxes(ax)

        return -1

    # Check if the 'by' parameter is valid
    if by not in ['month', 'year', 'month and year']:
        print_error_msg('plot_observations_by(): Only by `year` or `month` or `month and year` are acceptable.')
        fig.delaxes(ax)
        return -1

    # Check if the feature is categorical for 'year' or 'month' modes
    if feature_name and by != 'month and year' and df_[feature_name].dtype != 'object':
        print_error_msg(f'{func_name}(): For `year` or `month` modes only categorical features are acceptable.')
        fig.delaxes(ax)
        return -1

    # Check if the feature is numeric for 'month and year' mode
    if feature_name and by == 'month and year' and df_[feature_name].dtype == 'object':
        print_error_msg(f'{func_name}(): For `month and year` mode only numeric features are acceptable.')
        fig.delaxes(ax)
        return -1

    # Missing attributes (by, fig, ax)
    # Check if the 'by' parameter is set
    if not by:
        print_error_msg(f'{func_name}(): Please, set the `by` attribute.')
        fig.delaxes(ax)
        return -1

    # Check if the 'fig' parameter is set
    if not fig:
        print_error_msg(f'{func_name}(): Please, set the `fig` attribute.')
        fig.delaxes(ax)
        return -1

    # Check if the 'ax' parameter is set
    if not ax:
        print_error_msg(f'{func_name}(): Please, set the `ax` attribute.')
        fig.delaxes(ax)
        return -1

def plot_observations_by(by=None, df_=None, feature_name=None, fig=None, ax=None):
    """
    Plot observations by a specified feature and grouping.

    Parameters:
    - by: str, the grouping criterion ('year', 'month', or 'month and year')
    - df_: DataFrame, the input data
    - feature_name: str, the feature to analyze
    - fig: matplotlib figure object, optional
    - ax: matplotlib axes object, optional
    """

    if error_handling('plot_observations_by', by, df_, feature_name, fig, ax) == -1:
        return

    # Define the order of months
    months_order = pd.Categorical(['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
                                  ordered=True)

    # Determine the order of categories for the feature
    if feature_name == 'month':
        order = months_order
    else:
        order = pd.Categorical(df_[feature_name].value_counts().index, ordered=True)

    if feature_name:

        if by in ['year', 'month']:

            if by == 'year':
                # Create a crosstab of the feature by year
                observations = pd.crosstab(df_[feature_name], df_['year']).reindex(labels=order)
            else:
                # Create a crosstab of the feature by month
                lbls_order = order
                cols_order = months_order
                observations = pd.crosstab(df_[feature_name], df_['month']).reindex(labels=lbls_order,
                                                                                   columns=cols_order)

            # Plot the heatmap
            plot_heatmap(title_=f'{feature_name.capitalize()}\'s categories by {by}',
                         data_=np.log1p(observations),
                         annot_=make_round(observations, roundby=0),
                         fmt_='.0f',
                         cmap_=sns.color_palette('light:b', as_cmap=True),
                         ax_=ax)

        # ONLY FOR NUMERIC FEATURES
        elif by == 'month and year':
            # Create a crosstab of the feature by month and year
            observations_by_months_and_year = pd.crosstab(index=df_['month'], columns=df_['year'], values=df_[feature_name],
                                                          aggfunc='median').reindex(labels=months_order)

            if df_[feature_name].dtype == 'float':
                # For float features, use the original values
                data = observations_by_months_and_year
                annot = make_round(observations_by_months_and_year, roundby=2)
                fmt = '.2f'
                cmap = sns.color_palette('coolwarm', as_cmap=True)
            else:
                # For non-float features, use log-transformed values
                data = np.log1p(observations_by_months_and_year)
                annot = make_round(observations_by_months_and_year, roundby=0)
                fmt = '.0f'
                cmap = sns.color_palette('light:b', as_cmap=True)

            # Plot the heatmap
            plot_heatmap(title_=f'Median `{feature_name}` by {by}',
                         data_=data,
                         annot_=annot,
                         fmt_=fmt,
                         cmap_=cmap,
                         ax_=ax)

def plot_yes_proportion_by(by=None, df_=None, feature_name=None, fig=None, ax=None):
    """
    Plot the proportion of 'y' (yes) responses by specified feature and grouping.

    Parameters:
    - by: str, grouping variable ('year', 'month', or 'month and year')
    - df_: DataFrame, input data
    - feature_name: str, feature to analyze
    - fig: matplotlib figure object
    - ax: matplotlib axes object
    """

    # Error handling (assuming error_handling function exists)
    if error_handling('plot_yes_proportion_by', by, df_, feature_name, fig, ax) == -1:
        return

    # Define month order
    months_order = pd.Categorical(['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
                                 ordered=True)

    # Set order based on feature_name
    if feature_name == 'month':
        order = months_order
    else:
        order = pd.Categorical(df_[feature_name].value_counts().index, ordered=True)

    if feature_name:
        if by == 'year':
            # Calculate proportion by feature and year
            yes_proportion_by_year = df_.groupby([feature_name, 'year'])['y'].mean().unstack()

            # Reorder rows
            yes_proportion_by_year = yes_proportion_by_year.reindex(labels=order)

            # Prepare annotation
            annot = np.round(yes_proportion_by_year, 2)

            # Plot heatmap
            plot_heatmap(
                title_=f'YES-proportion by {feature_name}\'s categories and year',
                data_=yes_proportion_by_year,
                annot_=annot,
                fmt_='.2f',
                cmap_=sns.color_palette('light:b', as_cmap=True),
                ax_=ax
            )

        elif by == 'month':
            # Calculate proportion by feature and month
            yes_proportion_by_month = df_.groupby([feature_name, 'month'])['y'].mean().unstack()

            # Set row and column orders
            lbls_order = order
            cols_order = months_order

            # Reorder rows and columns
            yes_proportion_by_month = yes_proportion_by_month.reindex(
                labels=lbls_order,
                columns=cols_order
            )

            # Prepare annotation
            annot = np.round(yes_proportion_by_month, 2)
            fmt = '.2f'

            # Plot heatmap
            plot_heatmap(
                title_=f'YES-proportion by {feature_name}\'s categories and month',
                data_=yes_proportion_by_month,
                annot_=annot,
                fmt_=fmt,
                cmap_=sns.color_palette('light:b', as_cmap=True),
                ax_=ax
            )

        elif by == 'month and year':
            # Calculate proportion by month and year
            yes_proportion_by_month_and_year = df_.groupby(['month', 'year'])['y'].mean().unstack()

            # Reorder rows
            order = months_order
            yes_proportion_by_month_and_year = yes_proportion_by_month_and_year.reindex(labels=order)

            # Set colormap based on feature type
            if df_[feature_name].dtype == 'float':
                cmap = sns.color_palette('coolwarm', as_cmap=True)
            else:
                cmap = sns.color_palette('light:b', as_cmap=True)

            # Plot heatmap
            plot_heatmap(
                title_=f'YES-proportion by month and year',
                data_=yes_proportion_by_month_and_year,
                annot_=np.round(yes_proportion_by_month_and_year, 2),
                fmt_='.2f',
                cmap_=cmap,
                ax_=ax
                )

def plot_distribution(df_=None, feature_name=None, feature_type=None, num_gr=10, plot_options={}):
    """
    Plot the distribution of a feature in a DataFrame.

    Parameters:
    - df_: DataFrame containing the data
    - feature_name: Name of the feature to plot
    - feature_type: Type of the feature ('categorical' or 'numeric')
    - num_gr: Number of groups for categorical features (default: 10)
    - plot_options: Dictionary of plot options (default: {})
    """

    if feature_type == 'categorical':
        # Set up the figure and axis
        if 'fig' not in plot_options.keys():
            fig, ax = plt.subplots(figsize=(15, 3))
        else:
            fig = plot_options['fig']
            ax = plot_options['ax']

        # Define custom order for categorical features
        if feature_name != 'month':
            custom_order = pd.Categorical(df_[feature_name].value_counts(ascending=False).index,
                                          ordered=True)
        else:
            custom_order = pd.Categorical(['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
                                          ordered=True)

        # Plot the countplot
        sns.countplot(x=df_[feature_name],
                      order=custom_order,
                      ax=ax)

        # Add bar labels
        abs_values = df_[feature_name].value_counts(ascending=False)
        rel_values = []
        for i in (abs_values / df_.shape[0] * 100):
            if i >= 1:
                rel_values.append(f'{i:.0f}')
            else:
                rel_values.append(f'<1')

        lbls = [f'{p[0]} ({p[1]}%)' for p in zip(abs_values, rel_values)]
        ax.bar_label(container=ax.containers[0], labels=lbls)

        # Set titles and labels
        ax.set_title(f'`{feature_name}` distribution', fontsize=16, pad=20)
        ax.set_xlabel('')

        if 'fig' not in plot_options.keys() and 'ax' not in plot_options.keys():
            plt.tight_layout()

    elif feature_type == 'numeric':
        # Set up the figure and axes
        if 'fig' not in plot_options.keys() and 'axes' not in plot_options.keys():
            fig, axes = plt.subplots(2, 1,
                                     figsize=(15, 4),
                                     gridspec_kw={'height_ratios': [3, 0.5]},
                                     sharex=True)
        else:
            fig = plot_options['fig']
            axes = plot_options['axes']

        # Plot the histogram and KDE
        axes[0].set_title(f'`{feature_name}` distribution', fontsize=16, pad=20)
        sns.histplot(df_[feature_name], stat='density', lw=0, ax=axes[0])
        sns.kdeplot(df_[feature_name], color='#ffb500', lw=3, ax=axes[0])

        # Plot feature statistics
        _, mean_val, _, min_val, q25, q50, q75, max_val = np.round(df_[feature_name].describe(), 2)

        feature_stats_text = (f'min: {min_val}\n' + f'q25: {q25}\n' + f'q50: {q50}\n' +
                            f'q75: {q75}\n' + f'max: {max_val}\n' + f'mean: {mean_val}')

        x_coor = np.quantile(axes[0].get_xlim(), 0.85)
        y_coor = np.quantile(axes[0].get_ylim(), 0.2)

        font = {
            'color': 'black',
            'size': 14
        }

        box = {
            'facecolor': 'white',
            'edgecolor': 'grey',
            'alpha': 0.5,
            'boxstyle': 'round'
        }

        axes[0].text(x_coor, y_coor, feature_stats_text, fontdict=font, bbox=box)

        # Plot the boxplot
        sns.boxplot(x=df_[feature_name], ax=axes[1])

        plt.tight_layout()

        if 'fig' not in plot_options.keys():
            plt.tight_layout()

    else:
        print('plot_distribution(): Only `categorical` and `numeric` are acceptable.')
        fig.delaxes(ax)
        return


def plot_yes_proportion(df_=None, feature_name=None, feature_type=None, num_gr=10, plot_options={}):
    """
    Plot the proportion of 'yes' responses for a given feature.

    Parameters:
    - df_: DataFrame containing the data.
    - feature_name: Name of the feature to plot.
    - feature_type: Type of the feature ('categorical' or 'numeric').
    - num_gr: Number of groups for numeric features (default is 10).
    - plot_options: Dictionary containing plot options (e.g., 'fig' and 'ax').
    """

    # Set up the figure and axis
    if 'fig' not in plot_options.keys():
        fig, ax = plt.subplots(figsize=(15, 3))
    else:
        fig = plot_options['fig']
        ax = plot_options['ax']

    if feature_type == 'categorical':
        # Handle categorical features
        if feature_name != 'month':
            custom_order = pd.Categorical(df_[feature_name].value_counts(ascending=False).index, ordered=True)
        else:
            custom_order = pd.Categorical(['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], ordered=True)

        sns.barplot(x=df_[feature_name], y=df_['y'], order=custom_order, ci=95, errwidth=2, ax=ax)

        # Calculate and annotate the 'yes' proportion
        df_yes_proba = pd.crosstab(df_[feature_name], df_['y'])
        df_yes_proba = (df_yes_proba[1] / (df_yes_proba[0] + df_yes_proba[1])).reindex(custom_order)

        coors_x = np.arange(0, df_yes_proba.shape[0])
        for i in range(df_yes_proba.shape[0]):
            plt.annotate(f"{df_yes_proba[i]:.2f}", xy=(coors_x[i]-0.2, df_yes_proba[i]), ha='center', va='bottom')

        # Set titles and labels
        ax.set_title('YES-proportion', fontsize=16, pad=20)
        ax.set_ylabel('Proportion')

        if 'fig' not in plot_options.keys():
            plt.tight_layout()

    elif feature_type == 'numeric':
        # Handle numeric features
        n_unique = df_[feature_name].nunique()

        if num_gr > n_unique:
            num_gr = n_unique
            print(f'WARNING:\n'
                  f'The number of groups CANNOT be more than number of unique values ({n_unique}).\n'
                  f'It will be automatically reduced to that number.')
        else:
            num_gr += 1

        # Create bins for numeric features
        bins = np.linspace(start=df_[feature_name].min(), stop=df_[feature_name].max(), num=num_gr)
        bins_series = pd.cut(df_[feature_name], bins=bins, include_lowest=True)
        bins_series.name = f'{feature_name}_bins'

        df_bins_yes = pd.DataFrame(pd.concat([df_[feature_name], bins_series, df_['y']], axis=1))

        yes_answers = df_bins_yes.groupby(bins_series.name)['y'].agg(lambda y: y.eq(1).sum())
        bin_observations = df_bins_yes.groupby(bins_series.name)['y'].count()

        # Calculate the 'yes' proportion for each bin
        df_yes_proba = yes_answers / bin_observations
        df_yes_proba.name = 'mean'
        df_yes_proba = pd.DataFrame(df_yes_proba)

        # Plot the 'yes' proportion
        sns.barplot(x=df_bins_yes[f'{feature_name}_bins'], y=df_bins_yes['y'], ci=95, color='b', errwidth=2, ax=ax)

        coors_x = np.arange(0, df_yes_proba['mean'].shape[0])
        for i in range(df_yes_proba['mean'].shape[0]):
            plt.annotate(f'{df_yes_proba["mean"][i]:.2f}', xy=(coors_x[i]-0.2, df_yes_proba['mean'][i]), ha='center', va='bottom')

        # Set titles and labels
        ax.set_title('YES-proportion', fontsize=16, pad=20)
        ax.set_ylabel('Proportion')

        # Calculate and set x-axis labels
        abs_values = df_bins_yes.groupby(bins_series.name)['y'].count()
        rel_values = []
        for i in (abs_values / df_.shape[0] * 100):
            if i >= 0.5:
                rel_values.append(f'{i:.0f}')
            else:
                rel_values.append(f'<0.5')

        bin_dtype = 'int' if df_[feature_name].dtype == 'int' else 'float'

        if bin_dtype == 'int':
            x_lbls = [f'({int(i.left)}..{int(i.right)}]\n-------\n{j} ({k}%)'
                      for i, j, k in zip(df_yes_proba.index, abs_values, rel_values)]
        else:
            x_lbls = [f'({i.left:.2f}..{i.right:.2f}]\n-------\n{j} ({k}%)'
                      for i, j, k in zip(df_yes_proba.index, abs_values, rel_values)]
        ax.set_xticklabels(x_lbls)

        if 'fig' not in plot_options.keys():
            plt.tight_layout()

    else:
        print('plot_yes_proportion(): Only `categorical` and `numeric` are acceptable.')
        fig.delaxes(ax)
        return
