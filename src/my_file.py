import src.my_library as my_library

pd, np, plt, ticker, sns = my_library.make_inital_imports()

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
                              plot_options = {'fig', fig, 'ax': ax[0]})
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
    