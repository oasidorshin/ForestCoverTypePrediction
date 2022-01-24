# Taken from https://gist.github.com/marnixkoops/e68815d30474786e2b293682ed7cdb01
def smoothing_target_encoder(df, column, target, weight=100):
    """
    Target-based encoding is numerization of a categorical variables via the target variable. This replaces the
    categorical variable with just one new numerical variable. Each category or level of the categorical variable
    is represented by it's summary statistic of the target. Main purpose is to deal with high cardinality categorical
    features.
    Smoothing adds the requirement that there must be at least m values for the sample mean to replace the global mean.
    Source: https://www.wikiwand.com/en/Additive_smoothing
    Args:
        df (pandas df): Pandas DataFrame containing the categorical column and target.
        column (string): Categorical variable column to be encoded.
        target (string): Target on which to encode.
        method (string): Summary statistic of the target.
        weight (int): Weight of the overall mean.
    Returns:
        array: Encoded categorical variable column.
    """
    # Compute the global mean
    mean = df[target].mean()

    # Compute the number of values and the mean of each group
    agg = df.groupby(column)[target].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the 'smoothed' means
    smooth = (counts * means + weight * mean) / (counts + weight)

    # Replace each value by the according smoothed mean
    return df[column].map(smooth)