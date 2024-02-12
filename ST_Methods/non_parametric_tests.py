import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import permutation_test, bootstrap

from ST_Methods.interleaved_sequential_test import InterleavingST


class NonParametricTests(InterleavingST):
    """
    Class for running non-parametric statistical tests of the scipy package on interleaved datasets.
    Included tests: permutation test, bootstrap test
    """

    def calculate_threshold(self):
        """Dummy Method"""
        raise NotImplementedError("This method must not be used in NonParametricTests class.")

    def __init__(self, sid_columns, grouping_column, treatment_variant, control_variant, item_id_column="item_id",
                 variant_column="variant"):
        """
        Initialize the NonParametricTests class.

        Args:
            sid_columns (list of str or numeric): Session identifier columns.
            grouping_column (numeric, str, or datetime): Column for grouping data.
            treatment_variant (str): Identifier for the treatment group variant.
            control_variant (str): Identifier for the control group variant.
            item_id_column (str, optional): Item identifier column. Defaults to "item_id".
            variant_column (str, optional): Variant column. Defaults to "variant".
        """
        self.sid = sid_columns
        self.grouping = grouping_column
        self.tr = treatment_variant
        self.ctr = control_variant
        self.variant = variant_column
        self.item_id = item_id_column

        # Placeholder for the treatment group data after transformation.
        self.tr_data = None
        # Placeholder for the ctr group data after transformation.
        self.ctr_data = None

    def calculate_threshold(self):
        pass

    def transform(self, interleaved_dataset):
        """
        Transforms the data for non-parametric tests.

        Args:
            interleaved_dataset (DataFrame): Spark DataFrame of the interleaving experiment. Contains:
                - sid_columns (list of str or numeric): Session identifier columns.
                - grouping_column (numeric, str, or datetime): A column used for grouping data.
                - item_id_column (str): Item identifier column.
                - variant_column (str): Variant column.
        """
        # Group the data by session identifiers and variant, then count the occurrences and convert to Pandas DataFrame.
        aggregated_data = interleaved_dataset.groupBy(*self.sid, self.variant).count().toPandas()

        # Extract and store the count of occurrences for the control variant.
        self.ctr_data = aggregated_data[aggregated_data[self.variant] == self.ctr][['count']]

        # Extract and store the count of occurrences for the treatment variant.
        self.tr_data = aggregated_data[aggregated_data[self.variant] == self.tr][['count']]

    def run_test(self, test_type='permutation', n_resamples=10 ** 4, **kwargs):
        """
        Runs the specified non-parametric test.

        Args:
            test_type (str, optional): The type of test to run. Either 'permutation' or 'bootstrap'.
                Default is 'permutation'.
            n_resamples (int, optional): Number of resamples for the test. Default is 10**4.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Union[float, bool]]: Test results including p-value, confidence interval, and other metrics.
        """
        # Check the specified test type and run the corresponding test method.
        if test_type == 'permutation':
            # If test_type is 'permutation', run the permutation test with the specified parameters.
            return self._run_permutation_test(n_resamples=n_resamples, **kwargs)
        elif test_type == 'bootstrap':
            # If test_type is 'bootstrap', run the bootstrap test with the specified parameters.
            return self._run_bootstrap_test(n_resamples=n_resamples, **kwargs)
        else:
            # If the test_type is neither 'permutation' nor 'bootstrap', raise a ValueError.
            raise ValueError("Invalid test_type. Choose either 'permutation' or 'bootstrap'")

    def _run_permutation_test(self, n_resamples=10 ** 4, batch=None, alternative='two-sided', alpha=0.05):
        """
        Runs a permutation test on the data.

        Args:
            n_resamples (int, optional): Number of resamples for the permutation test.
                Default is 10**4.
            batch (int, optional): Batch size for the test. If None, the max size of the variants is used.
                Default is None.
            alternative (str, optional): The alternative hypothesis.
                Default is 'two-sided'.
            alpha (float, optional): Significance level for confidence interval.
                Default is 0.05.

        Returns:
            Dict[str, Union[ndarray, float, bool, Tuple[float, float]]]: A dictionary with test results
                - 'null_distribution' (np.ndarray): The values of the test statistic generated under the null
                    hypothesis.
                - 'pvalue' (float or ndarray): The p-value for the given alternative.
                - 'statistic' (float or ndarray): The observed test statistic(s) computed from the data.
                - 'within_ci' (bool): Indicates whether statistic is within the confidence interlval
                - 'confidence_interval' (Tuple[float, float]): The lower and upper bounds of the confidence interval.
        """
        # If batch is not provided, set it as the maximum size between the treatment and control data.
        if batch is None:
            batch = np.max([self.tr_data.size, self.ctr_data.size])

        # Determine the percentiles for confidence interval calculation based on the alternative hypothesis.
        if alternative == 'two-sided':
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
        elif alternative == 'greater':
            lower_percentile = alpha * 100
            upper_percentile = 100
        elif alternative == 'less':
            lower_percentile = 0
            upper_percentile = (1 - alpha) * 100
        else:
            # Raise an error if an unsupported alternative hypothesis is provided.
            raise ValueError("Invalid alternative. Choose either 'two-sided', 'greater', or 'less'")

        # Execute the permutation test with the specified parameters and statistic function.
        res_permutation = permutation_test((self.tr_data, self.ctr_data),
                                           statistic=lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis),
                                           n_resamples=n_resamples, batch=batch, vectorized=True,
                                           alternative=alternative)

        # Calculate the confidence interval based on the percentiles determined above.
        low, high = np.percentile(res_permutation.null_distribution, [lower_percentile, upper_percentile])

        # Check if the observed statistic falls within the confidence interval.
        within_ci = (low < res_permutation.statistic[0]) & (res_permutation.statistic[0] < high)

        # Parse the result of the permutation test into a dictionary format.
        stats = self._parse_test_result(res_permutation)

        # Add additional metrics to the parsed results.
        stats["within_ci"] = within_ci
        stats['confidence_interval'] = (low, high)

        return stats

    def _run_bootstrap_test(self, n_resamples=10 ** 5, batch=None):
        """
        Runs a bootstrap test on the data.

        Args:
            n_resamples (int, optional): Number of resamples for the bootstrap test. Default is 10**5.
            batch (int, optional): Batch size for the test. If None, the max size of the variants is used.
                Default is None.

        Returns:
            Dict[str, Union[ndarray, float, ConfidenceInterval]]: A dictionary with test results
                - bootstrap_distribution (np.ndarray): The bootstrap distribution of the test statistic,
                    generated by resampling the data.
                - confidence_interval (ConfidenceInterval): An object containing the lower and upper bounds
                    of the confidence interval.
                - standard_error (np.ndarray): The standard error of the bootstrap distribution.
                - statistic (float): The observed test statistic computed from the data.
        """
        # If batch is not provided, set it as the maximum size between the treatment and control data.
        if batch is None:
            batch = np.max([self.tr_data.size, self.ctr_data.size])

        # Define the statistic function: the difference in means between two data sets.
        def statistic(x, y, axis):
            return np.mean(x, axis=axis) - np.mean(y, axis=axis)

        # Execute the bootstrap test with specified parameters and statistic function.
        res_bootstrap = bootstrap((self.tr_data, self.ctr_data),
                                  statistic=statistic,
                                  n_resamples=n_resamples, batch=batch, vectorized=True)

        # Calculate the observed statistic value (without resampling).
        observed_statistic = statistic(self.tr_data, self.ctr_data, axis=0)

        # Parse the result of the bootstrap test into a dictionary format.
        stats = self._parse_test_result(res_bootstrap)

        # Add the observed statistic to the parsed results.
        stats["statistic"] = observed_statistic[0]

        return stats

    def visualize(self, results, test_type='permutation', plot_size=(12, 8), bins=100):
        """
        Generates histograms for the test results based on the test_type.

        Args:
            results (Dict[str, Any]): The test results.
            test_type (str): The type of test, either 'permutation' or 'bootstrap'.
            plot_size (tuple, optional): Size of the plot. Default is (12, 8).
            bins (int, optional): Number of histogram bins. Default is 100.
        """
        # Create a figure and a set of subplots with specified plot size.
        fig, ax = plt.subplots(figsize=plot_size)

        # Check the specified test type and plot the corresponding histogram.
        if test_type == "permutation":
            # If test_type is 'permutation', plot a histogram of the null_distribution from the results.
            ax.hist(results["null_distribution"], bins=bins)
        elif test_type == "bootstrap":
            # If test_type is 'bootstrap', plot a histogram of the bootstrap_distribution from the results.
            ax.hist(results["bootstrap_distribution"][0], bins=bins)
        else:
            # If the test_type is neither 'permutation' nor 'bootstrap', raise a ValueError.
            raise ValueError("Invalid test_type. Choose either 'permutation' or 'bootstrap'")

        # Add a vertical dashed line at the location of the observed statistic.
        plt.axvline(results["statistic"], color='r', linestyle='dashed', linewidth=2, label='Metric Value')

        # Set the title, x-axis label, and y-axis label of the plot.
        ax.set_title(f'{test_type.capitalize()} Distribution Histogram of Mean Statistic')
        ax.set_xlabel('Value of Statistic')
        ax.set_ylabel('Frequency')

        # Add a legend to the plot.
        plt.legend()

        # Display the plot.
        plt.show()

    def _parse_test_result(self, test_result):
        """
        Parses the result of a statistical test and returns a dictionary.

        Args:
            test_result: The result object from the test.

        Returns:
            Dict[str, Any]: A dictionary containing parsed results, where keys are string identifiers of the results
                and values could be various types including float, numpy arrays, etc.
        """
        # Initialize the result dictionary.
        parsed_result = {}

        # Try to extract all attributes of the test result object.
        for attr in dir(test_result):
            # Ensure that we're not accessing magic methods/attributes.
            if not attr.startswith("__"):
                try:
                    value = getattr(test_result, attr)
                    # Add to dictionary if it's not a method.
                    if not callable(value):
                        parsed_result[attr] = value
                except Exception as e:
                    print(f"Failed to get attribute {attr} from test result: {str(e)}")

        return parsed_result
