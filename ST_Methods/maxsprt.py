import sys
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql.functions import (count, col, when, lit, sum, countDistinct, signum, max, abs, log, rand)
from pyspark.sql.window import Window
from tqdm import tqdm

import plotly.graph_objects as go

from ST_Methods.interleaved_sequential_test import InterleavingST


class MaxSPRT(InterleavingST):

    def __init__(self, sid_columns, grouping_column, treatment_variant, control_variant, item_id_column="item_id",
                 variant_column="variant"):
        """
        Initializes the MaxSPRT object with session identifiers and grouping columns.

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

        # Placeholder for the threshold value for OBF test.
        self.threshold = None

        # Placeholder for transformed/prepared data for OBF test.
        self.transformed_interleaved_dataset = None

    def _prep_for_interleaving(self, data_set):
        # TODO: provide the datatype of the fields of your dfs in docstring
        """
        Prepares the input DataFrame for interleaving simulations and threshold calculation.

        Args:
            data_set (DataFrame): Spark DataFrame containing data for interleaving simulations. Contains:
                - sid_columns (list of str or numeric): Session identifier columns.
                - grouping_column (numeric, str, or datetime): Column for grouping data.
                - item_id_column (str): Item identifier column.

        Returns:
            Pandas DataFrame: DataFrame with prepared data for interleaving simulations. Contains:
                - sid_columns (list of str or numeric): Session identifier columns.
                - grouping_column (numeric, str, or datetime): Grouping column.
                - a_1_clicks (int): Count of clicks for variant a_1.
                - a_2_clicks (int): Count of clicks for variant a_2.
        """
        # TODO: too heavy commenting inside the method. Big chunks explaining the method functionality should
        #  be moved to the docstring. Comments should be not more than one line length. Alternatively code chunks that
        #  require long comments can be abstracted into separate private methods with comments placed into docstring

        # Select columns and assign variants.
        data_with_supersessions = data_set.select(*self.sid, self.grouping, self.item_id)
        data_with_supersessions = data_with_supersessions \
            .withColumn("variant", when(rand() < 0.5, "a_1").otherwise("a_2")) \
            .groupBy(*self.sid, "variant") \
            .agg(max(self.grouping).alias(self.grouping), countDistinct(self.item_id).alias("clicks"))

        # Separate data into two datasets based on variant.
        a_1 = data_with_supersessions.where(col("variant") == "a_1") \
            .select(*self.sid, self.grouping, col("clicks").alias("a_1_clicks"))
        a_2 = data_with_supersessions.where(col("variant") == "a_2") \
            .select(*self.sid, self.grouping, col("clicks").alias("a_2_clicks"))

        # Merge datasets and handle missing values, if any.
        data_with_supersessions_aa = a_1.join(a_2, [*self.sid, self.grouping], "outer") \
            .fillna(0, subset=["a_1_clicks", "a_2_clicks"])

        # Convert Spark DataFrame to Pandas DataFrame.
        data_aggregated = data_with_supersessions_aa.toPandas()

        return data_aggregated

    def calculate_threshold(self, data_set, num_experiments, number_of_measurements=42, alpha=0.05,
                            plot_distribution=True):
        """
        Calculates the L_i statistic threshold for sequential testing.

        Args:
            # TODO  data types
            data_set (Spark DataFrame): DataFrame containing data for interleaving simulations. Contains:
                - sid_columns (list of str or numeric): Session identifier columns.
                - grouping_column (numeric, str, or datetime): A column used for grouping data.
                - item_id_column (str): Item identifier column.
            num_experiments (int): Number of experiments to run.
            number_of_measurements (int, optional): Number of groupings for each experiment.
                Defaults to 42.
            alpha (float, optional): Significance level, typically in the range (0, 1).
                Defaults to 0.05.
            plot_distribution (bool, optional): If True, a histogram of the calculated L_i values will be plotted.
                Defaults to True.

        Returns:
            float: 1-alpha percentile of the calculated L_i thresholds. This value can be used as the threshold in
                subsequent MaxSPRT tests to control the type I error rate to be less than or equal to alpha.
        """
        # Initialize an empty NumPy array to store L_i thresholds from each experiment.
        thresholds = np.array([])

        # Run the specified number of experiments.
        for _ in tqdm(range(num_experiments)):
            # Prepare data for interleaving experiments.
            aa_exp = self._prep_for_interleaving(data_set=data_set)

            # Randomly sample the specified number of measurements from the prepared data.
            aa_exp = aa_exp.sample(n=number_of_measurements, replace=False)

            # Assign scores based on comparison of a_1_clicks and a_2_clicks.
            aa_exp["score"] = np.where(aa_exp["a_1_clicks"] > aa_exp["a_2_clicks"], 1, -1)

            # TODO: Shall we format uppercase variable? In the paper they are uppercase. Thus, I decided to keep them
            #  uppercase also in the codes, for less confusion. But this is aggainst PEP8

            # Calculate single-measurement versions of T_i and m_i.
            T_i_sngl = aa_exp["score"].abs().sum() + aa_exp["score"].eq(0).sum()
            m_i_sngl = np.where(aa_exp["score"] == 1, 1, 0).sum() + aa_exp["score"].eq(0).sum() / 2

            # Cumulatively sum the values to get the running totals of T_i and m_i.
            T_i = T_i_sngl.cumsum()
            m_i = m_i_sngl.cumsum()

            # Calculate p_1i, the observed proportion of 1s at each step.
            p_1i = m_i / T_i

            # Compute the components of the L_i statistic.
            numerator_1 = m_i * np.log2(p_1i)
            numerator_2 = (T_i - m_i) * np.log2(1 - p_1i)
            denominator = T_i

            # Calculate the L_i statistic.
            L_i = numerator_1 + numerator_2 + denominator

            # Append L_i values to the thresholds array.
            thresholds = np.append(thresholds, L_i)

        # If plot_distribution is True, plot the histogram of L_i values.
        if plot_distribution:
            self._plot_Li_thresholds(thresholds=thresholds)

        # Calculate and return the (1 - alpha) percentile of the L_i thresholds.
        percentile_value = np.quantile(thresholds, 1 - alpha)

        return percentile_value

    def transform(self, interleaved_dataset):
        # TODO: mention dataframe type and field datatypes
        # TODO: Respect PEP and move overly long docstrings to the next line
        """
        Transforms the dataset for running the MaxSPRT test.

        Args:
            interleaved_dataset (DataFrame): Spark DataFrame of the interleaving experiment. Contains:
                - sid_columns (list of str or numeric): Session identifier columns.
                - grouping_column (numeric, str, or datetime): A column used for grouping data.
                - item_id_column (str): Item identifier column.
                - variant_column (str): Variant column.

        Returns:
            Spark DataFrame: Transformed  DataFrame for MaxSPRT test. Contains:
                - grouping_column (numeric, str, or datetime): A column used for grouping data.
                - sid_columns (list of str or numeric): Session identifier columns.
                - score (int): Calculated as 1, -1, or 0 based on the comparison of clicks between
                    treatment and control.
                - tr_clicks (int): Number of clicks for the treatment group.
                - cntrl_clicks (int): Number of clicks for the control group.

        The same DataFrame is also stored as an instance attribute: `self.transformed_interleaved_dataset`.
        """
        # Group the dataset by variant, session identifiers, and the grouping column, and count distinct item clicks.
        grouped_interleaved_dataset = interleaved_dataset.select(*self.sid, self.grouping, self.item_id, self.variant) \
            .groupBy(self.variant, *self.sid, self.grouping).agg(countDistinct(self.item_id).alias("clicks"))

        # Separating data into treatment and control datasets.
        tr_dataset = grouped_interleaved_dataset.where(col(self.variant) == self.tr) \
            .withColumnRenamed("clicks", "tr_clicks").drop(self.variant)
        cntrl_dataset = grouped_interleaved_dataset.where(col(self.variant) == self.ctr) \
            .withColumnRenamed("clicks", "cntrl_clicks").drop(self.variant)

        # Merging datasets on session id and grouping columns, and filling N/A values with zeros.
        interleaving_exp = tr_dataset.join(cntrl_dataset, [*self.sid, self.grouping], "outer").fillna(0, subset=[
            "tr_clicks", "cntrl_clicks"])

        # Computing the score based on clicks.
        interleaving_exp = interleaving_exp.withColumn(
            "score",
            when(col("tr_clicks") > col("cntrl_clicks"), lit(1))  # B
            .when(col("tr_clicks") < col("cntrl_clicks"), lit(-1))  # A
            .otherwise(lit(0)),
        )

        # Store the transformed dataset as an attribute of the class for use in subsequent methods.
        self.transformed_interleaved_dataset = interleaving_exp

        return interleaving_exp

    def run_test(self, L_i_threshold):
        """
        Runs the MaxSPRT test and calculates the L_i statistic.

        Args:
            L_i_threshold (float): Threshold for the L_i statistic.

        Returns:
            Spark DataFrame: DataFrame with calculated MaxSPRT statistics. Contains:
                - grouping_column (numeric, str, or datetime): A column used for grouping data.
                - T_i_sngl (int): Count of rows per group.
                - m_i_sngl (int): Sum of scores per group.
                - T_i (int): Cumulative sum of |score| + count(score=0), per group.
                - m_i (int): Cumulative sum of score=1 + (count(score=0))/2, per group.
                - p_1i (float): m_i / T_i.
                - L_i (float): Log-likelihood ratio statistic, sign determined by (p_1i - 0.5).
                - threshold (float): User-specified L_i threshold.
                - can_stop (bool): True if |L_i| >= threshold, False otherwise.
        """

        # Retrieve the transformed interleaved dataset, prepared previously using the `transform` method.
        interleaving_exp = self.transformed_interleaved_dataset

        # Window specification for cumulative operations.
        window_spec = Window().partitionBy().orderBy(self.grouping).rowsBetween(-sys.maxsize, 0)

        # Compute metrics using grouped aggregations and window functions.
        interleaving_exp = (
            interleaving_exp.groupBy(self.grouping)
            .agg(
                # - T_i_sngl: the sum of |score| plus the count of scores equal to 0.
                (sum(abs(col("score"))) + count(when(col("score") == 0, True))).alias("T_i_sngl"),
                # - m_i_sngl: the sum of scores equal to 1 plus half the count of scores equal to 0
                (sum(when(col("score") == 1, 1).otherwise(0)) + count(when(col("score") == 0, True)) / 2)
                .alias("m_i_sngl")
            )
            # Cumulative sum of T_i_sngl across groups.
            .withColumn("T_i", sum("T_i_sngl").over(window_spec))
            # Cumulative sum of m_i_sngl across groups.
            .withColumn("m_i", sum("m_i_sngl").over(window_spec))
            # Compute p_1i as m_i divided by T_i for each group.
            .withColumn("p_1i", col("m_i") / col("T_i"))
            # Compute components of the L_i statistic, involving log-likelihood calculations.
            .withColumn("numerator_1", col("m_i") * log(2.0, col("p_1i")))
            .withColumn("numerator_2", (col("T_i") - col("m_i")) * log(2.0, 1 - col("p_1i")))
            .withColumn("denominator", col("T_i"))
            # Compute L_i statistic and assign its direction based on the sign of (p_1i - 0.5).
            .withColumn("L_i", col("numerator_1") + col("numerator_2") + col("denominator"))
            .withColumn("L_i", signum(col("p_1i") - 0.5) * col("L_i"))
            # Add the user-defined L_i threshold for comparison and compute whether the experiment can stop.
            .withColumn("threshold", lit(L_i_threshold))
            .withColumn("can_stop", col("threshold") <= abs(col("L_i")))
        )

        # Order the resulting DataFrame by the grouping column and return.
        return interleaving_exp.orderBy(self.grouping)

    def visualize(self, df):
        """
        Creates an interactive plot of the output from the run_test method. Includes L_i, positive threshold, and
        negative threshold on the y-axis and 'grouping' on the x-axis.

        Parameters:
        df (Pyspark DataFrame): DataFrame output from the run_test method. Contains:
            - grouping_column (numeric, str, or datetime): A column used for grouping data.
            - L_i (float): Log-likelihood ratio statistic
            - threshold (float): Threshold of the L_i statistic.
        """

        # Convert the Pyspark DataFrame to a Pandas DataFrame and sort it by the grouping column.
        df = df.toPandas().sort_values(self.grouping)

        # Create a trace for L_i: a line that represents the L_i statistic across different groups.
        trace1 = go.Scatter(
            x=df[self.grouping],
            y=df['L_i'],
            mode='lines',
            name='L_i'
        )

        # Create a trace for the positive threshold: a dotted line representing the upper bound for L_i statistic.
        trace2 = go.Scatter(
            x=df[self.grouping],
            y=df['threshold'],
            mode='lines',
            name='positive threshold',
            line=dict(color='green', width=2, dash='dot')
        )

        # Create a trace for the negative threshold: a dotted line representing the lower bound for L_i statistic.
        trace3 = go.Scatter(
            x=df[self.grouping],
            # Negative threshold.
            y=-df['threshold'],
            mode='lines',
            name='negative threshold',
            line=dict(color='red', width=2, dash='dot')
        )

        # Define the layout of the plot, specifying titles, axis labels, grid colors, and font settings.
        layout = go.Layout(
            title='MaxSPRT Visualisation',
            xaxis=dict(title=self.grouping, gridcolor='lightgray'),
            yaxis=dict(title='Values', gridcolor='lightgray'),
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )

        # Create the figure by combining the traces and layout, then display the plot.
        fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
        fig.show()

    def _plot_Li_thresholds(self, thresholds, alpha=0.05):
        """
        Plots a histogram of L_i thresholds and calculates the 1-alpha confidence threshold. This value is used as a
        critical value in the MaxSPRT to decide whether to stop the test or continue.

        Args:
            thresholds (numpy.ndarray): A NumPy array of calculated L_i thresholds.
            alpha (float, optional): The significance level of the confidence interval. Defaults to 0.05.
        """

        # Set the style of the plot to 'seaborn-darkgrid' for better visualization.
        plt.style.use('seaborn-darkgrid')

        # The histogram is divided into 50 bins and is displayed with a teal color and 50% transparency.
        plt.hist(thresholds, bins=50, color='#008080', alpha=0.5)

        # Title and labels are added to the plot for better interpretation of the visualized data.
        plt.title('Histogram of L_i Thresholds')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        # Adding a grid to the plot for better readability of the values.
        plt.grid(True)

        # Calculate the L_i threshold which is the (1-alpha) quantile of the provided thresholds.
        L_i_threshold = np.quantile(thresholds, 1 - alpha)

        # Output the number of A/A test iterations and the calculated (1-alpha) quantile of L_i thresholds.
        print(f"Number of A/A experiment iterations: {thresholds.size}")
        print(f"The {100 * (1 - alpha)}% confidence threshold is: {np.round(L_i_threshold, 5)}")

        # Add a vertical red line to the histogram that indicates the (1-alpha) quantile of the L_i thresholds.
        plt.axvline(x=L_i_threshold, color='red')

        # Display the plot.
        plt.show()
