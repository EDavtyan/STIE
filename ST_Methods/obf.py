import sys
import numpy as np
from pyspark.sql.functions import (count, col, when, lit, sum, variance, first, desc, signum, row_number, max, abs)
from pyspark.sql.window import Window
from tqdm import tqdm

import plotly.graph_objects as go

from ST_Methods.interleaved_sequential_test import InterleavingST


class OBF(InterleavingST):

    def __init__(self, sid_columns, grouping_column, treatment_variant, control_variant, item_id_column="item_id",
                 variant_column="variant"):
        """
        Initializes the OBF object with necessary column names and variant identifiers.

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

    def transform(self, interleaved_dataset):
        """
        Transforms the dataset for running the OBF test.

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
        # Grouping by variant, session id columns, and grouping column, then counting clicks.
        grouped_interleaved_dataset = interleaved_dataset.select(*self.sid, self.grouping, self.item_id, self.variant) \
            .groupBy(self.variant, *self.sid, self.grouping) \
            .agg(count(self.item_id).alias("clicks"))

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
            .otherwise(lit(0)),  # Equal
        )

        # Store the transformed dataset as an attribute of the class for use in subsequent methods.
        self.transformed_interleaved_dataset = interleaving_exp

        return interleaving_exp

    def calculate_threshold(self, number_of_measurements, iterations, alpha=0.05):
        """
        Calculates the threshold for the OBF test using Monte Carlo simulations.

        Monte Carlo simulations are used to generate a distribution of maximum sum
        of squares values. This involves running a set number of random experiments
        (defined by `iterations`) and observing the distribution of a certain statistic
        (here, the maximum squared sum of random normal variables).

        A critical value (threshold) is identified from the distribution of max sum of
        squares values. This is the (1 - alpha) percentile of the distribution.

        Parameters:
            number_of_measurements (int): Number of measurements in each simulation.
            iterations (int): Number of Monte Carlo iterations.
            alpha (float): Significance level, in the range (0, 1).

        Returns:
            float: The calculated threshold. This value will be used in subsequent
                   testing as a threshold for OBF test.
        """
        # Running Monte Carlo simulations to generate max sum of squares values.
        thresholds = self._monte_carlo_simulation(number_of_measurements, iterations)

        # Identifying and returning the (1 - alpha) percentile as the threshold.
        return np.percentile(thresholds, 100 * (1 - alpha))

    def run_test(self, O_i_threshold):
        """
        Runs the OBF test and calculates the O_i statistics.

        This method retrieves the transformed interleaved dataset and applies a
        series of transformations to calculate intermediate and final statistics
        for the OBF test. This includes computing the variance of the score,
        aggregating the data by grouping column, calculating various metrics
        per group, and more.

        Args:
            O_i_threshold (float): Threshold for the O_i statistic.

        Returns:
            DataFrame: Resulting DataFrame with calculated O_i statistics.
                Result DataFrame Columns:
                - grouping_column (str, numeric, or datetime): The column used for grouping data.
                - T_i_sngl (int): Count of rows per group.
                - m_i_sngl (int): Sum of scores per group.
                - D_i (float): Variance of the score.
                - i (int): Row number based on the order of the grouping column.
                - T_i (int): Cumulative count of scores up to the current group.
                - m_i (int): Cumulative sum of scores up to the current group.
                - m_i_pow2 (int): Square of m_i.
                - O_i (float): The OBF statistic for each group.
                - O_i*i (float): O_i multiplied by i.
                - threshold (float): The provided O_i_threshold replicated across rows.
                - can_stop (bool): Boolean flag indicating whether the experiment can be stopped early.
        """
        # Retrieve the transformed interleaved dataset from the instance attribute.
        interleaving_exp = self.transformed_interleaved_dataset

        # Define a window specification for cumulative operations, ordering by the grouping column.
        window_spec = Window().partitionBy().orderBy(self.grouping).rowsBetween(-sys.maxsize, 0)

        # Series of transformations to calculate intermediate and final statistics for the OBF test.
        interleaving_exp = (interleaving_exp
                            # Create a row number based on the order of the grouping column.
                            .withColumn("row_number", row_number().over(Window()
                                                                        .partitionBy(self.grouping)
                                                                        .orderBy(self.grouping)))
                            # Compute the variance of the score, considering all previous rows.
                            .withColumn('D_i', variance(col('score')).over(Window().partitionBy()
                                                                           .orderBy("row_number")
                                                                           .rowsBetween(-sys.maxsize, 0)))
                            # Carry the first (most recent) D_i value down the column.
                            .withColumn('D_i', first("D_i").over(Window()
                                                                 .partitionBy(self.grouping)
                                                                 .orderBy(desc("row_number"))))
                            # Aggregate the data by grouping column, calculating various metrics per group.
                            .groupBy(self.grouping).agg(
                                                        # - T_i_sngl: Count of rows per group.
                                                        count("*").alias("T_i_sngl"),
                                                        # - m_i_sngl: Sum of scores per group.
                                                        sum(col("score")).alias("m_i_sngl"),
                                                        # - D_i: The computed variance for each group.
                                                        max(col("D_i")).alias("D_i")
                                                        )
                            # Compute additional metrics and the O_i statistic for each group using window functions.
                            .withColumn("i", row_number().over(window_spec))
                            .withColumn("T_i", sum("T_i_sngl").over(window_spec))
                            .withColumn("m_i", sum("m_i_sngl").over(window_spec))
                            .withColumn("m_i_pow2", col("m_i") ** 2)
                            .withColumn('O_i', col('m_i_pow2') / (col('T_i') * col('D_i')))
                            .withColumn('O_i*i', col('O_i') * col("i"))
                            # Multiply 'O_i*i' by the sign of 'm_i' to ensure 'O_i*i' takes the sign of 'm_i'.
                            .withColumn('O_i*i', signum("m_i") * col('O_i*i'))
                            # Add the provided O_i*i threshold to each row and check if the experiment can stop.
                            .withColumn("threshold", lit(O_i_threshold))
                            .withColumn("can_stop", col("threshold") <= abs(col("O_i*i")))
                            )

        return interleaving_exp

    def visualize(self, df):
        """
        Creates an interactive plot of the output of the run_test function. Includes O_i*i, positive threshold, and
        negative threshold on the y-axis and 'grouping' on the x-axis.

        Parameters:
            df (Pyspark DataFrame): Output from the run_test function, which includes:
                - grouping_column (str, numeric, or datetime): The column used for grouping data.
                - O_i*i (float): The OBF statistic for each group.
                - threshold (float): The provided O_i_threshold replicated across rows.
        """

        # Convert Spark DataFrame to Pandas DataFrame and sort by the grouping column.
        df = df.toPandas().sort_values(self.grouping)

        # Create Plotly traces for the O_i*i statistic, O_i threshold, and negative O_i threshold.
        trace_oi = go.Scatter(
            x=df[self.grouping],
            y=df['O_i*i'],
            mode='lines',
            name='O_i*i',
            line=dict(color='royalblue', width=2)
        )

        # Create a trace for the positive threshold: a dotted line representing the upper bound for O_i*i statistic.
        trace_threshold = go.Scatter(
            x=df[self.grouping],
            y=df['threshold'],
            mode='lines',
            name='positive threshold',
            line=dict(color='green', width=2, dash='dot')
        )

        # Create a trace for the negative threshold: a dotted line representing the lower bound for O_i*i statistic.
        trace_neg_threshold = go.Scatter(
            x=df[self.grouping],
            # Negative threshold.
            y=-df['threshold'],
            mode='lines',
            name='negative threshold',
            line=dict(color='red', width=2, dash='dot')
        )

        # Define the layout of the plot, including titles, axis labels, and theming.
        layout = go.Layout(
            title='OBF Visualisation',
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

        # Combine the traces into a single figure and apply the layout.
        fig = go.Figure(data=[trace_oi, trace_threshold, trace_neg_threshold], layout=layout)

        # Display the interactive plot.
        fig.show()

    def _monte_carlo_simulation(self, number_of_measurements, iterations):
        """
        Perform Monte Carlo simulation to generate max sum of squares values
        for the OBF test. The simulation involves performing a specified number
        of iterations, each of which involves drawing a series of random values
        from a standard normal distribution, cumulatively summing them, squaring
        the resultant values, and finding the maximum value in the series. This
        maximum value is recorded for each iteration, resulting in a distribution
        of max sum of squares values, which is returned.

        Parameters:
            number_of_measurements (int): Number of random values drawn in each simulation.
            iterations (int): Number of Monte Carlo iterations to perform.

        Returns:
            max_sum_squares (list of float): List containing the maximum squared cumulative
                                             sum value from each iteration.
        """

        # TODO: move to docstring
        max_sum_squares = [
            np.max(np.square(np.cumsum(np.random.standard_normal(number_of_measurements))))
            for _ in tqdm(range(iterations))
        ]

        return max_sum_squares
