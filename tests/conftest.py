import pytest
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np


# Initialize a Spark session
@pytest.fixture(scope="session")
def spark():
    spark = SparkSession.builder.appName("TestApp").getOrCreate()
    yield spark
    spark.stop()


# Fixture for creating datasets
@pytest.fixture
def simulate_dataset(spark, request):
    is_interleaved = request.param  # obtain the parameterized value

    # Fixing the random seed for reproducibility
    np.random.seed(42)
    # Parameters
    start_time = pd.Timestamp("2023-06-05 06:00:00")
    end_time = pd.Timestamp("2023-06-05 23:00:00")
    mean_rows_per_hour = 1150
    stddev_rows_per_hour = 240
    unique_sids = 2000  # Number of unique sid_1 and sid_2 identifiers
    p_A = 0.45  # Probability of selecting variant A

    # Generate consecutive hours
    hours = pd.date_range(start=start_time, end=end_time, freq='H')

    # Generate unique sid_1 and sid_2 identifiers
    sid_1 = np.random.randint(100, 1000, size=unique_sids)
    sid_2 = np.random.randint(1000, 2000, size=unique_sids)

    # Initialize DataFrame
    simulated_data = pd.DataFrame(columns=['hours', 'sid_1', 'sid_2', 'item_id', 'variant'])

    # Let the start item_id from a 15-digit number
    item_id_start = 10 ** 14

    # Generate data for each hour
    for hour in hours:
        # Generate number of rows for the current hour using a normal distribution
        num_rows = int(np.random.normal(loc=mean_rows_per_hour, scale=stddev_rows_per_hour))

        if is_interleaved:
            # Create Interleaved DataFrame for the current hour
            current_data = pd.DataFrame({
                'hours': [hour] * num_rows,
                'sid_1': np.random.choice(sid_1, num_rows),
                'sid_2': np.random.choice(sid_2, num_rows),
                'item_id': np.arange(item_id_start, item_id_start + num_rows),
                'variant': np.random.choice(['A', 'B'], num_rows, p=[p_A, 1 - p_A])
            })
        else:
            # Create Ordinary DataFrame for the current hour
            current_data = pd.DataFrame({
                'hours': [hour] * num_rows,
                'sid_1': np.random.choice(sid_1, num_rows),
                'sid_2': np.random.choice(sid_2, num_rows),
                'item_id': np.arange(item_id_start, item_id_start + num_rows)
            })

        # Append to the main DataFrame
        simulated_data = pd.concat([simulated_data, current_data], ignore_index=True)

        # Update item_id_start for uniqueness in the next iteration
        item_id_start += num_rows

    # Convert the Pandas DataFrame to a Spark DataFrame
    simulated_data_spark = spark.createDataFrame(simulated_data)

    return simulated_data_spark
