# Sequential Testing for Interleaving Experiments (STIE)

## Description

STIE is a Python package designed to streamline the application of sequential testing methods, with a special emphasis
on online interleaving experimentations. This suite amplifies the potential of interleaving experimentation by offering
a structured approach to decide on the early termination of an experiment or to continue with data collection. This
ensures optimal resource allocation and efficiency in experimental settings.

The package is anchored around the following sequential testing methods:

### 1. Maximized Sequential Probability Ratio Test (MaxSPRT)

The Maximized Sequential Probability Ratio Test (MaxSPRT) [^1] is a statistical method primarily developed to detect
adverse
events in drug and vaccine safety surveillance, especially those that are rare and may not be identified during phase 3
trials. The method is designed to identify such events as swiftly as possible while minimizing false alarms. MaxSPRT is
an adaptation of Wald's classical Sequential Probability Ratio Test (SPRT) but is designed to be more sensitive across a
range of relative risks by utilizing a composite alternative hypothesis instead of a specific value. This method is
particularly relevant for continuous or near-continuous sequential test procedures, which reevaluate data on a daily or
weekly basis, ensuring that potential issues are detected and addressed promptly. The method has been applied in
real-world scenarios, such as vaccine safety surveillance, and has been compared with the classical SPRT, demonstrating
its applicability and effectiveness in various real-time vaccine safety surveillance applications.

### 2. O’Brien & Fleming’s Multiple Testing Procedure (OBF)

O’Brien & Fleming's Multiple Testing Procedure (OBF) [^2] is a statistical method designed for clinical trials,
particularly
those where the response to treatment is both dichotomous (i.e., success or failure) and immediate. The method involves
periodically reviewing data, with a predetermined maximum number of tests (N) and a fixed number of observations
collected between successive tests. The procedure allows for early termination of the study if initial results indicate
a marked superiority of one treatment over the other, while maintaining control over the type I error rate. OBF provides
a framework that allows for the ethical and statistical integrity of clinical trials, offering the chance for early
termination when one treatment proves superior while preserving power equivalent to that of a single sample procedure of
the same size.

### Adaptation for Online Experimentations

Both the MaxSPRT and OBF methods have been adeptly adapted for online interleaving experimentations by Kharitonov et
al. [^3]. The adaptation of these methods to the realm of online experimentations allows for dynamic monitoring of
experimental outcomes and provides a reliable stoppage criterion that ensures experimental efficacy without compromising
statistical integrity. In the context of online experiments, particularly in search engine evaluation, these methods
have demonstrated the ability to significantly reduce the duration of experiments while maintaining reliability in
outcomes, ensuring that experimentation is both statistically sound and resource-efficient. This adaptation is
particularly crucial in online settings where rapid decision-making is essential and where the methods can be utilized
to dynamically observe and analyze experimental data, allowing for early detection of significant divergences between
control and treatment variants.

STIE stands as a harmonious fusion of time-tested statistical methodologies from the aforementioned methods and the
ever-evolving, data-centric nature of online experiments. This synergy ensures that your experimentation endeavors
remain both statistically rigorous and judicious in resource consumption.

## Setup

```bash
git clone https://github.com/EDavtyan/STIE.git
```

## Dependencies

This project relies on several key Python libraries for data processing, visualization, and computation. To ensure compatibility and functionality, the following specific versions of each library are required:

- **NumPy**: A fundamental package for scientific computing with Python, used for efficient operations on multi-dimensional arrays and matrices.
- **Pandas**: An open-source data analysis and manipulation tool, crucial for handling and analyzing input data.
- **Plotly**: An interactive graphing library for making interactive plots and dashboards.
- **tqdm**: A fast, extensible progress bar for loops and command-line programs.
- **SciPy**: Used for mathematical algorithms and convenience functions built on the NumPy extension of Python.
- **Matplotlib**: A plotting library for creating static, interactive, and animated visualizations in Python.
- **s3fs**: A Pythonic file interface to S3. It allows you to mount S3 buckets as if they were local files or directories.
- **PySpark**: Provides an interface for Apache Spark, allowing for scalable, high-throughput processing of big data sets.

To install these dependencies, you can use the following command:

```bash
pip install -r requirements.txt
```

Please ensure you have a compatible Python version installed, preferably Python 3.8 or newer, as these libraries may not support earlier versions of Python.


## Example Usage

### MaxSPRT Test

Below is an example of how to utilize the `MaxSPRT` class to conduct a Maximized Sequential Probability Ratio Test on an
interleaving experiment dataset:

```python
# Loading threshold and experiment data, selecting relevant columns
threshold_data = spark.read.format("parquet").load(data_dir_1)
.select("sid", "keyword", "time_hours", "photo_id")
experiment_data = spark.read.format("parquet").load(data_dir_2)
.select("sid", "keyword", "time_hours", "photo_id", "variant")

# Defining parameters for session identification, grouping, and variant naming
sid = ["sid", "keyword"]
grouping = "time_hours"
control = "A"
treatment = "B"
item_id = "photo_id"
variant_column = "variant"

# Initializing the MaxSPRT object with the defined parameters
maxsprt_test = MaxSPRT(sid_column=sid,
                       grouping_columns=grouping,
                       treatment_variant=treatment,
                       control_variant=control,
                       item_id_column=item_id,
                       variant_column=variant_column)

# Calculating the threshold using the threshold data and 1000 Monte Carlo simulations
threshold = maxsprt_test.calculate_threshold(data_set=threshold_data,
                                             num_experiments=1000)

# Transforming the experiment data and running the MaxSPRT test
maxsprt_test.transform(experiment_data)
results = maxsprt_test.run_test(L_i_threshold=threshold)

# Visualizing the results
maxsprt_test.visualize(results)
```

In this example, `threshold_data` is used to calculate the threshold for the L_i statistic, and `experiment_data`
contains the actual experimental data, including variant assignments (treatment/control). Ensure to modify `data_dir_1`
and `data_dir_2` to your actual data directory paths.

### OBF Test

Below is an example usage of the `OBF` class from the STIE package. This demonstrates how to utilize the OBF method to
run a sequential test on an interleaving experiment dataset:

```python
# Importing necessary PySpark type
from pyspark.sql.types import TimestampType

# Loading the threshold data from a specified directory and selecting relevant columns
threshold_data = spark.read.format("parquet").load(data_dir_1)
.select("sid", "keyword", "time_hours", "photo_id")

# Loading the experimental data (including variant assignments) from another directory
experiment_data = spark.read.format("parquet").load(data_dir_2)
.select("sid", "keyword", "time_hours", "photo_id", "variant")

# Defining variables for session identification, grouping, variant naming, etc.
sid = ["sid", "keyword"]
grouping = "time_hours"
control = "A"
treatment = "B"
item_id = "photo_id"
variant_column = "variant"

# Initializing the OBF object with the defined parameters.
obf_test = OBF(sid_column=sid,
               grouping_columns=grouping,
               treatment_variant=treatment,
               control_variant=control,
               item_id_column=item_id,
               variant_column=variant_column)

# Calculating the threshold for the O_i statistic using defined parameters.
threshold = obf_test.calculate_threshold(number_of_measurements=45,
                                         iterations=100000)

# Transforming the experiment data to a format suitable for running the OBF test.
obf_test.transform(experiment_data)

# Running the OBF test with the calculated O_i threshold and storing the results.
results = obf_test.run_test(O_i_threshold=threshold)

# Visualizing the results using the built-in visualization method.
obf_test.visualize(results)
```

In this example, after loading `threshold_data` and `experiment_data` as described in the MaxSPRT example, we initialize
an `OBF` object, calculate the O_i threshold, transform the experimental data, run the OBF test, and visualize the
results. The `calculate_threshold` method is specifically tailored to the OBF method, considering a specified number of
measurements and iterations for threshold calculation. Make sure that your data paths `data_dir_1` and `data_dir_2` are
modified to point to your actual data directories.

---

### References

[^1]: Kulldorff, M., Davis, R. L., Kolczak, M., Lewis, E., Lieu, T., & Platt, R. (2011). A Maximized Sequential
Probability Ratio Test for Drug and Vaccine Safety Surveillance. _Sequential Analysis, 30_(1),
58–78. [DOI: 10.1080/07474946.2011.539924](https://doi.org/10.1080/07474946.2011.539924).

[^2]: O'Brien, P. C., & Fleming, T. R. (1979). A Multiple Testing Procedure for Clinical Trials. _Biometrics, 35_(3),
549–556. Available at: [JSTOR](http://www.jstor.org/stable/2530245).

[^3]: Kharitonov, E., Vorobev, A., Macdonald, C., Serdyukov, P., & Ounis, I. (2015). Sequential Testing for Early
Stopping of Online Experiments. In Proceedings of the 38th International ACM SIGIR Conference on Research and
Development in Information Retrieval (pp. 473-482)
. [DOI: 10.1145/2766462.2767729](https://doi.org/10.1145/2766462.2767729).
