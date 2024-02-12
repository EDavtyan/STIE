from obf import OBF
from maxsprt import MaxSPRT
from non_parametric_tests import NonParametricTests

import os
import s3fs
from pyspark.sql import SparkSession

from pyspark.sql.functions import *
from pyspark.sql.window import *

spark = SparkSession.builder.getOrCreate()
s3 = s3fs.S3FileSystem(anon=False)

class WriteReader:
    def __init__(self, base_path):
        self.base_path = base_path

    def write_and_read(self, df, name):
        path = os.path.join(self.base_path, name)
        df.write.mode("Overwrite").format("parquet").save(path)
        df = spark.read.format("parquet").load(path)
        return df

    def read(self, name):
        path = os.path.join(self.base_path, name)
        df = spark.read.format("parquet").load(path)
        return df

def ls(path):
    display(dbutils.fs.ls(path))
def rm(path):
    dbutils.fs.rm(path, True)

if __name__ == "__main__":
    path = "s3://pa-ai-datascience-storage-dev/users/ub/edgar.davtyan/Tasks/Research on AI Gen/Interleaving/PackageScripts/TestData/"

    write_reader_parquet = WriteReader(path)
    write_reader_delta = WriteReader(path)

    wr = write_reader_parquet.write_and_read
    rr = write_reader_parquet.read

    ai_apply_click_aa = rr("sample_data_apply_click_aa").select("device_id", "sid", "keyword", "time_hours", "item_id")
    ai_apply_click = rr("sample_data_apply_click")

    sid = ["sid", "keyword"]
    grouping = "time_hours"

    MaxSPRT_tester = MaxSPRT(sid_columns=sid,
                             grouping_column=grouping,
                             treatment_variant="stable_diffusion_v22",
                             control_variant="stable_diffusion_v21",
                             item_id_column="photo_id")

    threshold = MaxSPRT_tester.calculate_threshold(data_set=ai_apply_click,
                                       num_experiments=2)

    MaxSPRT_tester.transform(ai_apply_click)
    results = MaxSPRT_tester.run_test()

    MaxSPRT_tester.visualize(MaxSPRT_tester)