import io

import numpy as np
import pandas as pd
from PIL import Image
from keras import Model
from keras.src.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.src.utils import img_to_array
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType, element_at, split, col, udf
from pyspark.sql.types import ArrayType, FloatType

#

# LOCAL CONFIG
DATA_PATH = "resources/fruits-360_dataset/fruits-360/Test/"
RESULTS_PATH = "results/"
PCA_RESULTS_PATH = "results_pca/"

spark = (SparkSession
         .builder
         .appName('local-app')
         .master('local')
         .config("spark.driver.memory", "15g")
         .config("spark.sql.parquet.writeLegacyFormat", 'true')
         .getOrCreate()
         )

# AWS CONFIG
# DATA_PATH = "s3a://xavier-project-8-bucket/Test/"
# RESULTS_PATH = "s3a://xavier-project-8-bucket/results/"
# PCA_RESULTS_PATH = "s3a://xavier-project-8-bucket/results_pca/"
#
# CPU_CORES_PER_NODE = 4
# MEMORY_PER_NODE = 15
#
# spark = (SparkSession
#          .builder
#          .appName('emr-image-processing')
#          .config("spark.executor.memory", MEMORY_PER_NODE)
#          .config("spark.driver.memory", MEMORY_PER_NODE)
#          .config("spark.executor.cores", CPU_CORES_PER_NODE)
#          .config("spark.sql.parquet.writeLegacyFormat", 'true')
#          .getOrCreate()
#          )

sc = spark.sparkContext

# CONFIG
pd.options.display.max_columns = None
pd.options.display.max_rows = None
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")


def load_model():
    """
    Returns a MobileNetV2 model with top layer removed
    and broadcasted pretrained weights.
    """
    model = MobileNetV2(weights='imagenet', include_top=True, input_shape=(224, 224, 3))

    for layer in model.layers:
        layer.trainable = False

    new_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    new_model.set_weights(broadcasted_weights.value)
    return new_model


def preprocess(content):
    """
    Preprocesses raw image bytes for prediction.
    """
    img = Image.open(io.BytesIO(content)).resize([224, 224])
    arr = img_to_array(img)
    return preprocess_input(arr)


def featurize_series(model, content_series):
    """
    Featurize a pd.Series of raw images using the input model.
    :return: a pd.Series of image features
    """
    input = np.stack(content_series.map(preprocess))
    preds = model.predict(input)
    # For some layers, output features will be multi-dimensional tensors.
    # We flatten the feature tensors to vectors for easier storage in Spark DataFrames.
    output = [p.flatten() for p in preds]
    return pd.Series(output)


@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter):
    '''
    This method is a Scalar Iterator pandas UDF wrapping our featurization function.
    The decorator specifies that this returns a Spark DataFrame column of type ArrayType(FloatType).

    :param content_series_iter: This argument is an iterator over batches of data, where each batch
                              is a pandas Series of image data.
    '''
    # With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
    # for multiple data batches. This amortizes the overhead of loading big models.
    model = load_model()
    for content_series in content_series_iter:
        yield featurize_series(model, content_series)


if __name__ == '__main__':
    print("Starting Spark script.\n")

    images = spark.read.format("binaryFile") \
        .option("pathGlobFilter", "*.jpg") \
        .option("recursiveFileLookup", "true") \
        .load(DATA_PATH)

    #  jovyan
    #  jupyter

    print(f"Loaded {images.count()} images from {DATA_PATH}")

    images = images.withColumn('label', element_at(split(images['path'], '/'), -2))

    # Transfer learning, we don't keep the last layer
    model = MobileNetV2(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    new_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    broadcasted_weights = sc.broadcast(new_model.get_weights())
    print("The weights have been broadcasted.\n")

    # Featurize images and saving the results
    numbers_to_float_udf = udf(lambda x: [float(number) for number in x], ArrayType(FloatType()))
    array_to_vector_udf = udf(lambda x: Vectors.dense(x), VectorUDT())
    features_df = (images
                    # 20 partitions
                   .repartition(20)
                   .withColumn("features", featurize_udf("content"))
                   .withColumn("features", numbers_to_float_udf("features"))
                   .withColumn("features", array_to_vector_udf("features"))
                   .select("path", "label", "features"))
    features_df.write.mode("overwrite").parquet(RESULTS_PATH)
    print(f"The features have been saved to {RESULTS_PATH}.\n")

    # Load the results back
    spark_df = spark.read.parquet(RESULTS_PATH)
    print("The results have been loaded back.\n")

    # Missing PCA step added
    pca = PCA(k=42, inputCol="features", outputCol="pca_features")
    pca_model = pca.fit(spark_df)
    df_pca = pca_model.transform(spark_df)

    # Saving df with PCA as parquet and csv
    (df_pca.write.mode("overwrite")
     .parquet(PCA_RESULTS_PATH))
    (df_pca
     .withColumn("features", col("features").cast("string"))
     .withColumn("pca_features", col("pca_features").cast("string"))
     .write
     .mode("overwrite")
     .option("header", "true")
     .format("csv")
     .save(f"{PCA_RESULTS_PATH}_csv"))

    explained_variance = pca_model.explainedVariance
    # print("Explained variance ratio:", explained_variance)
    print(f"Total variance: {sum(explained_variance.toArray())}")

    # Load the results back again
    spark_pca_df = spark.read.parquet(PCA_RESULTS_PATH)
    print("The results with PCA have been loaded back.\n")

    print(spark_pca_df.show(5))
