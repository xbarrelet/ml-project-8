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
from pyspark.sql.functions import pandas_udf, PandasUDFType, element_at, split, udf, monotonically_increasing_id

DATA_PATH = "resources/fruits-360_dataset/fruits-360/Test/"
RESULTS_PATH = "results/"
RESULTS_PATH2 = "results2/"

# SPARK OBJECTS
spark = (SparkSession
         .builder
         .appName('local-app')
         .master('local')
         .config("spark.driver.memory", "15g")
         .config("spark.sql.parquet.writeLegacyFormat", 'true')
         .getOrCreate()
         )
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
    # array_to_vector = udf(lambda x: Vectors.dense(x), VectorUDT())
    # spark_df = features_df.withColumn("features", array_to_vector("features"))

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


def vector_to_array(vector):
    """Convert a spark vector to a numpy array"""
    return vector.toArray()


def incremental_pca(spark_df, n_components=10, batch_size=1000):
    """
    Implements incremental PCA using batched processing.

    Args:
        spark_df: Input Spark DataFrame with features column containing vectors
        n_components: Number of PCA components to keep
        batch_size: Size of batches for incremental processing

    Returns:
        Transformed DataFrame and explained variance ratios
    """
    # First, count total rows to determine number of batches
    # First, optimize partitioning based on data size and batch size
    total_rows = spark_df.count()
    num_partitions = max(total_rows // batch_size, 1)  # At least one partition

    # Add unique IDs and repartition the data
    df_partitioned = (spark_df
                      .withColumn("id", monotonically_increasing_id())
                      .repartition(num_partitions)
                      .cache())  # Cache the repartitioned data

    # Process each partition in parallel
    def process_partition(iterator):
        """Process a partition of data and return local statistics"""
        partition_data = []
        for row in iterator:
            partition_data.append(vector_to_array(row.features))

        if not partition_data:
            return

        partition_data = np.array(partition_data)

        # Calculate local statistics
        local_sum = partition_data.sum(axis=0)
        local_n_samples = len(partition_data)

        # Calculate local SVD
        local_centered = partition_data - partition_data.mean(axis=0)
        U, S, Vt = np.linalg.svd(local_centered, full_matrices=False)

        # Return local statistics
        yield (local_n_samples, local_sum, S[:n_components], Vt[:n_components])

    # Collect local statistics from all partitions
    local_stats = df_partitioned.rdd.mapPartitions(process_partition).collect()

    # Combine statistics from all partitions
    total_n_samples = sum(stat[0] for stat in local_stats)
    mean_sum = sum(stat[1] for stat in local_stats)
    global_mean = mean_sum / total_n_samples

    # Combine components using SVD update
    all_components = []
    all_singular_values = []

    for _, _, S, Vt in local_stats:
        all_components.append(S.reshape(-1, 1) * Vt)
        all_singular_values.extend(S)

    if all_components:
        combined = np.vstack(all_components)
        U_combined, S_combined, Vt_combined = np.linalg.svd(combined, full_matrices=False)

        # Get final components and variance
        components = Vt_combined[:n_components]
        singular_values = S_combined[:n_components]
        explained_variance = (singular_values ** 2) / (total_n_samples - 1)
        total_var = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_var
    else:
        raise ValueError("No data was processed")

    # Create transformation function
    def transform_vector(v):
        v_array = vector_to_array(v)
        transformed = np.dot(v_array - global_mean, components.T)
        return Vectors.dense(transformed)

    # Register UDF for transformation
    transform_udf = udf(transform_vector, VectorUDT())

    # Transform dataset in parallel
    result_df = df_partitioned.withColumn("pca_features", transform_udf("features"))

    # Cleanup cached data
    df_partitioned.unpersist()

    return result_df, explained_variance_ratio


if __name__ == '__main__':
    print("Starting Spark script.\n")

    images = spark.read.format("binaryFile") \
        .option("pathGlobFilter", "*.jpg") \
        .option("recursiveFileLookup", "true") \
        .load(DATA_PATH)

    images = images.withColumn('label', element_at(split(images['path'], '/'), -2))

    # Transfer learning, we don't keep the last layer
    model = MobileNetV2(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    new_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    broadcasted_weights = sc.broadcast(new_model.get_weights())
    print("The weights have been broadcasted.\n")

    # Featurize images and saving the results
    # numbers_to_float_udf = udf(lambda x: [float(number) for number in x], ArrayType(FloatType()))
    # array_to_vector_udf = udf(lambda x: Vectors.dense(x), VectorUDT())
    # features_df = (images
    #                .repartition(20)
    #                .withColumn("features", featurize_udf("content"))
    #                .withColumn("features", numbers_to_float_udf("features"))
    #                .withColumn("features", array_to_vector_udf("features"))
    #                .select("path", "label", "features"))
    # features_df.write.mode("overwrite").parquet(RESULTS_PATH)

    # Load the results back
    spark_df = spark.read.parquet(RESULTS_PATH)
    print("The results have been loaded back.\n")

    # Missing PCA step added
    pca = PCA(k=10, inputCol="features", outputCol="pca_features")
    pca_model = pca.fit(spark_df)
    df_pca = pca_model.transform(spark_df)

    explained_variance = pca_model.explainedVariance
    print("Explained variance ratio:", explained_variance)
    # [0.10140967559306546,0.08004581517922967,0.06344774721367424,0.050141178300393364,0.035335157853023255,0.029163995040856393,0.027726724002401414,0.02285213787857822,0.019861126212198935,0.019093213598319357]

    # Run incremental PCA
    # result_df, explained_variance_ratio = incremental_pca(spark_df, n_components=10, batch_size=1000)
    # print("Explained variance ratio:", explained_variance_ratio)
    # Explained variance ratio: [0.22636246 0.17868195 0.14160965 0.11183201 0.07883316 0.0649908
    #  0.06171848 0.05064554 0.0431696  0.04215633]

    # df_pca = df_pca.select(
    #     "path",
    #     "label",
    #     col("pca_features").getItem(0).alias("pca_1"),
    #     col("pca_features").getItem(1).alias("pca_2")
    # )

    # Show the results
    # df_pca.show(5)

    spark.stop()
