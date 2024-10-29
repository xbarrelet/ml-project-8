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
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, DoubleType

DATA_PATH = "resources/fruits-360_dataset/fruits-360/Test/"
RESULTS_PATH = "results/"

# SPARK OBJECTS
spark = (SparkSession
         .builder
         .appName('local-app')
         .master('local')
         .config("spark.sql.parquet.writeLegacyFormat", 'true')
         .getOrCreate()
         )
sc = spark.sparkContext

# PANDAS CONFIG
pd.options.display.max_columns = None
pd.options.display.max_rows = None


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
    # for multiple data batches.  This amortizes the overhead of loading big models.
    model = load_model()
    for content_series in content_series_iter:
        yield featurize_series(model, content_series)





if __name__ == '__main__':
    print("Starting Spark script.\n")

    images = spark.read.format("binaryFile") \
        .option("pathGlobFilter", "*.jpg") \
        .option("recursiveFileLookup", "true") \
        .load(DATA_PATH)

    # TODO: Fais des scripts avec boto3 pour creer l'archi sur AWS?

    images = images.withColumn('label', element_at(split(images['path'], '/'), -2))

    # Transfer learning, we don't keep the last layer
    model = MobileNetV2(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    new_model = Model(inputs=model.input, outputs=model.layers[-2].output)

    broadcasted_weights = sc.broadcast(new_model.get_weights())
    print("The weights have been broadcasted.\n")

    # features_df = images.repartition(20).select(col("path"), col("label"), featurize_udf("content").alias("features"))
    # features_df.write.mode("overwrite").parquet(RESULTS_PATH)

    # Load the results back
    df = pd.read_parquet(RESULTS_PATH, engine='pyarrow')
    print(df.head(5))

    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")

    # DF preparation for PCA
    df['features'] = df['features'].apply(lambda x: [float(val) for val in x])
    array_to_vector = udf(lambda x: Vectors.dense(x), VectorUDT())
    spark_df_with_vector = spark.createDataFrame(df).withColumn("features_vector", array_to_vector("features"))

    # Missing PCA step added
    for k in range(6):
        pca = PCA(k=k, inputCol="features_vector", outputCol="pca_features")

        pca_model = pca.fit(spark_df_with_vector)
        df_pca = pca_model.transform(spark_df_with_vector)

        explained_variance = pca_model.explainedVariance
        print("Explained variance ratio:", explained_variance)

        # df_pca = df_pca.select(
        #     "path",
        #     "label",
        #     col("pca_features").getItem(0).alias("pca_1"),
        #     col("pca_features").getItem(1).alias("pca_2")
        # )

        # Show the results
        # df_pca.show(5)


