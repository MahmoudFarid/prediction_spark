from datetime import datetime

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, row_number
from pyspark.sql.window import Window
from pyspark.sql.types import DateType, StringType, IntegerType
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.linalg import Vectors, DenseVector, VectorUDT
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


prediction_features = ['overall', 'doctor', 'specialty', 'procedure', 'priority']
change_to_month_func = udf(lambda record: int(datetime.strftime(datetime.strptime(record, '%d/%m/%Y'), '%Y%m')), IntegerType())
change_to_date_func = udf(lambda record: datetime.strptime(str(record), '%Y%m'), DateType())
to_vector = udf(lambda record: Vectors.dense(record), VectorUDT())
to_vectors = udf(lambda col_a, col_b: Vectors.sparse(col_a, col_b))


def predict_admissions(csv, predict_by='Overall', predict_period=3):
    """
    The function is the entry point to the prediction module.
    :param csv: --string: path to the csv file containing the data
    :param predict_by: -- string: The choices should be 'Overall', 'Doctor', 'Specialty', 'Procedure',
           and 'Priority'
    :param predict_period: -- integer: The choices should be 3, 6, 12, 24, 36
    :return: dataframe with the required results
    """
    if predict_by.lower() not in prediction_features:
        raise ValueError("Unknown prediction feature. Predict_by should be one of %s" % prediction_features)

    # Creating Spark Context and Spark Session
    scobj = SparkContext.getOrCreate()
    spark =  SparkSession(scobj)

    # Reading data from CSV
    csv_file = spark.read.csv(csv, header=True)

    # Changing date format
    csv_file_date = csv_file.withColumn('Date', change_to_month_func(col('Removal Date'))).drop('Removal Date')

    if predict_by.lower() == 'overall':
        overall_df = overall_prediction_training(csv_file_date, predict_period)


def overall_prediction_training(csv, predict_period=3):
    """
    The function calculates prediction for overall number of admissions
    :param csv: -- dataframe: containing all the data
    :param predict_period: -- integer: number of months to predict on
    :return: -- dataframe: aggregated and with prediction
    """
    grouped = csv.groupby('Date').agg({'Date': 'count'})
    grouped_with_date = grouped.withColumn('Date', change_to_date_func(col('Date')))
    window_row = Window().orderBy('Date')
    grouped_with_id = grouped_with_date.withColumn('id', row_number().over(window_row)
                                                   ).withColumn('id', to_vector(col('id')))

    maximum_features = calculate_max_bins(grouped_with_id, 'id')

    model, evaluator = create_model(grouped_with_id, features_col='id', label_col='count(Date)',
                                    max_bins=maximum_features)


def create_model(training_data, features_col, label_col, max_bins=32):
    """
    Create machine learning model
    :param training_data: -- dataframe: training dataset
    :param features_col: -- col: containing all the features needed.
    :param label_col: -- col: label
    :return: model created and its evaluator
    """

    # Create Decision Tree Model
    dt = DecisionTreeRegressor()

    # Create params for the model
    params = ParamGridBuilder().baseOn({dt.featuresCol: features_col}).baseOn({dt.labelCol: label_col}).addGrid(
        dt.maxDepth, [3, 5, 7]).addGrid(dt.maxBins, 32 if max_bins <= 32 else max_bins)

    # Model Evaluator
    dt_evaluator = RegressionEvaluator(labelCol=label_col)

    # Create model with Cross Validation to get the best results
    dt_cv = CrossValidator(estimator=dt, estimatorParamMaps=params, evaluator=dt_evaluator)

    dt_cv_model = dt_cv.fit(training_data)

    return dt_cv_model, dt_evaluator


def calculate_max_bins(training_df, column):
    """
    Calculate max bins needed for the DecisionTreeRegressor
    :param training_df: -- dataframe: training dataset
    :param column: -- column name
    :return: -- integer: max bins
    """
    return training_df.select(column).rdd.max()[0]
