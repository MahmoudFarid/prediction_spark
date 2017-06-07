import sys
import os

from datetime import datetime, date, timedelta

from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, udf, row_number, last, max, round
from pyspark.sql.window import Window
from pyspark.sql.types import DateType, StringType, IntegerType, DoubleType
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.linalg import Vectors, DenseVector, VectorUDT
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

spark = SparkSession.builder.config('spark.sql.crossJoin.enabled', 'true').getOrCreate()
prediction_features = ['overall', 'doctor', 'specialty', 'procedure', 'priority']

change_to_month_func = udf(lambda record: int(datetime.strftime(datetime.strptime(record, '%d/%m/%Y'), '%Y%m')),
                           IntegerType())
change_to_date_func = udf(lambda record: datetime.strptime(str(record), '%Y%m'), DateType())
change_date_to_month = udf(lambda record: datetime(record.year, record.month, 1), DateType())
to_vector = udf(lambda record: Vectors.dense(record), VectorUDT())
to_vectors = udf(lambda col_a, col_b: Vectors.sparse(col_a, col_b))
negative_to_zero = udf(lambda col: 0.0 if col < 0.0 else col, DoubleType())


def perform_prediction(csv, predict_by='Overall', predict_period=3):
    """
    The function is the entry point to the prediction module.
    :param csv: --string: path to the csv file containing the data
    :param predict_by: -- string: The choices should be 'Overall', 'Doctor', 'Specialty', 'Procedure',
           and 'Priority'
    :param predict_period: -- integer: The choices should be 3, 6, 12, 24, 36
    :return:
    """
    if predict_by.lower() not in prediction_features:
        raise ValueError("Unknown prediction feature. Predict_by should be one of %s" % prediction_features)

    # Reading data from CSV
    csv_file = spark.read.csv(csv, header=True)

    # Changing date format
    date_col = get_date_column_name(csv_file.columns)

    if not date_col:
        raise ValueError("Date column should contain the word 'date'")

    csv_file_date = csv_file.withColumn('Date', change_to_month_func(col(date_col)))

    if predict_by.lower() == 'overall':
        training_df = overall_prediction_grouping(csv_file_date)
        model, evaluator = create_model(training_df, features_col='id', label_col='count(Date)')
        prediction_dataset = create_prediction_df(training_df, predict_period)

        prediction_df = model.transform(prediction_dataset)
        final_prediction_df = prediction_df.select('Date', round('prediction', 0)).withColumn(
            'prediction', negative_to_zero(col('round(prediction, 0)')))

        # print "The mean average error is %s" % evaluator.evaluate(prediction_df, {evaluator.metricName: "mae"})

        return final_prediction_df.select(change_date_to_month(col('Date')).alias('Date'), 'prediction').coalesce(
            1).write.csv('%s%sprediction_%s_%s_%s.csv' % (os.path.dirname(csv), os.sep, os.path.basename(
                os.path.splitext(csv)[0]), predict_by, predict_period),
                         mode='overwrite', header=True)

    else:
        if predict_by.lower() not in [c.lower() for c in csv_file_date.columns]:
            raise ValueError("Predict by field should be one the CSV file columns")

        str_name = predict_by.lower()
        column_name = str_name[0].capitalize() + str_name[1:]

        training_df = specialized_prediction_grouping(csv_file_date, column=column_name)
        model, evaluator = create_model(training_df, 'features', 'count(%s)' % column_name)
        specialized_prediction_dataset = create_prediction_df(training_df, predict_period, other_column=column_name)

        prediction_df = model.transform(specialized_prediction_dataset)

        final_prediction_df = prediction_df.select('Date', round('prediction', 0), column_name).withColumn(
            'prediction', negative_to_zero(col('round(prediction, 0)')))

        # print "The mean average error is %s" % evaluator.evaluate(prediction_df, {evaluator.metricName: "mae"})

        return final_prediction_df.select(change_date_to_month(col('Date')).alias(
            'Date'), column_name, 'prediction').coalesce(1).write.csv(
            '%s%sprediction_%s_%s_%s.csv' % (os.path.dirname(csv), os.sep, os.path.basename(os.path.splitext(csv)[0]),
                                             predict_by, predict_period),
            mode='overwrite', header=True)


def overall_prediction_grouping(csv):
    """
    Grouping dataset by date
    :param csv: -- dataframe: containing all the data
    :return: -- dataframe: grouped
    """
    grouped = csv.groupby('Date').agg({'Date': 'count'})
    grouped_with_date = grouped.withColumn('Date', change_to_date_func(col('Date')))
    window_row = Window().orderBy('Date')
    grouped_indexed = grouped_with_date.withColumn('id', row_number().over(window_row))

    return grouped_indexed.withColumn('id', to_vector(col('id')))


def create_model(training_data, features_col, label_col):
    """
    Create machine learning model
    :param training_data: -- dataframe: training dataset
    :param features_col: -- col: containing all the features needed.
    :param label_col: -- col: label
    :return: model created and its evaluator
    """

    # Create Generalized Linear Regression Model
    glr = GeneralizedLinearRegression()

    # Create params for the model
    params = ParamGridBuilder().baseOn({glr.labelCol: label_col}).baseOn({glr.featuresCol: features_col}).addGrid(
        glr.family, ["gaussian", "poisson"]).build()

    # Model Evaluator
    glr_evaluator = RegressionEvaluator(labelCol=label_col)

    # Create model with Cross Validation to get the best results
    glr_cv = CrossValidator(estimator=glr, estimatorParamMaps=params, evaluator=glr_evaluator)

    dt_cv_model = glr_cv.fit(training_data)

    return dt_cv_model, glr_evaluator


def create_prediction_df(training_df, prediction_period, other_column=None):
    """
    :param training_df: -- dataframe: for training
    :param prediction_period: -- integer: number of period to predict on
    :param other_column: -- col column for prediction other than the overall rank
    :return: -- dataframe and corresponding dates
    """
    if other_column:
        last_date = training_df.select('Date').distinct().orderBy('Date').select(last('Date')).collect()[0][0]
        last_id = training_df.select('id').distinct().orderBy('id').select(max('id')).collect()[0][0]

        date_rows = list(Row(float(last_id + i), date(last_date.year, last_date.month, 1) + timedelta(days=i * 31))
                         for i in range(1, prediction_period + 1))

        date_df = spark.createDataFrame(date_rows, ['id', 'Date'])

        specialized_rows = training_df.select(other_column, other_column + '_idx').distinct()

        prediction_df = specialized_rows.join(date_df)

        assembler = VectorAssembler(inputCols=['id', other_column + '_idx'], outputCol='features')
        return assembler.transform(prediction_df)

    else:
        last_date = training_df.orderBy('Date').select(last('Date')).collect()[0][0]
        last_id = training_df.orderBy('id').select(max('id')).collect()[0][0][0]
        prediction_rows = list(Row(float(last_id + i), date(last_date.year, last_date.month, 1) +
                                   timedelta(days=i * 31)) for i in range(1, prediction_period + 1))

        prediction_df = spark.createDataFrame(prediction_rows, ['id', 'Date'])

    return prediction_df.withColumn('id', to_vector(col('id')))


def specialized_prediction_grouping(csv, column):
    """
    Grouping dataset by date
    :param csv: -- dataframe: containing all the data
    :param column: -- string: column name used for aggregation
    :return: -- dataframe: grouped
    """
    grouped = csv.groupby('Date', column).agg({column: 'count'})
    grouped_with_date = grouped.withColumn('Date', change_to_date_func(col('Date')))
    window_row = Window().partitionBy(column).orderBy('Date')
    grouped_with_date_and_id = grouped_with_date.withColumn('id', row_number().over(window_row))

    str_indexer = StringIndexer(inputCol=column, outputCol=column + '_idx')
    model_str_indexer = str_indexer.fit(grouped_with_date_and_id)
    grouped_with_date_and_id_indexed = model_str_indexer.transform(grouped_with_date_and_id)

    assembler = VectorAssembler(inputCols=['id', column + '_idx'], outputCol='features')

    return assembler.transform(grouped_with_date_and_id_indexed)


def get_date_column_name(column_names):
    """
    :param column_names: -- list: containing dataframe column names
    :return: -- string: date column name
    """
    for column in column_names:
        if 'date' in column.lower():
            return column

    return ''


if __name__ == '__main__':
    try:
        path = sys.argv[1]
        if 'csv' not in path:
            raise ValueError("File should be in csv format")

        if 'dmission' not in path:
            raise ValueError("This file for Admissions only!")

        predict_by = sys.argv[2]
        period = int(sys.argv[3])
        perform_prediction(path, predict_by=predict_by, predict_period=period)
    except IndexError:
        print "Usage: run.py <CSV path> <predict on> <period>"
        sys.exit(1)
