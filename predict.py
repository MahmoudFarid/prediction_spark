import sys

from datetime import datetime, date, timedelta

from pyspark import SparkContext
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, udf, row_number, last, max, round
from pyspark.sql.window import Window
from pyspark.sql.types import DateType, StringType, IntegerType
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.linalg import Vectors, DenseVector, VectorUDT
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

prediction_features = ['overall', 'doctor', 'specialty', 'procedure', 'priority']
change_to_month_func = udf(lambda record: int(datetime.strftime(datetime.strptime(record, '%d/%m/%Y'), '%Y%m')),
                           IntegerType())
change_to_date_func = udf(lambda record: datetime.strptime(str(record), '%Y%m'), DateType())
change_date_to_month = udf(lambda record: datetime(record.year, record.month, 1), DateType())
to_vector = udf(lambda record: Vectors.dense(record), VectorUDT())
to_vectors = udf(lambda col_a, col_b: Vectors.sparse(col_a, col_b))

# Creating Spark Context and Spark Session
scobj = SparkContext.getOrCreate()
spark = SparkSession(scobj).builder.config('spark.sql.crossJoin.enabled', 'true').getOrCreate()


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
        final_prediction_df = ''
        for idx in xrange(1, predict_period + 1):
            if idx == 1:
                training_df = overall_prediction_grouping(csv_file_date)
                model, evaluator = overall_prediction_training(training_df)
                prediction_dataset = create_prediction_df(training_df, 1)

                prediction_df = model.transform(prediction_dataset.withColumn('id', to_vector(col('id'))).select(
                    'Date', 'id'))

                if final_prediction_df:
                    final_prediction_df.union(prediction_df.select('Date', round('prediction', 0)))
                else:
                    final_prediction_df = prediction_df.select('Date', round('prediction', 0))

            else:
                training_df = overall_prediction_grouping(csv_file_date, prediction_dataset=prediction_df
                                                          if prediction_df else None)
                model, evaluator = overall_prediction_training(training_df)
                prediction_dataset = create_prediction_df(training_df, 1)

                prediction_df = model.transform(
                    prediction_dataset.withColumn('id', to_vector(col('id'))).select('Date', 'id'))

                if final_prediction_df:
                    final_prediction_df = final_prediction_df.union(prediction_df.select('Date',
                                                                                         round('prediction', 0)))
                else:
                    final_prediction_df = prediction_df.select('Date', round('prediction', 0))

        if final_prediction_df:
            final_prediction_df.select(change_date_to_month(col('Date')).alias('Date'), col(
                'round(prediction, 0)').alias('prediction')).coalesce(1).write.csv(
                'Prediction_overall_%s_%s.csv' % (csv.split('.')[0], predict_period), mode='overwrite', header=True)

        else:
            raise ValueError('Prediction period should be greater than 1.')

    else:
        final_prediction_df = ''
        str_name = predict_by.lower()
        column_name = str_name[0].capitalize() + str_name[1:]
        for idx in xrange(1, predict_period + 1):
            if idx == 1:
                training_df = specialized_prediction_grouping(csv_file_date, column=column_name)
                model, evaluator = specialized_prediction_training(training_df, column=column_name)
                specialized_prediction_dataset = create_prediction_df(training_df, 1,
                                                                      other_column=column_name)

                prediction_df = model.transform(specialized_prediction_dataset)

                if final_prediction_df:
                    final_prediction_df = final_prediction_df.union(prediction_df.select(
                        'Date', column_name, round('prediction', 0)))
                else:
                    final_prediction_df = prediction_df.select('Date', column_name, round('prediction', 0))

            else:
                training_df = specialized_prediction_grouping(csv_file_date,
                                                              prediction_dataset=prediction_df
                                                              if prediction_df else None,
                                                              column=column_name)
                model, evaluator = specialized_prediction_training(training_df, column=column_name)
                specialized_prediction_dataset = create_prediction_df(training_df, 1,
                                                                      other_column=column_name)

                prediction_df = model.transform(specialized_prediction_dataset)

                if final_prediction_df:
                    final_prediction_df = final_prediction_df.union(prediction_df.select(
                        'Date', column_name, round('prediction', 0)))
                else:
                    final_prediction_df = prediction_df.select('Date', column_name, round('prediction', 0))

        if final_prediction_df:
            final_prediction_df.select(change_date_to_month(col('Date')).alias('Date'), column_name, col(
                'round(prediction, 0)').alias('prediction')).coalesce(1).write.csv(
                'Prediction_%s_%s_%s.csv' % (csv.split('.')[0], predict_by, predict_period), mode='overwrite',
                header=True)
        else:
            raise ValueError('Prediction period should be greater than 1.')


def overall_prediction_grouping(csv, prediction_dataset=None):
    """
    Grouping dataset by date
    :param csv: -- dataframe: containing all the data
    :param prediction_dataset: -- dataframe: containing previous prediction
    :return: -- dataframe: grouped
    """
    grouped = csv.groupby('Date').agg({'Date': 'count'})
    grouped_with_date = grouped.withColumn('Date', change_to_date_func(col('Date')))
    window_row = Window().orderBy('Date')
    grouped_indexed = grouped_with_date.withColumn('id', row_number().over(window_row))

    if prediction_dataset:
        grouped_with_cols = grouped_indexed.select('Date', 'id', 'count(Date)').withColumn('id', to_vector(col('id')))

        prediction_dataset_with_cols = prediction_dataset.select('Date', 'id', 'prediction').withColumnRenamed(
            'prediction', 'count(Date)')
        return grouped_with_cols.union(prediction_dataset_with_cols)
    else:
        return grouped_indexed.withColumn('id', to_vector(col('id')))


def overall_prediction_training(grouped_with_id):
    """
    The function calculates prediction for overall number of admissions
    :param grouped_with_id: -- dataframe: containing all the data
    :return: model created and its evaluator
    """

    maximum_features = calculate_max_bins(grouped_with_id, 'id', overall=True)

    return create_model(grouped_with_id, features_col='id', label_col='count(Date)',
                        max_bins=maximum_features)


def create_model(training_data, features_col, label_col, max_bins=32):
    """
    Create machine learning model
    :param training_data: -- dataframe: training dataset
    :param features_col: -- col: containing all the features needed.
    :param label_col: -- col: label
    :param max_bins: -- integer: number of bins needed for
    :return: model created and its evaluator
    """

    # Create Decision Tree Model
    dt = DecisionTreeRegressor()

    # Create params for the model
    params = ParamGridBuilder().baseOn({dt.featuresCol: features_col}).baseOn({dt.labelCol: label_col}).addGrid(
        dt.maxDepth, [3, 5, 7]).addGrid(dt.maxBins, [32 if max_bins <= 32 else max_bins + 1]).build()

    # Model Evaluator
    dt_evaluator = RegressionEvaluator(labelCol=label_col)

    # Create model with Cross Validation to get the best results
    dt_cv = CrossValidator(estimator=dt, estimatorParamMaps=params, evaluator=dt_evaluator)

    dt_cv_model = dt_cv.fit(training_data)

    return dt_cv_model, dt_evaluator


def calculate_max_bins(training_df, column, overall=True):
    """
    Calculate max bins needed for the DecisionTreeRegressor
    :param training_df: -- dataframe: training dataset
    :param column: -- column name
    :return: -- integer: max bins
    """
    collected_maximum = training_df.select(max(column)).collect()
    return collected_maximum[0][0][0] if overall else collected_maximum[0][0]


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

    return prediction_df


def specialized_prediction_grouping(csv, column, prediction_dataset=None):
    """
    Grouping dataset by date
    :param csv: -- dataframe: containing all the data
    :param column: -- string: column name used for aggregation
    :param prediction_dataset: -- dataframe: to add for calculation
    :return: -- dataframe: grouped
    """
    grouped = csv.groupby('Date', column).agg({column: 'count'})
    grouped_with_date = grouped.withColumn('Date', change_to_date_func(col('Date')))
    window_row = Window().partitionBy(column).orderBy('Date')
    grouped_with_date_and_id = grouped_with_date.withColumn('id', row_number().over(window_row))

    strindexer = StringIndexer(inputCol=column, outputCol=column + '_idx')
    model_strindexer = strindexer.fit(grouped_with_date_and_id)
    grouped_with_date_and_id_indexed = model_strindexer.transform(grouped_with_date_and_id)

    assembler = VectorAssembler(inputCols=['id', column + '_idx'], outputCol='features')

    if prediction_dataset:
        grouped_with_selected_cols = grouped_with_date_and_id_indexed.select('Date', 'id', column + '_idx', column,
                                                                             'count(%s)' % column)
        prediction_dataset_with_cols = prediction_dataset.select('Date', 'id', column + '_idx', column, 'prediction')
        new_training_df = grouped_with_selected_cols.union(prediction_dataset_with_cols.withColumnRenamed('prediction',
                                                           'count(%s)' % column))

        return assembler.transform(new_training_df)
    else:
        return assembler.transform(grouped_with_date_and_id_indexed)


def specialized_prediction_training(grouped_with_id, column):
    """
    The function calculates prediction for overall number of admissions
    :param column: -- string: column name used for aggregation
    :param grouped_with_id: -- dataframe: containing all the data
    :return: model created and its evaluator
    """

    maximum_features_id = calculate_max_bins(grouped_with_id, 'id', overall=False)
    maximum_features_col = calculate_max_bins(grouped_with_id, column + '_idx', overall=False)

    maximum_features = maximum_features_id if maximum_features_id >= maximum_features_col else maximum_features_col

    return create_model(grouped_with_id, features_col='features', label_col='count(' + column + ')',
                        max_bins=maximum_features)


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

        predict_by = sys.argv[2]
        period = int(sys.argv[3])
        perform_prediction(path, predict_by=predict_by, predict_period=period)
    except IndexError:
        print "Usage: run.py <CSV path> <predict on> <period>"
        sys.exit(1)
