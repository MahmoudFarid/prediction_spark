# prediction_spark

This Repo will include script that predict some data from CSVs files.

To run the admissions.py:
- Open python shell using the command python
- In the shell type:
    from admissions import predict_admissions
    predict_admissions(csv_path, predict_by, predict_period)

    predict_by should be one of the columns of the csv file
    predict_period should be any integer number from 1 to 36 months