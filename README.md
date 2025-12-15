
  ## For preprocessing
  ### prerequisite:
      - datasource should be added and ray cluster created with datasource mount
        command :
           python /home/<username>/insurance_mlops/preprocessing.py
           add env variable:
             - DATASET_PATH: <mounted path>
  ## For training
      python /home/<username>/insurance_mlops/training.py
      env:
      DATASET_PATH:<mounted path> <mouted path which is giving during ray cluster creation>
      PROJECT_NAME: <mlflow experiment name> <not mandatory>
  ## For deployment
      python /home/<username>/insurance_mlops/deployment.app
  ## For prediction
       python /home/<username>/insurance_mlops/prediction.py
  ### Data source is snowflake
  ## For preprocessing
      cmd:
        python /home/<username>/insurance_mlops/preprocessing_snowflake.py
      pip packages:
         snowflake-connector-python,pandas
      env variable:
         OUT_TABLE_NAME : <output table name> <not mandatory>
      
  ## For training
     cmd:
        python /home/<username>/insurance_mlops/training_snowflake.py
     pip packages:
         snowflake-connector-python,pandas
     env variable:
         SNOWFLAKE_TABLE : <table name>
  ## For deployment
      python /home/<username>/insurance_mlops/deployment.app
  ## For prediction
       python /home/<username>/insurance_mlops/prediction.py
     
