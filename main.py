from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_trainer import ModelTrainerPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline
STAGE_NAME= "Data Ingestion stage"


try:
    logger.info(f">>>>>>>> stage {STAGE_NAME} started <<<<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<")
    
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare base model"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_base_model = PrepareBaseModelTrainingPipeline()
   prepare_base_model.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME= "Model Trainer Stage"
try:
    logger.info(f">>>>>>>>>>>> {STAGE_NAME} started <<<<<<<<<<<<")
    obj= ModelTrainerPipeline()
    obj.main()
    logger.info(f">>>>>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME="Model Eval"

try:
    logger.info(f">>>>>>>>>>>> {STAGE_NAME} started <<<<<<<<<<<<")
    obj=ModelEvaluationPipeline()
    obj.main()
    logger.info(f">>>>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e
            