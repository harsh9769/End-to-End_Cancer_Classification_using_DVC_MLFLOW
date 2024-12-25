from cnnClassifier.components.model_evaluation import Evaluation
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier import logger

STAGE_NAME="Model Eval"


class ModelEvaluationPipeline:

    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        #evaluation.log_into_mlflow()


if __name__ == "__main__":

    try:
        logger.info(f">>>>>>>>>>>> {STAGE_NAME} started <<<<<<<<<<<<")
        obj=ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
