import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.config.configuration import EvaluationConfig
import dagshub
from cnnClassifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1. / 255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=tuple(self.config.params_image_size[:2]),
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        # Set the MLflow registry URI
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Initialize DagsHub integration
        dagshub.init(
            repo_owner='harsh9769',
            repo_name='End-to-End_Cancer_Classification_using_DVC_MLFLOW',
            mlflow=True
        )

        with mlflow.start_run():
            # Log parameters and metrics
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})
            mlflow.log_param('additional_param', 'value')
            mlflow.log_metric('additional_metric', 1)

            # Check if tracking URL type is not a file store
            if tracking_url_type_store != "file":
                # Log and register the model in MLflow's model registry
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                # Log the model locally
                mlflow.keras.log_model(self.model, "model")
