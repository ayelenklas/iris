from iris.data import get_data, holdout
from iris.pipeline import TaxiFarePipeline
import joblib

class Trainer():

    def __init__(self):
        pass

    def fit(self):
        self.pipeline.fit(self.X_train, self.y_train)

    def save_pipeline(self):
        joblib.dump(self.pipeline,"pipeline.joblib")

    def train(self):
        # get data
        df = get_data()

        # holdout
        (self.X_train, self.X_test, self.y_train, self.y_test) = holdout(df)

        # pipeline
        tf_pipeline = TaxiFarePipeline()
        self.pipeline = tf_pipeline.create_pipeline()

        # fit pipeline
        self.fit()

        # save pipeline
        self.save_pipeline()
