from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class TaxiFarePipeline():

    def __init__(self):
        pass

    def create_pipeline(self):
        return Pipeline(
            steps=[('scaler',
                    StandardScaler()), ('regressor', LinearRegression())])
