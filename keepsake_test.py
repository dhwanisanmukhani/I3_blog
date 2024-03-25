import pandas as pd
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import cross_validate
import joblib
import keepsake

def train(learning_rate=0.005, num_epochs=20, training_data_path='2024-03-15.pkl'):
# Load the data
    experiment = keepsake.init(
        path=".",
        params={"learning_rate": learning_rate, "num_epochs": num_epochs, "training_data": training_data_path},
    )
    data = pd.read_pickle(training_data_path)

    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(data[['userid', 'movieid', 'rating']], reader)

    algo = SVD(lr_all=learning_rate, n_epochs=num_epochs)    
    cv_results = cross_validate(algo, dataset, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    rmse_mean = cv_results['test_rmse'].mean()
    mae_mean = cv_results['test_mae'].mean()
    
    trainset = dataset.build_full_trainset()
    algo.fit(trainset)

    joblib.dump(algo, 'svd_model.pkl')
    experiment.checkpoint(
            path="svd_model.pkl",
            step=num_epochs,
            metrics={"RMSE": rmse_mean, "MAE": mae_mean},
            primary_metric=("RMSE", "minimize"),
    )
    experiment.stop()
    experiments = keepsake.experiments.list()
    experiments.scatter(param="learning_rate", metric="RMSE")
    experiments.plot(metric="RMSE")

if __name__ == "__main__":
    train(0.001, 30)