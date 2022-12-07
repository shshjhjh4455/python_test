# Surprise 패키지를 이용한 추천 시스템 구현

from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
from surprise.model_selection import cross_validate
import pandas as pd
from surprise import Reader
from surprise.dataset import DatasetAutoFolds



# 데이터 로드
data = Dataset.load_builtin("ml-100k")
trainset, testset = train_test_split(data, test_size=0.25, random_state=0)
svd = SVD(random_state=0)
svd.fit(trainset)

predictions = svd.test(testset)

# print([(pred.uid, pred.iid, pred.est) for pred in predictions[:3]])
# uid = str(196)
# iid = str(302)
# pred = svd.predict(uid, iid, r_ui=4, verbose=True)
# print(pred)

# accuracy.rmse(predictions)

ratings = pd.read_csv("data/ml-latest/ratings.csv")
ratings.to_csv("data/ml-latest/ratings_noH.csv", index=False, header=False)

reader = Reader(
    line_format="user item rating timestamp", sep=",", rating_scale=(0.5, 5.0)
)
data = Dataset.load_from_file("data/ml-latest/ratings_noH.csv", reader=reader)

trainset, testset = train_test_split(data, test_size=0.25, random_state=0)
svd = SVD(n_factors=50, random_state=0)
svd.fit(trainset)
predictions = svd.test(testset)
print(accuracy.rmse(predictions))


ratings = pd.read_csv("data/ml-latest/ratings.csv")
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
svd = SVD(random_state=0)
cross_validate(svd, data, measures=["RMSE", "MAE"], cv=5, verbose=True)

param_grid = {"n_epochs": [5, 10], "lr_all": [0.002, 0.005], "n_factors": [50, 100, 200]}
gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=5)
gs.fit(data)

print(gs.best_score["rmse"])
print(gs.best_params["rmse"])

reader = Reader(
    line_format="user item rating timestamp", sep=",", rating_scale=(0.5, 5.0)
)
data_folds = DatasetAutoFolds(ratings_file="data/ml-latest/ratings_noH.csv", reader=reader)
train = data_folds.build_full_trainset()

svd = SVD(n_epochs=50,random_state=0)
svd.fit(train)

movies = pd.read_csv("data/ml-latest/movies.csv")
moviesIds = ratings[ratings["userId"] == 9]["movieId"]

def get_unseen_surprise(ratings, movies, userId):
    seen_movies = ratings[ratings["userId"] == userId]["movieId"].tolist()
    total_movies = movies["movieId"].tolist()
    unseen_movies = [movie for movie in total_movies if movie not in seen_movies]
    print("평가한 영화 수: ", len(seen_movies), "추천 대상 영화 수: ", len(unseen_movies), "전체 영화 수: ", len(total_movies))
    return unseen_movies

def recomm_movie_by_surprise(svd, userId, unseen_movies, top_n=10):
    predictions = [svd.predict(str(userId), str(movieId)) for movieId in unseen_movies]
    def sortkey_est(pred):
        return pred.est
    predictions.sort(key=sortkey_est, reverse=True)
    top_predictions = predictions[:top_n]
    top_movie_ids = [int(pred.iid) for pred in top_predictions]
    top_movie_rating = [pred.est for pred in top_predictions]
    top_movie_titles = movies[movies["movieId"].isin(top_movie_ids)]["title"]
    top_movie_preds = [(id, title, rating) for id, title, rating in zip(top_movie_ids, top_movie_titles, top_movie_rating)]
    return top_movie_preds

unseen_movies = get_unseen_surprise(ratings, movies, 9)
top_movie_preds = recomm_movie_by_surprise(svd, 9, unseen_movies, top_n=10)
for top_movie in top_movie_preds:
    print(top_movie)
    