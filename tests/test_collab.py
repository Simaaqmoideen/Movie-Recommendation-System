import pandas as pd
import pytest

from src.collab import (
    ItemItemCF,
    evaluate_popularity_baseline,
    evaluate_model,
    global_mean_rmse,
    popular_recommendations,
    precision_at_k,
    rmse,
    time_split,
)


def test_time_split_holds_out_latest_timestamps_and_keeps_ties_together():
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2, 3, 3],
            "movieId": [10, 20, 30, 10, 20, 30, 10, 20],
            "rating": [3, 4, 5, 3, 4, 5, 3, 4],
            "timestamp": [1, 2, 3, 1, 2, 2, 7, 7],
        }
    )

    train, test = time_split(ratings, test_frac=0.2)

    assert set(test.loc[test["userId"] == 1, "timestamp"]) == {3}
    assert set(test.loc[test["userId"] == 2, "timestamp"]) == {2}
    assert not (test["userId"] == 3).any()
    assert train.groupby("userId").size().loc[1] >= 1
    assert train.groupby("userId").size().loc[2] >= 1


def test_time_split_rejects_invalid_fraction():
    ratings = pd.DataFrame(
        {"userId": [1], "movieId": [1], "rating": [4.0], "timestamp": [1]}
    )
    with pytest.raises(ValueError, match="between 0 and 1"):
        time_split(ratings, test_frac=1.0)


def test_global_mean_rmse_uses_train_only_mean():
    train = pd.DataFrame(
        {"userId": [1, 2], "movieId": [10, 20], "rating": [3.0, 3.0], "timestamp": [1, 1]}
    )
    test = pd.DataFrame(
        {"userId": [1, 2], "movieId": [20, 10], "rating": [5.0, 1.0], "timestamp": [2, 2]}
    )

    assert global_mean_rmse(train, test) == pytest.approx(2.0)


def test_popular_recommendations_excludes_items_seen_in_training():
    train = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3],
            "movieId": [10, 30, 20, 30, 20, 30],
            "rating": [3, 4, 4, 5, 3, 4],
            "timestamp": [1, 2, 1, 2, 1, 2],
        }
    )

    assert popular_recommendations(train, user_id=1, k=2) == [20]


def test_precision_at_k_uses_fixed_k_denominator():
    assert precision_at_k([10, 20], {10}, k=2) == pytest.approx(0.5)


def test_popularity_evaluation_uses_held_out_relevant_movie():
    train = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3],
            "movieId": [10, 30, 20, 30, 20, 30],
            "rating": [3, 4, 4, 5, 3, 4],
            "timestamp": [1, 2, 1, 2, 1, 2],
        }
    )
    test = pd.DataFrame(
        {"userId": [1], "movieId": [20], "rating": [5.0], "timestamp": [3]}
    )

    result = evaluate_popularity_baseline(train, test, k=1)

    assert result["precision_at_k"] == pytest.approx(1.0)
    assert result["eligible_users"] == 1


def small_training_ratings():
    return pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "movieId": [10, 20, 30, 10, 20, 30, 10, 20, 30],
            "rating": [5, 4, 1, 4, 5, 1, 1, 2, 5],
            "timestamp": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        }
    )


def test_item_item_fit_and_predict_are_bounded_and_finite():
    model = ItemItemCF().fit(small_training_ratings())

    predictions = [model.predict(1, movie_id) for movie_id in [10, 20, 30, 99]]

    assert all(0.5 <= prediction <= 5.0 for prediction in predictions)
    assert model.predict(999, 10) == pytest.approx(model.global_mean)
    assert model.predict(1, 99) == pytest.approx(model.user_mean[1])


def test_item_item_recommendations_exclude_seen_movies():
    train = small_training_ratings()
    train.loc[len(train)] = [2, 40, 5, 4]
    train.loc[len(train)] = [3, 40, 1, 4]
    model = ItemItemCF().fit(train)

    recommendations = model.recommend_for_user(1, n=10)

    assert recommendations["movieId"].tolist() == [40]
    assert not set(recommendations["movieId"]) & set(train.loc[train.userId == 1, "movieId"])


def test_item_item_recommendations_reject_unknown_user_and_invalid_n():
    model = ItemItemCF().fit(small_training_ratings())
    with pytest.raises(KeyError, match="Unknown user"):
        model.recommend_for_user(999)
    with pytest.raises(ValueError, match="positive"):
        model.recommend_for_user(1, n=0)


def test_item_item_recommends_from_ephemeral_new_user_profile():
    model = ItemItemCF().fit(small_training_ratings())
    result = model.recommend_from_profile({10: 5.0, 20: 4.0}, n=5)

    assert result["movieId"].tolist() == [30]
    assert not {10, 20} & set(result["movieId"])
    assert 0.5 <= result.iloc[0]["predicted_rating"] <= 5.0


def test_item_item_profile_rejects_bad_ratings():
    model = ItemItemCF().fit(small_training_ratings())
    with pytest.raises(ValueError, match="between 0.5 and 5.0"):
        model.recommend_from_profile({10: 6.0})


def test_rmse_evaluates_all_rows_in_test():
    train = small_training_ratings()
    test = pd.DataFrame(
        {
            "userId": [1, 2, 3],
            "movieId": [20, 30, 10],
            "rating": [4.0, 1.0, 1.0],
            "timestamp": [4, 4, 4],
        }
    )
    model = ItemItemCF().fit(train)

    score = rmse(model, test)

    assert 0.0 < score < 5.0


def test_evaluate_model_reports_both_baselines():
    train = small_training_ratings()
    train.loc[len(train)] = [2, 40, 5, 4]
    train.loc[len(train)] = [3, 40, 1, 4]
    test = pd.DataFrame(
        {"userId": [1], "movieId": [40], "rating": [5.0], "timestamp": [5]}
    )
    model = ItemItemCF().fit(train)

    result = evaluate_model(model, train, test, k=1)

    assert result["global_mean_rmse"] >= 0
    assert result["item_item_rmse"] >= 0
    assert result["popularity_precision_at_k"] == pytest.approx(1.0)
    assert result["item_item_precision_at_k"] >= 0
