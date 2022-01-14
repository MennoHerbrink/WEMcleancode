from datetime import strptime, time

import numpy as np
import pandas as pd
import unidecode
from sklearn.linear_model import LinearRegression

supply_words = ["pan", "rasp", "kom"]


def normalize_text(text):
    unaccented_text = unidecode.unidecode(text.lower())
    return unaccented_text.split()


def read_recipes():
    recipes = pd.read_csv(
        "data/lunch_recipes.csv", parse_dates=["date"], usecols=["date", "recipe"]
    )
    for wrd in supply_words:
        recipes[f"{wrd}"] = recipes.recipe.apply(
            lambda text: normalize_text(text).count(wrd) > 0
        )
        recipes[f"{wrd}"] = recipes[f"{wrd}"].apply(lambda x: x is True)
    recipes = recipes.drop("recipe", axis=1)
    return recipes


def read_attendance():
    attendance = pd.read_csv("data/key_tag_logs.csv", parse_dates=["timestamp"])
    attendance["date"] = attendance.timestamp.apply(lambda x: x.date())
    attendance["time"] = attendance.timestamp.apply(lambda x: x.time())

    attendance_dates = pd.DataFrame(
        np.array(attendance.date), columns=["date"]
    ).drop_duplicates()
    for name in attendance.name.unique():
        lunchdates = []
        for datum in attendance.date.unique():
            names = attendance[attendance.name == name]
            names = names[names.date == datum]

            dataframe_check_in = names[names.event == "check in"]
            dataframe_check_in = dataframe_check_in[
                dataframe_check_in.time < time(12, 0, 0)
            ]

            df_check_out = names[names.event == "check out"]
            df_check_out = df_check_out[df_check_out.time > time(12, 0, 0)]
            if df_check_out.shape[0] > 0 and dataframe_check_in.shape[0] > 0:
                lunchdates.append(datum)

        attendance_dates[f"{name}"] = attendance_dates.date.apply(
            lambda x: 1 if x in list(lunchdates) else 0
        )
    return attendance_dates.apply(lambda x: x.date())


def read_dishwasher():
    dishwaher = pd.read_csv("data/dishwasher_log.csv")
    dishwaher["date"] = dishwaher.date.apply(lambda x: strptime(x, "%Y-%m-%d"))


def train_model(alpha=0.1):
    recipes = read_recipes()
    attendance = read_attendance()
    dishwaher = read_dishwasher()

    train_data = (
        recipes.merge(attendance, on="date", how="outer").merge(dishwaher).fillna(0)
    )
    train_data = train_data.drop("date", axis=1)
    target = "dishwashers"
    reg = LinearRegression(fit_intercept=False, positive=True).fit(
        train_data.drop(target, axis=1), train_data[target]
    )
    return dict(zip(reg.feature_names_in_, [round(c, 3) for c in reg.coef_]))


if __name__ == "__main__":

    print(train_model())
