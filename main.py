from datetime import *
import unidecode

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, RidgeCV

supply_words = ["pan", "rasp", "kom"]

def normalize_text(text):
    unaccented_text = unidecode.unidecode(text.lower())
    return unaccented_text.split()


def getRecipesDF():
    recipes = pd.read_csv("data/lunch_recipes.csv", parse_dates=["date"], usecols = ["date", "recipe"])  # Read lunch recipes dataframe.
    for wrd in supply_words:
        recipes[f"{wrd}"] = recipes.recipe.apply(
            lambda text: normalize_text(text).count(wrd) > 0
        )  ## count the amount of times a word occurs in the recipe.
        recipes[f"{wrd}"] = recipes[f"{wrd}"].apply(lambda x: x is True)
    recipes = recipes.drop("recipe", axis=1)
    return recipes


def read_attendance_sheet():
    attendance = pd.read_csv("data/key_tag_logs.csv", parse_dates=["timestamp"])
    attendance["date"] = attendance.timestamp.apply(lambda x: x.date())
    #attendance['date'] = attendance.timestamp.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    attendance["time"] = attendance.timestamp.apply(lambda x: x.time())

    result = pd.DataFrame(np.array(attendance.date), columns=["date"]).drop_duplicates()
    print(result.dtypes)
    print(attendance.dtypes)
    for name in attendance.name.unique():
        lunchdates = []
        for datum in attendance.date.unique():
            df2 = attendance[attendance.name == name]
            df2 = df2[df2.date == datum]

            dataframe_check_in = df2[df2.event == "check in"]
            dataframe_check_in = dataframe_check_in[
                dataframe_check_in.time < time(12, 0, 0)
            ]

            df_check_out = df2[df2.event == "check out"]
            df_check_out = df_check_out[df_check_out.time > time(12, 0, 0)]
            if df_check_out.shape[0] > 0 and dataframe_check_in.shape[0] > 0:
                lunchdates.append(datum)

        result[f"{name}"] = result.date.apply(
            lambda x: 1 if x in list(lunchdates) else 0
        )
    return result.apply(lambda x: x.date())


def train_model(alpha=0.1):
    recipes = getRecipesDF()
    attendance = read_attendance_sheet()
    l = pd.read_csv("data/dishwasher_log.csv")
    l["date"] = l.date.apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))

    print(recipes.dtypes)
    print(attendance.dtypes)
    df = recipes.merge(attendance, on="date", how="outer").merge(l).fillna(0)
    reg = LinearRegression(fit_intercept=False, positive=True).fit(
        df.drop(["dishwashers", "date"], axis=1), df["dishwashers"]
    )
    return dict(zip(reg.feature_names_in_, [round(c, 3) for c in reg.coef_]))


if __name__ == "__main__":

    print(train_model())
