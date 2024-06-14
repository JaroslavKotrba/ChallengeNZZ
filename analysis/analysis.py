# ANALYSIS

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 100)

import os

os.getcwd()
os.listdir()

### DATA
data = pd.read_json("../data/dataset_propensity.json")
print(data.shape)
print(data.columns)

### FEATURES
df = data.copy()

# Categorical and Numerical features
df.dtypes
catvars = df.select_dtypes(include="object")
numvars = df.select_dtypes(include=["int32", "int64", "float32", "float64"])

# $buy (y)
df["buy"].value_counts()
df = df[df["buy"].notnull()]  # y cannot be null

fig = px.pie(
    values=df["buy"].value_counts(),
    names=["NotBought", "Bought"],
    width=700,
    height=400,
    color_discrete_sequence=["black", "skyblue"],
    title="Bought vs NotBought subscribtion",
)
fig.show()

### CATEGORICAL


def contingency_table(df, feature):
    table = pd.crosstab(df[feature], df["buy"], margins=True, margins_name="Total")
    table["Percentage"] = round((table[1] / table["Total"]) * 100, 2)
    total_row = table.loc["Total"]
    table = table.drop("Total")
    sorted_table = table.sort_values(by=[1, "Percentage"], ascending=False)
    sorted_table.loc["Total"] = total_row

    return sorted_table


def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = "Other"
    return categorical_map


def plot_percentage(df, feature):
    # Generate the contingency table
    contingency = contingency_table(df, feature)
    contingency = contingency.drop("Total")

    # Create a bar plot with Plotly graph objects
    fig = go.Figure(
        data=[
            go.Bar(
                x=contingency.index,
                y=contingency["Percentage"],
                text=contingency["Percentage"],
                textposition="auto",
                marker=dict(color="black"),
                hovertemplate="Category: %{x}<br>Percentage of Purchases: %{y:.2f}%<extra></extra>",
            )
        ]
    )

    # Update the layout for better readability
    fig.update_layout(
        title=f"Percentage of Purchases by {feature}",
        xaxis_title=feature,
        yaxis_title="Purchase Percentage",
        xaxis_tickangle=-90,
        height=600,
        width=1000,
        plot_bgcolor="rgba(0, 0, 0, 0)",
    )

    fig.show()


# $langNew
df["langNew"].value_counts()
contingency_table(df, "langNew")
plot_percentage(df, "langNew")

# $mostLikedCategories
df["mostLikedCategories"].value_counts()

df[df["mostLikedCategories"].isna()]
df["mostLikedCategories"] = df["mostLikedCategories"].replace("", "Empty")
df["mostLikedCategories"] = df["mostLikedCategories"].fillna("NotAvailable")

feature_map = shorten_categories(df.mostLikedCategories.value_counts(), 500)
df["mostLikedCategories"] = df["mostLikedCategories"].map(feature_map)
contingency_table(df, "mostLikedCategories")
plot_percentage(df, "mostLikedCategories")

# $operatingSystemNew
df["operatingSystemNew"].value_counts()
feature_map = shorten_categories(df.operatingSystemNew.value_counts(), 1000)
df["operatingSystemNew"] = df["operatingSystemNew"].map(feature_map)
contingency_table(df, "operatingSystemNew")
plot_percentage(df, "operatingSystemNew")

# $preferredTimeOfDay
df["preferredTimeOfDay"].value_counts()
contingency_table(df, "preferredTimeOfDay")
plot_percentage(df, "preferredTimeOfDay")

### NUMERICAL

# Correlation
plt.figure(figsize=(16, 10))
sns.heatmap(numvars.corr(), cmap="plasma")
plt.show


def plot_distribution(df, column):
    if column not in df.columns:
        raise KeyError(f"The column '{column}' does not exist in the DataFrame.")

    fig = go.Figure(
        data=[
            go.Histogram(
                x=df[column], nbinsx=20, marker=dict(color="black"), opacity=0.75
            )
        ]
    )

    fig.update_layout(
        title=f"Distribution of {column}",
        xaxis_title=column,
        yaxis_title="Frequency",
        height=600,
        width=1000,
        template="simple_white",
    )

    fig.show()


# $active_days
df["active_days"].value_counts()
plot_distribution(df, "active_days")

# $clicks_from_newsletters_total
df["clicks_from_newsletters_total"].value_counts()
df["clicks_from_newsletters_total"] = df["clicks_from_newsletters_total"].fillna(0)


### DF FINAL
df_final = df[
    [
        "langNew",
        "mostLikedCategories",
        "operatingSystemNew",
        "preferredTimeOfDay",
        "active_days",
        "afternoon",
        "bing",
        "buy",
        "clicks_from_newsletters_total",
        "contentTypeArticles",
        "contentTypeGalleries",
        "contentTypeRegularArticles",
        "contentTypeVideos",
        "cookiesNumber",
        "days_since_registration",
        "evening",
        "facebook",
        "flagBildstrecke",
        "flagErklart",
        "flagGastkommentar",
        "flagGlosse",
        "flagInterview",
        "flagKolumne",
        "flagKommentar",
        "flagLive",
        "flagPromotedContent",
        "flagPublireportage",
        "flagQuiz",
        "flagVideo",
        "flagWettbewerb",
        "google",
        "isMobile",
        "morning",
        "night",
        "noon",
        "notWeekend_active_days",
        "notWeekend_sessions",
        "num_display_articles",
        "num_of_sessions",
        "num_read_articles",
        "number_of_newsletters",
        "nzz",
        "other",
        "paygate_impressions",
        "time_from_last_session",
        "twitter",
        "usageBellevue",
        "usageGames",
        "usageMyNZZ",
        "usageNZZAS",
        "usageNZZWeather",
        "usageNewsletters",
        "usageReaderDiscussions",
        "usageUserAccount",
        "usageVideos",
        "weekend_active_days",
        "weekend_sessions",
        "yahoo",
    ]
]

# Remove NAs
df_final[df_final["contentTypeArticles"].isna()]
print(df_final.isnull().sum())
df_final = df_final.dropna()

### SAVE
df_final.to_csv("../data/data_clean.csv", index=False)
