#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# Load and Prepare Dataset
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ratings_.csv", header=None)

    df.columns = [
        "user_id",
        "prod_id",
        "rating",
        "timestamp"
    ]

    # Remove timestamp
    df.drop("timestamp", axis=1, inplace=True)

    # Active users
    user_counts = df["user_id"].value_counts()
    active_users = user_counts[
        user_counts >= 50
    ].index

    # Popular products
    prod_counts = df["prod_id"].value_counts()
    popular_products = prod_counts[
        prod_counts >= 5
    ].index

    df = df[
        (df.user_id.isin(active_users)) &
        (df.prod_id.isin(popular_products))
    ]

    return df


# ------------------------------
# User Similarity Matrix
# ------------------------------
@st.cache_resource
def build_similarity(df):

    user_item = df.pivot_table(
        index="user_id",
        columns="prod_id",
        values="rating",
        fill_value=0
    )

    sim = cosine_similarity(user_item)

    similarity_df = pd.DataFrame(
        sim,
        index=user_item.index,
        columns=user_item.index
    )

    return user_item, similarity_df


# ------------------------------
# Recommendation Function
# ------------------------------
def recommend_products(
    user_id,
    df,
    similarity_df,
    top_n=5
):

    similar_users = (
        similarity_df[user_id]
        .sort_values(
            ascending=False
        )[1:6]
        .index
    )

    user_seen = set(
        df[
            df.user_id == user_id
        ]["prod_id"]
    )

    candidate = df[
        df.user_id.isin(
            similar_users
        )
    ]

    recommendations = (
        candidate.groupby(
            "prod_id"
        )["rating"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    recommendations = recommendations[
        ~recommendations.index.isin(
            user_seen
        )
    ]

    return recommendations.head(
        top_n
    )


# ------------------------------
# Streamlit App
# ------------------------------
st.set_page_config(
page_title="Amazon Product Recommender",
layout="wide"
)

st.title(
"Amazon Collaborative Filtering Recommendation System"
)

st.write(
"Recommend products based on similar users using cosine similarity."
)

# Load data
df = load_data()
user_item, similarity_df = build_similarity(df)

users = sorted(
    df["user_id"].unique()
)

selected_user = st.selectbox(
"Select User ID",
users
)

if st.button(
"Get Recommendations"
):

    recs = recommend_products(
        selected_user,
        df,
        similarity_df
    )

    st.subheader(
    "Recommended Products"
    )

    for i, product in enumerate(
        recs.index,
        1
    ):
        st.write(
        f"{i}. Product ID: {product}"
        )

# Dataset info
st.sidebar.header(
"Dataset Summary"
)

st.sidebar.write(
f"Users: {df.user_id.nunique()}"
)

st.sidebar.write(
f"Products: {df.prod_id.nunique()}"
)

st.sidebar.write(
f"Ratings: {len(df)}"
)

