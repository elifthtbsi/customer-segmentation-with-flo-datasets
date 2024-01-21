import pandas as pd
from datetime import timedelta

df = pd.read_csv("flo_data_20k.csv")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)


def create_rfm_table(dataframe):

    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_online"] + dataframe["customer_value_total_ever_offline"]

    for col in dataframe.columns:
        if 'date' in col:
            dataframe[col] = pd.to_datetime(dataframe[col], format='%Y-%m-%d')

    analysis_date = dataframe["last_order_date"].max() + timedelta(days=2)
    rfm = dataframe.groupby("master_id").agg({"last_order_date": lambda date:(analysis_date - date.max()).days,
                                              "order_num_total": lambda order_num_total: order_num_total,
                                              "customer_value_total": lambda customer_value_total: customer_value_total})

    rfm.columns = ["recency", "frequency", "monetary"]

    rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

    rfm["rf_score"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

    seg_map = {r'[1-2][1-2]': 'hibernating',
               r'[1-2][3-4]': 'at_Risk',
               r'[1-2]5': 'cant_loose',
               r'3[1-2]': 'about_to_sleep',
               r'33': 'need_attention',
               r'[3-4][4-5]': 'loyal_customers',
               r'41': 'promising',
               r'51': 'new_customers',
               r'[4-5][2-3]': 'potential_loyalists',
               r'5[4-5]': 'champions'}

    rfm["segment"] = rfm["rf_score"].replace(seg_map, regex=True)
    rfm.reset_index(inplace=True)

    return rfm


rfm = create_rfm_table(df)

target_segment_for_female = rfm[(rfm["segment"] == "loyal_customers") | (rfm["segment"] == "champions")]

merged_df_female = pd.merge(df, target_segment_for_female, on="master_id")

target_customers_female = merged_df_female[merged_df_female["interested_in_categories_12"].apply(lambda categories: "KADIN" in categories)]

target_customers_female["master_id"].to_csv("target_customers_female.csv", index=False)

target_segment_for_male_kids = rfm[(rfm["segment"] == "hibernating") | (rfm["segment"] == "about_to_sleep") | (rfm["segment"] == "new_customers")]

merged_df_male_kids = pd.merge(df, target_segment_for_male_kids, on="master_id")

target_customers_male_kids = merged_df_male_kids[merged_df_male_kids["interested_in_categories_12"].apply(lambda categories: ("COCUK" in categories) | ("ERKEK" in categories))]

target_customers_male_kids["master_id"].to_csv("target_customers_male_kids.csv", index=False)