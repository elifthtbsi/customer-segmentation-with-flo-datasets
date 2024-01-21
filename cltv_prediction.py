import pandas as pd
from datetime import timedelta
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

dataframe = pd.read_csv('flo_data_20k.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)


def create_cltv_prediction(df, time):

    #Veri ön hazırlık aşaması
    columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_online",
               "customer_value_total_ever_offline"]
    for col in columns:
        replace_with_thresholds(df, col)

    df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]


    for col in df.columns:
        if 'date' in col:
            df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')

    analysis_date = df["last_order_date"].max() + timedelta(days=2)

    df["first_order_date"] = pd.to_datetime(df["first_order_date"])
    df["last_order_date"] = pd.to_datetime(df["last_order_date"])


    #CLTV yapısını oluşturma

    cltv_df = pd.DataFrame()
    cltv_df["customer_id"] = df["master_id"]
    cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).dt.days) / 7
    cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).dt.days) / 7
    cltv_df["frequency"] = df["order_num_total"]
    cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]
    cltv_df = cltv_df[cltv_df['frequency'] > 1]


    #BG-NBD modelinin kurulması

    bgf = BetaGeoFitter(penalizer_coef=0.001)

    bgf.fit(cltv_df['frequency'],
            cltv_df["recency_cltv_weekly"],
            cltv_df["T_weekly"])


    #input olarak girilen süre (ay cinsinden) içerisinde beklenen satın almanın hesaplanması
    cltv_df["exp_sales_6_month"] = bgf.predict(4*time,
                                               cltv_df["frequency"],
                                               cltv_df["recency_cltv_weekly"],
                                               cltv_df["T_weekly"])


    ggf = GammaGammaFitter(penalizer_coef=0.01)

    ggf.fit(cltv_df['frequency'],
            cltv_df['monetary_cltv_avg'])

    cltv_df["exp_avg_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                       cltv_df["monetary_cltv_avg"])

    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df["frequency"],
                                       cltv_df["recency_cltv_weekly"],
                                       cltv_df["T_weekly"],
                                       cltv_df["monetary_cltv_avg"],
                                       time=time,
                                       freq="W",
                                       discount_rate=0.01)

    cltv_df["cltv"] = cltv
    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=['D', 'C', 'B', 'A'])

    return cltv_df

#6 aylık süre için cltv tahminleme işleminin yapılması
cltv_df = create_cltv_prediction(dataframe,6)

print(cltv_df.head(20))
