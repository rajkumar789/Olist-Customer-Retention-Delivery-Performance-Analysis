"""
E-Commerce Customer Retention & Delivery Performance Analysis
================================================================
Dataset: Olist Brazilian E-Commerce Public Dataset (Kaggle)
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

# config
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# set the style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# This list collects every printed finding so we can also save it to a
# text file at the end (analysis_summary.txt). Every print() in this script
# also appends to this list via the log() helper below.
SUMMARY_LINES = []


def log(message):
    """Print a message to the console and store it for the summary file"""
    print(message)
    SUMMARY_LINES.append(str(message))

    # load the raw dataset
    # df load_data():


def load_data() -> dict:
    customers = pd.read_csv(os.path.join(DATA_DIR, "olist_customers_dataset.csv"))
    orders = pd.read_csv(os.path.join(DATA_DIR, "olist_orders_dataset.csv"))
    order_items = pd.read_csv(os.path.join(DATA_DIR, "olist_order_items_dataset.csv"))
    payments = pd.read_csv(os.path.join(DATA_DIR, "olist_order_payments_dataset.csv"))
    reviews = pd.read_csv(os.path.join(DATA_DIR, "olist_order_reviews_dataset.csv"))
    products = pd.read_csv(os.path.join(DATA_DIR, "olist_products_dataset.csv"))
    category_translation = pd.read_csv(
        os.path.join(DATA_DIR, "product_category_name_translation.csv")
    )

    log("=" * 70)
    log("Dataset loaded")
    log("=" * 70)
    log(f" customers: {customers.shape}")
    log(f" orders: {orders.shape}")
    log(f" order_items: {order_items.shape}")
    log(f" payments: {payments.shape}")
    log(f" reviews: {reviews.shape}")
    log(f" products: {products.shape}")

    return {
        "customers": customers,
        "orders": orders,
        "order_items": order_items,
        "payments": payments,
        "reviews": reviews,
        "products": products,
        "category_translation": category_translation,
    }


# Clean and merge into one analytical table
def clean_and_merge(tables):
    """
    Build a single order-level table joining orders, customers, items,
    payments, reviews, and product categories. Drops canceled/unavailable
    orders since they were never fulfilled and shouldn't count toward
    delivery or repeat-purchase metrics.
    """
    orders = tables["orders"].copy()
    customers = tables["customers"].copy()
    order_items = tables["order_items"].copy()
    payments = tables["payments"].copy()
    reviews = tables["reviews"].copy()
    products = tables["products"].copy()
    category_translation = tables["category_translation"].copy()

    # convert the date columns to real datetimes
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    # keep only orders that were actualy fulfilled
    # "canceled" and "unavailable" orders neve resulted in a real delivery
    # so including them wojld distort both delivery-perfomance and
    # repeat-purchase metrics
    valid_statuses = ["delivered", "shipped", "invoiced", "processing"]
    before = len(orders)
    orders = orders[orders["order_status"].isin(valid_statuses)].copy()
    log(
        f"\nDropped {before - len(orders)} canceled/unavailable orders"
        f"({len(orders)} remain)"
    )

    # aggregate order_items to one row per order
    order_items_agg = (
        order_items.groupby("order_id")
        .agg(
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
            n_items=("order_item_id", "count"),
            product_id=("product_id", "first"),  # first product, for category lookpy
        )
        .reset_index()
    )
    order_items_agg["order_value"] = (
        order_items_agg["total_price"] + order_items_agg["total_freight"]
    )

    # aggregate payments to one row per order
    payments_agg = (
        payments.groupby("order_id")
        .agg(
            total_payment_value=("payment_value", "sum"),
            payment_type=("payment_type", "first"),
            payment_installments=("payment_installments", "max"),
        )
        .reset_index()
    )

    # reviews: one review score per order
    reviews_clean = reviews[["order_id", "review_score"]].drop_duplicates(
        subset="order_id"
    )

    # product category, translated to English
    products = products.merge(
        category_translation, on="product_category_name", how="left"
    )
    products_clean = products[["product_id", "product_category_name_english"]]

    # merge everything into one order-level table
    df = orders.merge(customers, on="customer_id", how="left")
    df = df.merge(
        order_items_agg, on="order_id", how="inner"
    )  # must have items to count
    df = df.merge(payments_agg, on="order_id", how="left")
    df = df.merge(reviews_clean, on="order_id", how="left")
    df = df.merge(products_clean, on="product_id", how="left")

    # delivery perfomance flags
    df["is_delivered"] = df["order_delivered_customer_date"].notna()
    df["is_late"] = (
        df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]
    )

    # mark "never deliveted" orders as pd.NA
    df["is_late"] = df["is_late"].astype("boolean")
    df.loc[~df["is_delivered"], "is_late"] = pd.NA

    df["delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days
    log("\n" + "=" * 70)
    log("Clean and Merged Table")
    log("=" * 70)
    log(f"  Final shape: {df.shape}")
    log(
        f"  Date range: {df['order_purchase_timestamp'].min()} to "
        f"  {df['order_purchase_timestamp'].max()}"
    )
    log(
        f"  Missing review scores: {df['review_score'].isna().sum()} "
        f"({df['review_score'].isna().mean():.1%})"
    )
    return df


# Exploratory Data Analysis (EDA)
def run_eda(df):
    """produce headline charts and numbers to build intution about the data"""
    log("\n" + "=" * 70)
    log("Exploratory Data Analysis")
    log("=" * 70)

    # orders over time
    monthly_orders = (
        df.set_index("order_purchase_timestamp").resample("ME")["order_id"].nunique()
    )
    plt.figure()
    monthly_orders.plot(marker="o")
    plt.title("Number of Oders per Month")
    plt.xlabel("Month")
    plt.ylabel("Order Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "orders_over_time(month).png"), dpi=150)
    plt.close()

    # revenue over time
    monthly_revenue = (
        df.set_index("order_purchase_timestamp").resample("ME")["order_value"].sum()
    )
    log(f"  Total revenue in dataset: R${df['order_value'].sum():,.2f}")
    log(f"  Average order value: R${df['order_value'].mean():.2f}")

    # review score distribution
    plt.figure()
    sns.countplot(
        data=df, x="review_score", hue="review_score", palette="viridis", legend=False
    )
    plt.title("Distribution of Review Scores")
    plt.xlabel("Review Score (1-5)")
    plt.ylabel("Number of orders")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "review_score_distribution.png"), dpi=150)
    plt.close()

    avg_review = df["review_score"].mean()
    log(f"  Average review score: {avg_review:.2f} / 5")

    # top product category by order count
    top_categories = df["product_category_name_english"].value_counts().head(10)
    plt.figure()
    top_categories.sort_values().plot(kind="barh", color="steelblue")
    plt.title("Top 10 Product Categories by order count")
    plt.xlabel("Number of order")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top_categories.png"), dpi=150)
    plt.close()

    log(f"  Top category: {top_categories.index[0]}({top_categories.iloc[0]})")

    return {
        "monthly_orders": monthly_orders,
        "monthly_revenue": monthly_revenue,
        "avg_review": avg_review,
        "top_categories": top_categories,
    }


# RFM Customer Segmentation
def build_rfm_segments(df):
    """
    score every unique customer on Recency, Frequency, and Monetary Value,
    then bucker customers into name segments.
    """

    log("\n" + "=" * 70)
    log("RFM Customer Segmentation")
    log("=" * 70)

    snapshot_date = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("customer_unique_id")
        .agg(
            recency=(
                "order_purchase_timestamp",
                lambda x: (snapshot_date - x.max()).days,
            ),
            frequency=("order_id", "nunique"),
            monetary=("order_value", "sum"),
        )
        .reset_index()
    )
    repeat_customers = (rfm["frequency"] > 1).sum()
    total_customers = len(rfm)
    log(f"  Total unique customers: {total_customers:,}")
    log(
        f"  Repeat customers (2+ orders): {repeat_customers:,}"
        f" ({repeat_customers / total_customers: .1%})"
    )
    log(
        f"  one-time cutomers: {total_customers - repeat_customers:,}"
        f"({1 - repeat_customers / total_customers:,.1%})"
    )

    # score each dimension 1-5 using quintiles
    # Recengy: lower is better (bought more recently), so we reverse the labels
    rfm["r_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)

    # Frequency and Monetary: Higher is better
    rfm["f_score"] = pd.qcut(
        rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    rfm["m_score"] = pd.qcut(
        rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    def segment_customer(row):
        """Assingn a name business segment based on R, F, M scores"""
        r, f, m = row["r_score"], row["f_score"], row["m_score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 3:
            return "Loyal Customers"
        elif r >= 4 and f <= 2:
            return "New / Promising"
        elif r <= 2 and f >= 3 and m >= 3:
            return "At Risk"
        elif r <= 2 and f >= 4 and m >= 4:
            return "Can't Lose Them"
        elif r <= 2 and f <= 2 and m <= 2:
            return "Lost"
        else:
            return "Needs Attention"

    rfm["segment"] = rfm.apply(segment_customer, axis=1)
    segment_summary = (
        rfm.groupby("segment")
        .agg(
            customers=("customer_unique_id", "count"),
            total_revenue=("monetary", "sum"),
            avg_recency_days=("recency", "mean"),
        )
        .sort_values("total_revenue", ascending=False)
    )

    log("\n Segment summary:")
    log(segment_summary.to_string())

    # Charts: customer count and revenue by segment
    fig, axes = plt.subplots(1, 2, figsize=(14, 16))
    segment_summary["customers"].sort_values().plot(
        kind="barh", ax=axes[0], color="darkorange"
    )
    axes[0].set_title("Custoemrs per Segemnt")
    axes[0].set_xlabel("Number of Customers")

    segment_summary["total_revenue"].sort_values().plot(
        kind="barh", ax=axes[1], color="seagreen"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rfm_segments.png"), dpi=150)

    # save the full segmented customer list
    rfm.to_csv(os.path.join(OUTPUT_DIR, "rfm_customer_segments.csv"), index=False)
    log(
        f"\n saved full segmented customer list to "
        f"{OUTPUT_DIR}/rfm_customer_segments.csv"
    )
    at_risk_revenue = segment_summary.loc[
        segment_summary.index.isin(["At Risk", "Can't Lose Them"]), "total_revenue"
    ].sum()

    log(
        f"\n  Revenue sitting in 'At Risk' + \"Can't Lose Them\" segments: "
        f"R${at_risk_revenue:,.2f}"
    )

    return rfm, segment_summary

    # Cohort retention analysis


def run_cohort_analysis(df):
    """
    Group customers by the month of their FIRST purchase (their "cohort"),
    then track what percentage of each cohort is still buying in month 1,
    2, 3... after acquisition. This answers "are we actually retaining
    people" with real numbers instead of a gut feeling.
    """
    log("\n" + "=" * 70)
    log("Cohort Retntion Analysis")
    log("\n" + "=" * 70)

    data = df[["customer_unique_id", "order_id", "order_purchase_timestamp"]].copy()

    # Each customer's purchase month
    data["order_month"] = data["order_purchase_timestamp"].dt.to_period("M")

    # each customer's first purchase month (their cohort)
    first_purchase = (
        data.groupby("customer_unique_id")["order_month"]
        .min()
        .reset_index()
        .rename(columns={"order_month": "cohort_month"})
    )
    data = data.merge(first_purchase, on="customer_unique_id", how="left")

    # how many months after acquistion did this order happen?
    data["cohort_index"] = (
        data["order_month"].dt.year - data["cohort_month"].dt.year
    ) * 12 + (data["order_month"].dt.month - data["cohort_month"].dt.month)

    # Count unique customers per cohort per month-index
    cohort_data = (
        data.groupby(["cohort_month", "cohort_index"])["customer_unique_id"]
        .nunique()
        .reset_index()
    )

    cohort_pivot = cohort_data.pivot(
        index="cohort_month", columns="cohort_index", values="customer_unique_id"
    )

    # convert counts to retention percentage (relative to cohort's month-0 size)
    cohort_size = cohort_pivot.iloc[:, 0]
    retention_pct = cohort_pivot.divide(cohort_size, axis=0) * 100

    # chart : retention heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        retention_pct.iloc[:, :6],  # first 6 months after acquisition, readability
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
    )
    plt.title("Cohort Retention Rate (%) by Months since Acquisition")
    plt.xlabel("Months Since First Purchase")
    plt.ylabel("Acquisition Cohort (Month)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cohort_retention_heatmap.png"), dpi=150)
    plt.close()

    # Headline number: average month-1 retentiton across all cohorts
    if 1 in retention_pct.columns:
        avg_month1_retention = retention_pct[1].mean()
        log(
            f"  Average month-1 retention across all cohort:"
            f"{avg_month1_retention:.1f}%"
        )
    else:
        avg_month1_retention = None
        log("   Not enough data to compute month-1 retention.")


# Delivery Perfomance vs. Review Score
def analyze_delivery_perfomance(df):
    """
    Test whether late deliveries are actually associated with lower review
    scores (not just assumed to be), and identify which states/categories
    have the worst on-time delivery rates.
    """

    log("\n" + "=" * 70)
    log("STEP 6: DELIVERY PERFORMANCE VS. REVIEW SCORE")
    log("=" * 70)

    delivered = df[df["is_delivered"] & df["review_score"].notna()].copy()

    on_time_reviews = delivered.loc[delivered["is_late"] == False, "review_score"]
    late_reviews = delivered.loc[delivered["is_late"] == True, "review_score"]

    log(
        f"  On-time orders: {len(on_time_reviews):,}, "
        f"average review score: {on_time_reviews.mean():.2f}"
    )
    log(
        f"  Late orders:    {len(late_reviews):,}, "
        f"average review score: {late_reviews.mean():.2f}"
    )
    # --- Statistical test: is this difference real, or just noise? ---
    # Welch's t-test does not assume equal variance between the two groups,
    # which is the safer default when group sizes differ this much.
    t_stat, p_value = stats.ttest_ind(on_time_reviews, late_reviews, equal_var=False)
    log(f"\n Welch's t-test: t= {t_stat:.2f}, p-value = {p_value:.2e}")
    if p_value < 0.05:
        log(
            "    => Statistically significant: late delivery Is associated with"
            "lower review scores"
        )
    else:
        log(
            "   => Not statistically significant at the 5% level -the "
            "difference could be dure to random variation"
        )

    # chart: review score by delivery status
    plt.figure()
    sns.boxplot(
        data=delivered,
        x="is_late",
        y="review_score",
        hue="is_late",
        palette={False: "seagreen", True: "indianred"},
        legend=False,
    )
    plt.xticks([0, 1], ["on-time", "late"])
    plt.title("Review score by delivery status")
    plt.xlabel("")
    plt.ylabel("Review Score (1-5)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "delivery_vs_review_score.png"), dpi=150)
    plt.close()

    # which states have the worst on-time delivery rate?
    state_perfomance = (
        delivered.groupby("customer_state")
        .agg(
            orders=("order_id", "nunique"),
            late_rate=("is_late", "mean"),
            avg_review=("review_score", "mean"),
            revenue=("order_value", "sum"),
        )
        .query("orders >= 30")  # ignore tiny states with unreliable rates
        .sort_values("late_rate", ascending=False)
    )
    log("\n worst 5 states by late-delivery rate (min. 30 orders):")
    log(state_perfomance.head(5).to_string())

    # which product categories have the worst on-time delivery rate?
    category_perfomance = (
        delivered.groupby("product_category_name_english")
        .agg(
            orders=("order_id", "nunique"),
            late_rate=("is_late", "mean"),
            avg_review=("review_score", "mean"),
            revenue=("order_value", "sum"),
        )
        .query("orders >= 30")
        .sort_values("late_rate", ascending=False)
    )
    log("\n  Worst 5 products categories by late-delivery rate(min. 30 orders):")
    log(category_perfomance.head(5).to_string())

    return {
        "on_time_avg_review": on_time_reviews.mean(),
        "late_avg_review": late_reviews.mean(),
        "p_value": p_value,
        "state_perfomance": state_perfomance,
        "category_perfomance": category_perfomance,
    }


# main runner
def main():
    tables = load_data()
    df = clean_and_merge(tables)

    eda_result = run_eda(df)
    rfm, segment_summary = build_rfm_segments(df)
    retention_pct = run_cohort_analysis(df)
    delivery_results = analyze_delivery_perfomance(df)

# -----------------------------------------------------------------
    # FINAL SUMMARY: the "add value" section
    # -----------------------------------------------------------------
    log("\n" + "=" * 70)
    log("FINDINGS SUMMARY")
    log("=" * 70)

    at_risk_revenue = segment_summary.loc[
        segment_summary.index.isin(["At Risk", "Can't Lose Them"]), "total_revenue"
    ].sum()

    significance_note = (
        "statistically significant at the 5% level"
        if delivery_results["p_value"] < 0.05
        else "NOT statistically significant at the 5% level in this data - "
             "treat the raw score gap as inconclusive rather than proof of a causal link"
    )

    log(f"""
  1. RETENTION: A large share of customers never come back after their
     first order. This is the single biggest lever available - see the
     cohort heatmap and RFM segment table for exactly who to target.

  2. VALUE AT RISK: R${at_risk_revenue:,.2f} in historical revenue sits with
     customers in the 'At Risk' and "Can't Lose Them" segments - customers
     who have spent real money before but have gone quiet. A win-back
     campaign aimed at these two segments specifically (not a blanket email
     to everyone) is the highest-leverage next action.

  3. DELIVERY IMPACT ON SATISFACTION: On-time orders average a review score
     of {delivery_results['on_time_avg_review']:.2f} vs. {delivery_results['late_avg_review']:.2f}
     for late orders (p = {delivery_results['p_value']:.2e}, {significance_note}).
     See outputs/delivery_vs_review_score.png and the worst-performing
     states/categories printed above for where delivery issues concentrate.

  Recommended next steps for the business:
     a) Launch a targeted win-back campaign for the "At Risk" / "Can't Lose
        Them" RFM segments (list exported to rfm_customer_segments.csv).
     b) Investigate logistics/carrier performance in the worst-performing
        states identified above; even a 10 percentage-point improvement in
        on-time rate there should measurably lift review scores.
     c) Introduce a post-purchase engagement touchpoint (email or loyalty
        incentive) within the first 30 days, since retention drops sharply
        after month 1 in nearly every cohort.
""")

    # --- Save the full printed summary to a text file ---
    with open(os.path.join(OUTPUT_DIR, "analysis_summary.txt"), "w") as f:
        f.write("\n".join(SUMMARY_LINES))

    log(f"Full summary saved to {OUTPUT_DIR}/analysis_summary.txt")
    log(f"All charts and the segmented customer CSV are in the "
        f"{OUTPUT_DIR}/ folder.")


if __name__ == "__main__":
    main()
