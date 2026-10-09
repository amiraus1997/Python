import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Hotel Booking Analysis", page_icon="🏨", layout="wide")


@st.cache_resource(show_spinner=False)
def load_data(path="hotel_bookings.csv"):
    """Read the CSV in bounded chunks and retain only aggregates plus a preview."""
    use_columns = [
        "hotel",
        "is_canceled",
        "lead_time",
        "arrival_date_month",
        "adults",
        "children",
        "babies",
        "market_segment",
        "adr",
    ]
    column_types = {
        "hotel": "category",
        "is_canceled": "int8",
        "lead_time": "int16",
        "arrival_date_month": "category",
        "adults": "int16",
        "children": "float32",
        "babies": "int16",
        "market_segment": "category",
        "adr": "float32",
    }
    lead_labels = ["0-7 days", "8-30 days", "31-90 days", "91-180 days", "181+ days"]
    group_columns = [
        "hotel",
        "arrival_date_month",
        "market_segment",
        "is_canceled",
        "lead_time_band",
    ]

    aggregate_parts = []
    preview_parts = []
    preview_rows_remaining = 5000

    reader = pd.read_csv(
        path,
        usecols=use_columns,
        dtype=column_types,
        chunksize=10000,
    )

    for chunk in reader:
        chunk["children"] = chunk["children"].fillna(0).astype("int16")
        chunk["total_guests"] = (
            chunk["adults"] + chunk["children"] + chunk["babies"]
        ).astype("int16")
        chunk["lead_time_band"] = pd.cut(
            chunk["lead_time"],
            bins=[-1, 7, 30, 90, 180, np.inf],
            labels=lead_labels,
        )
        chunk = chunk.loc[(chunk["total_guests"] > 0) & (chunk["adr"] >= 0)]

        grouped = (
            chunk.groupby(group_columns, observed=True, as_index=False)
            .agg(
                bookings=("is_canceled", "size"),
                adr_sum=("adr", "sum"),
                lead_time_sum=("lead_time", "sum"),
            )
        )
        aggregate_parts.append(grouped)

        if preview_rows_remaining > 0:
            preview = chunk.head(preview_rows_remaining).copy()
            preview["cancellation_label"] = preview["is_canceled"].map(
                {0: "Not canceled", 1: "Canceled"}
            )
            preview_parts.append(preview)
            preview_rows_remaining -= len(preview)

    summary = pd.concat(aggregate_parts, ignore_index=True)
    summary = (
        summary.groupby(group_columns, observed=True, as_index=False)[
            ["bookings", "adr_sum", "lead_time_sum"]
        ]
        .sum()
    )
    summary["cancellation_label"] = summary["is_canceled"].map(
        {0: "Not canceled", 1: "Canceled"}
    ).astype("category")

    preview = pd.concat(preview_parts, ignore_index=True)
    return summary, preview


summary, preview = load_data()

st.title("Hotel Booking Analysis")
st.caption(
    "Interactive analysis generated from chunked CSV aggregates. "
    "The app never holds the complete source table in memory."
)

st.sidebar.header("Filters")

hotel_options = sorted(summary["hotel"].dropna().astype(str).unique().tolist())
selected_hotels = st.sidebar.multiselect("Hotel", hotel_options, default=hotel_options)

month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
months_present = set(summary["arrival_date_month"].dropna().astype(str))
available_months = [month for month in month_order if month in months_present]
selected_months = st.sidebar.multiselect(
    "Arrival month", available_months, default=available_months
)

segment_options = sorted(summary["market_segment"].dropna().astype(str).unique().tolist())
selected_segments = st.sidebar.multiselect(
    "Market segment", segment_options, default=segment_options
)

status_options = ["Not canceled", "Canceled"]
selected_statuses = st.sidebar.multiselect(
    "Booking status", status_options, default=status_options
)

# The aggregate table is small. Filtering it is inexpensive and bounded.
filtered = summary
if set(selected_hotels) != set(hotel_options):
    filtered = filtered.loc[filtered["hotel"].astype(str).isin(selected_hotels)]
if set(selected_months) != set(available_months):
    filtered = filtered.loc[filtered["arrival_date_month"].astype(str).isin(selected_months)]
if set(selected_segments) != set(segment_options):
    filtered = filtered.loc[filtered["market_segment"].astype(str).isin(selected_segments)]
if set(selected_statuses) != set(status_options):
    filtered = filtered.loc[filtered["cancellation_label"].astype(str).isin(selected_statuses)]

if filtered.empty:
    st.warning("No bookings match the selected filters.")
    st.stop()

filtered = filtered.assign(
    canceled_bookings=filtered["bookings"] * filtered["is_canceled"]
)
booking_count = int(filtered["bookings"].sum())
canceled_count = int(filtered["canceled_bookings"].sum())
average_adr = filtered["adr_sum"].sum() / booking_count
average_lead_time = filtered["lead_time_sum"].sum() / booking_count

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Bookings", "{:,.0f}".format(booking_count))
metric_2.metric("Cancellation rate", "{:.1%}".format(canceled_count / booking_count))
metric_3.metric("Average ADR", "{:,.2f}".format(average_adr))
metric_4.metric("Average lead time", "{:,.1f} days".format(average_lead_time))

left, right = st.columns(2)

with left:
    status_summary = (
        filtered.groupby("cancellation_label", observed=True, as_index=False)["bookings"]
        .sum()
    )
    figure = px.bar(
        status_summary,
        x="cancellation_label",
        y="bookings",
        color="cancellation_label",
        color_discrete_map={"Not canceled": "#59A14F", "Canceled": "#E15759"},
        title="Booking status",
        labels={"cancellation_label": "", "bookings": "Bookings"},
    )
    figure.update_layout(showlegend=False)
    st.plotly_chart(figure, use_container_width=True)

with right:
    hotel_summary = (
        filtered.groupby("hotel", observed=True, as_index=False)[
            ["bookings", "canceled_bookings"]
        ]
        .sum()
    )
    hotel_summary["cancellation_rate"] = (
        hotel_summary["canceled_bookings"] / hotel_summary["bookings"]
    )
    figure = px.bar(
        hotel_summary,
        x="hotel",
        y="cancellation_rate",
        color="hotel",
        title="Cancellation rate by hotel",
        labels={"hotel": "", "cancellation_rate": "Cancellation rate"},
    )
    figure.update_yaxes(tickformat=".0%")
    figure.update_layout(showlegend=False)
    st.plotly_chart(figure, use_container_width=True)

monthly_summary = (
    filtered.groupby(["arrival_date_month", "hotel"], observed=True, as_index=False)[
        ["bookings", "canceled_bookings"]
    ]
    .sum()
)
monthly_summary["cancellation_rate"] = (
    monthly_summary["canceled_bookings"] / monthly_summary["bookings"]
)
monthly_summary["arrival_date_month"] = pd.Categorical(
    monthly_summary["arrival_date_month"], categories=month_order, ordered=True
)
monthly_summary = monthly_summary.sort_values("arrival_date_month")

figure = px.line(
    monthly_summary,
    x="arrival_date_month",
    y="cancellation_rate",
    color="hotel",
    markers=True,
    title="Monthly cancellation rate",
    labels={
        "arrival_date_month": "Arrival month",
        "cancellation_rate": "Cancellation rate",
        "hotel": "Hotel",
    },
    category_orders={"arrival_date_month": month_order},
)
figure.update_yaxes(tickformat=".0%")
st.plotly_chart(figure, use_container_width=True)

left, right = st.columns(2)

with left:
    lead_summary = (
        filtered.groupby("lead_time_band", observed=True, as_index=False)[
            ["bookings", "canceled_bookings"]
        ]
        .sum()
    )
    lead_summary["cancellation_rate"] = (
        lead_summary["canceled_bookings"] / lead_summary["bookings"]
    )
    figure = px.bar(
        lead_summary,
        x="lead_time_band",
        y="cancellation_rate",
        title="Cancellation rate by lead time",
        labels={"lead_time_band": "Lead time", "cancellation_rate": "Cancellation rate"},
    )
    figure.update_yaxes(tickformat=".0%")
    st.plotly_chart(figure, use_container_width=True)

with right:
    segment_summary = (
        filtered.groupby("market_segment", observed=True, as_index=False)[
            ["bookings", "canceled_bookings"]
        ]
        .sum()
    )
    segment_summary["cancellation_rate"] = (
        segment_summary["canceled_bookings"] / segment_summary["bookings"]
    )
    segment_summary = segment_summary.sort_values("cancellation_rate")
    figure = px.bar(
        segment_summary,
        x="cancellation_rate",
        y="market_segment",
        orientation="h",
        title="Cancellation rate by market segment",
        labels={"market_segment": "", "cancellation_rate": "Cancellation rate"},
    )
    figure.update_xaxes(tickformat=".0%")
    st.plotly_chart(figure, use_container_width=True)

adr_summary = (
    filtered.groupby(["hotel", "cancellation_label"], observed=True, as_index=False)[
        ["bookings", "adr_sum"]
    ]
    .sum()
)
adr_summary["average_adr"] = adr_summary["adr_sum"] / adr_summary["bookings"]
figure = px.bar(
    adr_summary,
    x="hotel",
    y="average_adr",
    color="cancellation_label",
    barmode="group",
    title="Average daily rate by hotel and booking status",
    labels={"hotel": "", "average_adr": "Average ADR", "cancellation_label": "Status"},
)
st.plotly_chart(figure, use_container_width=True)

st.subheader("Booking sample")
preview_filtered = preview
if set(selected_hotels) != set(hotel_options):
    preview_filtered = preview_filtered.loc[
        preview_filtered["hotel"].astype(str).isin(selected_hotels)
    ]
if set(selected_months) != set(available_months):
    preview_filtered = preview_filtered.loc[
        preview_filtered["arrival_date_month"].astype(str).isin(selected_months)
    ]
if set(selected_segments) != set(segment_options):
    preview_filtered = preview_filtered.loc[
        preview_filtered["market_segment"].astype(str).isin(selected_segments)
    ]
if set(selected_statuses) != set(status_options):
    preview_filtered = preview_filtered.loc[
        preview_filtered["cancellation_label"].isin(selected_statuses)
    ]

preview_columns = [
    "hotel",
    "is_canceled",
    "lead_time",
    "arrival_date_month",
    "market_segment",
    "adr",
    "total_guests",
]
st.dataframe(preview_filtered.loc[:, preview_columns], use_container_width=True)
st.caption("The table is a bounded sample of up to 5,000 source rows.")

# Run from a terminal in the project folder:
# streamlit run hotel_bookings_app.py
