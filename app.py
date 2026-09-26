import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Flight Price Prediction",
    page_icon="✈️",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("Models/flight_price_model.pkl")


model = load_model()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv("Clean_Dataset.csv")

    # Same preprocessing as training
    data = data.drop("Unnamed: 0", axis=1)
    data = data.drop("flight", axis=1)

    data["class"] = data["class"].apply(
        lambda x: 1 if x == "Business" else 0
    )

    data["stops"] = pd.factorize(data["stops"])[0]

    data = data.join(
        pd.get_dummies(data["airline"], prefix="airline")
    ).drop("airline", axis=1)

    data = data.join(
        pd.get_dummies(data["source_city"], prefix="source")
    ).drop("source_city", axis=1)

    data = data.join(
        pd.get_dummies(data["destination_city"], prefix="dest")
    ).drop("destination_city", axis=1)

    data = data.join(
        pd.get_dummies(data["arrival_time"], prefix="arrival")
    ).drop("arrival_time", axis=1)

    data = data.join(
        pd.get_dummies(data["departure_time"], prefix="departure")
    ).drop("departure_time", axis=1)

    return data


data = load_data()


# ============================================================
# GET TRAINING FEATURES
# ============================================================

feature_columns = data.drop("price", axis=1).columns.tolist()


# ============================================================
# TITLE
# ============================================================

st.title("✈️ Flight Price Prediction")

st.markdown(
    """
    ### Machine Learning Flight Price Predictor

    Enter the flight information below to estimate the ticket price
    using a trained Random Forest regression model.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("✈️ Flight Information")


# Airline
airline = st.sidebar.selectbox(
    "Airline",
    sorted(
        pd.read_csv("Clean_Dataset.csv")["airline"].unique()
    )
)


# Source city
source_city = st.sidebar.selectbox(
    "Source City",
    sorted(
        pd.read_csv("Clean_Dataset.csv")["source_city"].unique()
    )
)


# Destination city
destination_city = st.sidebar.selectbox(
    "Destination City",
    sorted(
        pd.read_csv("Clean_Dataset.csv")["destination_city"].unique()
    )
)


# Departure time
departure_time = st.sidebar.selectbox(
    "Departure Time",
    sorted(
        pd.read_csv("Clean_Dataset.csv")["departure_time"].unique()
    )
)


# Arrival time
arrival_time = st.sidebar.selectbox(
    "Arrival Time",
    sorted(
        pd.read_csv("Clean_Dataset.csv")["arrival_time"].unique()
    )
)


# Stops
stops = st.sidebar.selectbox(
    "Number of Stops",
    [
        "zero",
        "one",
        "two_or_more"
    ]
)


# Class
flight_class = st.sidebar.selectbox(
    "Class",
    [
        "Economy",
        "Business"
    ]
)


# Duration
duration = st.sidebar.number_input(
    "Flight Duration (hours)",
    min_value=0.0,
    max_value=50.0,
    value=2.0,
    step=0.1
)


# Days left
days_left = st.sidebar.number_input(
    "Days Before Departure",
    min_value=1,
    max_value=50,
    value=15,
    step=1
)


# ============================================================
# PREDICTION
# ============================================================

st.divider()

if st.button("💰 Predict Flight Price", type="primary"):

    # --------------------------------------------------------
    # Create raw input
    # --------------------------------------------------------

    input_data = pd.DataFrame({
        "airline": [airline],
        "source_city": [source_city],
        "destination_city": [destination_city],
        "departure_time": [departure_time],
        "arrival_time": [arrival_time],
        "stops": [stops],
        "class": [flight_class],
        "duration": [duration],
        "days_left": [days_left]
    })


    # --------------------------------------------------------
    # Same preprocessing as training
    # --------------------------------------------------------

    input_data["class"] = input_data["class"].apply(
        lambda x: 1 if x == "Business" else 0
    )


    # IMPORTANT:
    # We need the same mapping as training.
    stops_mapping = {
        "zero": 0,
        "one": 1,
        "two_or_more": 2
    }

    input_data["stops"] = input_data["stops"].map(
        stops_mapping
    )


    input_data = input_data.join(
        pd.get_dummies(
            input_data["airline"],
            prefix="airline"
        )
    ).drop("airline", axis=1)


    input_data = input_data.join(
        pd.get_dummies(
            input_data["source_city"],
            prefix="source"
        )
    ).drop("source_city", axis=1)


    input_data = input_data.join(
        pd.get_dummies(
            input_data["destination_city"],
            prefix="dest"
        )
    ).drop("destination_city", axis=1)


    input_data = input_data.join(
        pd.get_dummies(
            input_data["arrival_time"],
            prefix="arrival"
        )
    ).drop("arrival_time", axis=1)


    input_data = input_data.join(
        pd.get_dummies(
            input_data["departure_time"],
            prefix="departure"
        )
    ).drop("departure_time", axis=1)


    # --------------------------------------------------------
    # Make sure columns are EXACTLY the same as training
    # --------------------------------------------------------

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(input_data)[0]


    # --------------------------------------------------------
    # Display prediction
    # --------------------------------------------------------

    st.success("Prediction completed successfully!")

    st.metric(
        label="💰 Estimated Flight Price",
        value=f"₹ {prediction:,.0f}"
    )


    # --------------------------------------------------------
    # Flight summary
    # --------------------------------------------------------

    st.divider()

    st.subheader("✈️ Flight Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Airline**")
        st.write(airline)

    with col2:
        st.write("**Route**")
        st.write(
            f"{source_city} → {destination_city}"
        )

    with col3:
        st.write("**Class**")
        st.write(flight_class)


    col4, col5, col6 = st.columns(3)

    with col4:
        st.write("**Departure**")
        st.write(departure_time)

    with col5:
        st.write("**Arrival**")
        st.write(arrival_time)

    with col6:
        st.write("**Stops**")
        st.write(stops)


    col7, col8 = st.columns(2)

    with col7:
        st.write("**Duration**")
        st.write(f"{duration} hours")

    with col8:
        st.write("**Days Before Departure**")
        st.write(days_left)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Flight Price Prediction | Random Forest Regression | Machine Learning"
)