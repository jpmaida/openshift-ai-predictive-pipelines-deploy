
import json
import os

import requests
import streamlit as st
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

MODEL_ENDPOINT = os.getenv("MODEL_ENDPOINT", "").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "high-value-customer")
MODEL_TOKEN = os.getenv("MODEL_TOKEN", "")
VERIFY_SSL = os.getenv("VERIFY_SSL", "true").lower() == "true"


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1250px;
        }

        .hero {
            padding: 2rem;
            border-radius: 16px;
            background: linear-gradient(120deg, #14213d, #253b67);
            color: white;
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            color: white;
            margin-bottom: 0.3rem;
        }

        .hero p {
            color: #d8e2f2;
            font-size: 1.05rem;
            margin-bottom: 0;
        }

        .result-card {
            padding: 1.5rem;
            border-radius: 14px;
            text-align: center;
            margin: 1rem 0;
        }

        .high-value {
            background: rgba(22, 163, 74, 0.12);
            border: 1px solid rgba(22, 163, 74, 0.5);
        }

        .not-high-value {
            background: rgba(100, 116, 139, 0.12);
            border: 1px solid rgba(100, 116, 139, 0.5);
        }

        .result-label {
            font-size: 1.6rem;
            font-weight: 800;
            margin: 0.5rem 0;
        }

        .result-number {
            font-size: 3.5rem;
            font-weight: 900;
            line-height: 1.2;
        }

        div[data-testid="stForm"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 12px;
            padding: 1.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🎯 Customer Intelligence</h1>
        <p>
            AI-powered High Value Customer Prediction
            <br>
            Understand your customers and identify those with
            the greatest potential value to your business.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("About")

    st.markdown(
        """
        This application uses a Machine Learning model
        deployed on **Red Hat OpenShift AI**.

        The model analyzes customer purchasing behavior
        and predicts whether a customer belongs to the
        High Value Customer category.
        """
    )

    st.divider()

    st.subheader("Model information")

    st.caption(f"Model: `{MODEL_NAME}`")
    st.caption("Framework: Scikit-learn")
    st.caption("Serving: MLServer / KServe")
    st.caption("Protocol: V2 Inference")

    st.divider()

    if MODEL_ENDPOINT:
        st.success("Model endpoint configured")
    else:
        st.error("MODEL_ENDPOINT is not configured")

    st.caption(
        "The prediction is performed remotely by "
        "the OpenShift AI model server."
    )


# ============================================================
# API CLIENT
# ============================================================

def build_inference_payload(features):
    """Build a KServe V2 request using MLServer DataFrame input."""

    inputs = []

    for name, value in features.items():

        if name == "Country":
            inputs.append({
                "name": name,
                "datatype": "BYTES",
                "shape": [1],
                "parameters": {
                    "content_type": "str"
                },
                "data": [value],
            })
        else:
            inputs.append({
                "name": name,
                "datatype": "FP64",
                "shape": [1],
                "data": [float(value)],
            })

    return {
        "parameters": {
            "content_type": "pd"
        },
        "inputs": inputs,
    }


def predict(features):
    """Send the customer data to the deployed model."""

    if not MODEL_ENDPOINT:
        raise ValueError(
            "MODEL_ENDPOINT environment variable is missing."
        )

    url = (
        f"{MODEL_ENDPOINT}/v2/models/"
        f"{MODEL_NAME}/infer"
    )

    headers = {
        "Content-Type": "application/json",
    }

    if MODEL_TOKEN:
        headers["Authorization"] = f"Bearer {MODEL_TOKEN}"

    payload = build_inference_payload(features)

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60,
        verify=VERIFY_SSL,
    )

    response.raise_for_status()

    result = response.json()

    outputs = result.get("outputs", [])

    prediction_output = next(
        (item for item in outputs if item.get("name") == "predict"),
        None,
    )

    if not prediction_output or not prediction_output.get("data"):
        raise ValueError(
            f"Unexpected model response: {result}"
        )

    prediction = int(prediction_output["data"][0])

    return prediction, result, payload


# ============================================================
# CUSTOMER FORM
# ============================================================

st.subheader("Customer analysis")

st.write(
    "Enter the customer's purchasing information below. "
    "The model will analyze the information and classify the customer."
)

with st.form("customer_form"):

    st.markdown("### 👤 Customer identification")

    customer_reference = st.text_input(
        "Customer reference",
        value="CUST-001",
        help="An identifier for display purposes. It is not sent to the model.",
    )

    st.divider()

    st.markdown("### 🛒 Purchasing behavior")

    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input(
            "Recency (days)",
            min_value=0.0,
            value=15.0,
            step=1.0,
            help="Number of days since the customer's last purchase.",
        )

    with col2:
        frequency = st.number_input(
            "Frequency (orders)",
            min_value=1.0,
            value=12.0,
            step=1.0,
            help="Total number of orders in the historical period.",
        )

    with col3:
        monetary = st.number_input(
            "Monetary (£)",
            min_value=0.0,
            value=2500.0,
            step=50.0,
            help="Total historical revenue generated by the customer.",
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        avg_order_value = st.number_input(
            "Average order value (£)",
            min_value=0.0,
            value=208.33,
            step=10.0,
        )

    with col2:
        total_quantity = st.number_input(
            "Total quantity",
            min_value=0.0,
            value=350.0,
            step=10.0,
            help="Total quantity of items purchased.",
        )

    with col3:
        unique_products = st.number_input(
            "Unique products",
            min_value=0.0,
            value=45.0,
            step=1.0,
            help="Number of different products purchased.",
        )

    st.divider()

    st.markdown("### 📊 Customer profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        avg_days_between_orders = st.number_input(
            "Average days between orders",
            min_value=0.0,
            value=12.5,
            step=1.0,
        )

    with col2:
        customer_tenure = st.number_input(
            "Customer tenure (days)",
            min_value=0.0,
            value=180.0,
            step=10.0,
            help="Number of days since the first purchase.",
        )

    with col3:
        avg_items_per_order = st.number_input(
            "Average items per order",
            min_value=0.0,
            value=29.16,
            step=1.0,
        )

    col1, col2 = st.columns(2)

    with col1:
        spend_last_30_days = st.number_input(
            "Spend in last 30 days (£)",
            min_value=0.0,
            value=800.0,
            step=50.0,
        )

    with col2:
        spend_previous_60_days = st.number_input(
            "Spend in previous 60 days (£)",
            min_value=0.0,
            value=1700.0,
            step=50.0,
        )

    st.divider()

    st.markdown("### 🌍 Location")

    country = st.text_input(
        "Country",
        value="United Kingdom",
        help="Country associated with the customer's transactions.",
    )

    st.divider()

    submitted = st.form_submit_button(
        "🚀 Analyze customer",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# INFERENCE
# ============================================================

if submitted:

    features = {
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary,
        "AvgOrderValue": avg_order_value,
        "TotalQuantity": total_quantity,
        "UniqueProducts": unique_products,
        "AvgDaysBetweenOrders": avg_days_between_orders,
        "CustomerTenure": customer_tenure,
        "AvgItemsPerOrder": avg_items_per_order,
        "SpendLast30Days": spend_last_30_days,
        "SpendPrevious60Days": spend_previous_60_days,
        "Country": country.strip(),
    }

    if not country.strip():
        st.error("Please provide a country.")
        st.stop()

    with st.spinner(
        "Analyzing customer with the OpenShift AI model..."
    ):

        try:
            prediction, result, payload = predict(features)

            st.divider()
            st.subheader("Analysis result")

            st.caption(
                f"Customer reference: {customer_reference or 'Not provided'}"
            )

            if prediction == 1:

                st.markdown(
                    """
                    <div class="result-card high-value">
                        <div class="result-number">1</div>
                        <div class="result-label">
                            🏆 HIGH VALUE CUSTOMER
                        </div>
                        <p>
                            The model classified this customer
                            as a high value customer.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.success(
                    "This customer belongs to the High Value Customer class."
                )

            elif prediction == 0:

                st.markdown(
                    """
                    <div class="result-card not-high-value">
                        <div class="result-number">0</div>
                        <div class="result-label">
                            STANDARD VALUE CUSTOMER
                        </div>
                        <p>
                            The model did not classify this customer
                            as a high value customer.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.info(
                    "This customer does not belong to the High Value Customer class."
                )

            else:
                st.warning(
                    f"The model returned an unexpected class: {prediction}"
                )

            st.markdown("### 📋 Customer profile summary")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total revenue",
                f"£ {monetary:,.2f}",
            )

            col2.metric(
                "Orders",
                f"{frequency:,.0f}",
            )

            col3.metric(
                "Days since last purchase",
                f"{recency:,.0f}",
            )

            col4.metric(
                "Unique products",
                f"{unique_products:,.0f}",
            )

            st.divider()

            st.markdown("### 🔍 Technical details")

            st.write(
                "The following information was exchanged with "
                "the OpenShift AI inference endpoint."
            )

            with st.expander("View request JSON"):
                st.json(payload)

            with st.expander("View raw model response"):
                st.json(result)

            st.caption(
                f"Model: {MODEL_NAME} | "
                f"Request ID: {result.get('id', 'N/A')}"
            )

            st.caption(
                "The prediction is a classification result, "
                "not a probability or guarantee of future revenue."
            )

        except requests.exceptions.HTTPError as exc:
            st.error(
                f"Model endpoint returned an HTTP error: {exc}"
            )

            if exc.response is not None:
                st.code(exc.response.text)

        except requests.exceptions.RequestException as exc:
            st.error(
                f"Could not communicate with the model endpoint: {exc}"
            )

        except Exception as exc:
            st.error(f"Prediction error: {exc}")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Intelligence | Powered by Scikit-learn "
    "and Red Hat OpenShift AI"
)
