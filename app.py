import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px

# Load ML model and encoders
regressor = pickle.load(open("regressor.pkl", "rb"))
oe1 = pickle.load(open("encoder1.pkl", "rb"))
oe2 = pickle.load(open("encoder2.pkl", "rb"))

# Load dataset
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv("AnalysisData.csv")
    return df


df = load_data()
df["CompletionYear"] = pd.DatetimeIndex(df["CompletionDate"]).year

# Main prediction function
def predict_production(
    ProppantIntensity_LBSPerFT,
    FluidIntensity_BBLPerFT,
    CompletionYear,
    Operator,
    WellDepth,
    LateralLength_FT,
    Porosity,
    WaterSaturation,
    Reservoir,
    CarbonateVolume,
    Maturity,
    TotalWellCost_USDMM,
):

    Operator = float(oe1.transform(np.array([Operator]).reshape(-1, 1)))
    Reservoir = float(oe2.transform(np.array([Reservoir]).reshape(-1, 1)))

    ProppantVolume = float(LateralLength_FT) * float(ProppantIntensity_LBSPerFT)
    LiquidVolume = float(LateralLength_FT) * float(FluidIntensity_BBLPerFT)

    input = np.array(
        [
            [
                ProppantVolume,
                LiquidVolume,
                CompletionYear,
                Operator,
                WellDepth,
                Porosity,
                WaterSaturation,
                Reservoir,
                CarbonateVolume,
                Maturity,
                TotalWellCost_USDMM,
            ]
        ]
    ).astype(np.float64)
    prediction = regressor.predict(input)

    return int(prediction)


# Geographical map
def production_plot(data):
    fig_map_box = px.scatter_mapbox(
        data,
        lat="SurfaceLatitude",
        lon="SurfaceLongitude",
        labels={
            "SurfaceLatitude": "Latitude",
            "SurfaceLongitude": "Longitude",
            "CompletionDate": "Production Date",
            "TotalWellCost_USDMM": "Well Cost in $MM",
        },
        hover_data=["Operator", "Reservoir", "CompletionDate", "TotalWellCost_USDMM"],
        color="CumOil12Month",
        size_max=100,
        mapbox_style="carto-darkmatter",
    )
    fig_map_box.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return st.plotly_chart(fig_map_box)


# Histogram plot
def histogram_plot(data):
    fig = px.histogram(
        data,
        x="CompletionYear",
        y="CumOil12Month",
        color="Reservoir",
        width=750,
        height=600,
        labels={
            "CumOil12Month": "Cumulative Production",
            "CompletionYear": "Production Year",
        },
        template="plotly_dark",
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return st.plotly_chart(fig)


# Main web app
def main():
    st.title("Annually Cumulative Oil Production")
    st.markdown(
        "This application is a web app used to analyze and estimate the 12-month cumulative production of oil wells in Texas. The data is provided by Quantum Energy Partners."
    )
    st.write("## Oil Production Dashboard")
    unique = list(df["Operator"].unique())
    unique.insert(0, "All operators")
    opt = st.selectbox(label="Operator", options=unique)

    check_box = st.checkbox(label="Display dataset")
    if check_box:
        if opt == "All operators":
            st.write(df)
        else:
            data = df[df["Operator"] == opt]
            st.write(data)

    if opt == "All operators":
        production_plot(df)
    else:
        data = df[df["Operator"] == opt]
        production_plot(data)
        histogram_plot(data)

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Oil Production ML Prediction</h2>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html=True)
    LateralLength_FT = st.sidebar.text_input("Lateral Length", "Type Here")
    ProppantIntensity_LBSPerFT = st.sidebar.text_input(
        "Proppant Intensity", "Type Here"
    )
    FluidIntensity_BBLPerFT = st.sidebar.text_input("Fluid Intensity", "Type Here")
    CompletionYear = st.sidebar.text_input("Production Year", "Type Here")
    Operator = st.sidebar.selectbox("Operator", df["Operator"].unique())
    Reservoir = st.sidebar.selectbox("Reservoir", df["Reservoir"].unique())
    WellDepth = st.sidebar.text_input("Well Depth", "Type Here")
    Porosity = st.sidebar.text_input("Porosity", "Type Here")
    WaterSaturation = st.sidebar.text_input(" Water Saturation", "Type Here")
    CarbonateVolume = st.sidebar.text_input("Carbonate Volume", "Type Here")
    Maturity = st.sidebar.text_input("Maturity", "Type Here")
    TotalWellCost_USDMM = st.sidebar.text_input("Total Well Cost", "Type Here")

    result = ""
    if st.sidebar.button("Predict"):
        result = predict_production(
            ProppantIntensity_LBSPerFT,
            FluidIntensity_BBLPerFT,
            CompletionYear,
            Operator,
            WellDepth,
            LateralLength_FT,
            Porosity,
            WaterSaturation,
            Reservoir,
            CarbonateVolume,
            Maturity,
            TotalWellCost_USDMM,
        )
    st.sidebar.success(
        "The Estimated Annually Cumulative Production is {}".format(result)
    )


if __name__ == "__main__":
    main()
