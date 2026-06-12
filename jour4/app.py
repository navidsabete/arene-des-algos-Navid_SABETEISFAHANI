import streamlit as st
import joblib

artefacts = joblib.load("breast_cancer_model.joblib")

modele = artefacts["modele"]
scaler = artefacts["scaler"]


st.title("Détection Cancer du Sein")

st.write(
    """
    Saisissez les 30 caractéristiques de la tumeur.
    Les valeurs affichées correspondent aux moyennes du jeu d'entraînement.
    """
)

with st.form("prediction_form"):

    with st.expander("Mesures moyennes", expanded=True):

        mean_radius = st.number_input("mean radius", value=14.0)
        mean_texture = st.number_input("mean texture", value=19.0)
        mean_perimeter = st.number_input("mean perimeter", value=90.0)
        mean_area = st.number_input("mean area", value=600.0)
        mean_smoothness = st.number_input("mean smoothness", value=0.10)
        mean_compactness = st.number_input("mean compactness", value=0.10)
        mean_concavity = st.number_input("mean concavity", value=0.10)
        mean_concave_points = st.number_input("mean concave points", value=0.05)
        mean_symmetry = st.number_input("mean symmetry", value=0.18)
        mean_fractal_dimension = st.number_input("mean fractal dimension", value=0.06)

    with st.expander("Erreurs de mesure"):

        radius_error = st.number_input("radius error", value=0.40)
        texture_error = st.number_input("texture error", value=1.20)
        perimeter_error = st.number_input("perimeter error", value=2.80)
        area_error = st.number_input("area error", value=40.0)
        smoothness_error = st.number_input("smoothness error", value=0.005)
        compactness_error = st.number_input("compactness error", value=0.020)
        concavity_error = st.number_input("concavity error", value=0.025)
        concave_points_error = st.number_input("concave points error", value=0.010)
        symmetry_error = st.number_input("symmetry error", value=0.020)
        fractal_dimension_error = st.number_input("fractal dimension error", value=0.003)

    with st.expander("Pires mesures observées"):

        worst_radius = st.number_input("worst radius", value=16.0)
        worst_texture = st.number_input("worst texture", value=25.0)
        worst_perimeter = st.number_input("worst perimeter", value=105.0)
        worst_area = st.number_input("worst area", value=800.0)
        worst_smoothness = st.number_input("worst smoothness", value=0.14)
        worst_compactness = st.number_input("worst compactness", value=0.25)
        worst_concavity = st.number_input("worst concavity", value=0.25)
        worst_concave_points = st.number_input("worst concave points", value=0.12)
        worst_symmetry = st.number_input("worst symmetry", value=0.30)
        worst_fractal_dimension = st.number_input("worst fractal dimension", value=0.08)

    submitted = st.form_submit_button("Prédire")

if submitted:

    features = [[
        mean_radius,
        mean_texture,
        mean_perimeter,
        mean_area,
        mean_smoothness,
        mean_compactness,
        mean_concavity,
        mean_concave_points,
        mean_symmetry,
        mean_fractal_dimension,

        radius_error,
        texture_error,
        perimeter_error,
        area_error,
        smoothness_error,
        compactness_error,
        concavity_error,
        concave_points_error,
        symmetry_error,
        fractal_dimension_error,

        worst_radius,
        worst_texture,
        worst_perimeter,
        worst_area,
        worst_smoothness,
        worst_compactness,
        worst_concavity,
        worst_concave_points,
        worst_symmetry,
        worst_fractal_dimension
    ]]

    X_scaled = scaler.transform(features)

    prediction = modele.predict(X_scaled)[0]
    proba = modele.predict_proba(X_scaled)[0]

    confiance = max(proba)

    label = "Bénigne" if prediction == 1 else "Maligne"

    st.success(f"Prédiction : {label}")
    st.metric("Confiance", f"{confiance:.1%}")
    st.progress(float(confiance))