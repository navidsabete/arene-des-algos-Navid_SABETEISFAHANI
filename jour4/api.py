import joblib
import numpy as np

from flask import Flask, request, jsonify

app = Flask(__name__)

artefacts = joblib.load("breast_cancer_model.joblib")

modele = artefacts["modele"]
scaler = artefacts["scaler"]

N_FEATURES = modele.n_features_in_

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    # Vérification JSON
    if data is None:
        return jsonify(
            {"error": "JSON invalide"}
        ), 400

    # Vérification clé 'features'
    if "features" not in data:
        return jsonify(
            {"error": "La clé 'features' est obligatoire"}
        ), 400

    features = data["features"]

    # Vérification type
    if not isinstance(features, list):
        return jsonify(
            {"error": "'features' doit être une liste"}
        ), 400

    # Tableau vide
    if len(features) == 0:
        return jsonify(
            {"error": "Liste de features vide"}
        ), 400

    # Nombre de variables
    if len(features) != N_FEATURES:
        return jsonify(
            {
                "error": f"Nombre de features incorrect. Attendu : {N_FEATURES}"
            }
        ), 400

    try:

        features = [float(x) for x in features]

        X = np.array(features).reshape(1, -1)

        X_scaled = scaler.transform(X)

        prediction = int(modele.predict(X_scaled)[0])

        if hasattr(modele, "predict_proba"):
            proba = float(
                np.max(modele.predict_proba(X_scaled))
            )
        else:
            proba = None

        return jsonify(
            {
                "prediction": prediction,
                "proba": proba,
                "label": "malin" if prediction == 0 else "benin"
            }
        )

    except Exception as e:
        return jsonify(
            {"error": str(e)}
        ), 400


if __name__ == "__main__":
    app.run(debug=True)