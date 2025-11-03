from flask import Flask,request,render_template
import numpy as np
import pandas  # noqa: F401 (kept for compatibility if used elsewhere)
import sklearn  # noqa: F401 (ensures sklearn is available for the unpickled model)
import pickle

# Load only the RandomForest model; no scalers are used.
def _safe_load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

model = _safe_load_model('model.pkl')

# Compatibility patch: models trained with older scikit-learn may miss new attributes
# e.g., 'monotonic_cst' referenced in newer sklearn versions.
def _patch_sklearn_model(m):
    try:
        # Set attribute on the top-level estimator if missing
        if not hasattr(m, "monotonic_cst"):
            setattr(m, "monotonic_cst", None)
        # Set attribute on each estimator in the ensemble if missing
        if hasattr(m, "estimators_"):
            for est in m.estimators_:
                if not hasattr(est, "monotonic_cst"):
                    setattr(est, "monotonic_cst", None)
    except Exception:
        # Best-effort patching; ignore if structure is different
        pass
    return m

model = _patch_sklearn_model(model)

# creating flask app
# Configure Flask to look for templates and static files in the project root,
# since `index.html` and `img.jpg` are located here (not in templates/static folders).
app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    # Parse and validate inputs as floats
    try:
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])
    except (KeyError, ValueError) as e:
        return render_template('index.html', result=f"Invalid input: {e}")

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list, dtype=float).reshape(1, -1)

    # Direct prediction with RandomForest (no scaling)
    try:
        prediction = model.predict(single_pred)
    except Exception as e:
        return render_template('index.html', result=f"Model prediction error: {e}")

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    pred_val = prediction[0]
    # Convert numpy scalar types to native Python types
    if hasattr(pred_val, 'item'):
        pred_val = pred_val.item()

    if pred_val in crop_dict:
        crop = crop_dict[pred_val]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('index.html',result = result)




# python main
if __name__ == "__main__":
    app.run(debug=True)