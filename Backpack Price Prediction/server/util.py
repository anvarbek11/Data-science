import pickle
import json
import pandas as pd
from pathlib import Path

# Global variables to hold artifacts
__model = None
__encoder = None
__features = None
__size_map = {'Small': 0, 'Medium': 1, 'Large': 2}
__category_columns = ['Brand', 'Material', 'Style', 'Color']


def load_saved_artifacts():
    print("Loading saved artifacts...")
    global __model, __encoder, __features

    try:
        artifacts_path = Path("model_artifacts")

        # Load model
        with open(artifacts_path / "lgbm_model.pkl", 'rb') as f:
            __model = pickle.load(f)

        # Load encoder
        with open(artifacts_path / "ordinal_encoder.pkl", 'rb') as f:
            __encoder = pickle.load(f)

        # Load features
        with open(artifacts_path / "features.json", 'r') as f:
            __features = json.load(f)

        print("Artifacts loaded successfully")
    except Exception as e:
        print(f"Error loading artifacts: {str(e)}")
        raise


def get_category_options(category):
    if not __encoder or not __category_columns:
        raise Exception("Artifacts not loaded")

    if category not in __category_columns:
        return []

    try:
        cat_index = __category_columns.index(category)
        return [str(x) for x in __encoder.categories_[cat_index] if str(x) != 'nan']
    except Exception as e:
        print(f"Error getting category options: {str(e)}")
        return []


def predict_price(input_data):
    """Process input and return prediction using the same logic as training"""
    try:
        if not __model or not __encoder or not __features:
            raise Exception("Model artifacts not loaded")

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Feature engineering (matches training pipeline)
        df['Size'] = df['Size'].map(__size_map).fillna(-1)
        df['Weight_per_compartment'] = df['Weight Capacity (kg)'] / (df['Compartments'] + 1)
        df['Laptop Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0}).fillna(0)
        df['Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0}).fillna(0)

        # Encode categoricals
        categorical_cols = [col for col in __category_columns if col in df.columns]
        df[categorical_cols] = __encoder.transform(df[categorical_cols])

        # Ensure correct features and dtypes
        df = df[__features].astype('float32')
        for f in __features:
            if f not in df.columns:
                df[f] = 0  # Default for missing features

        return float(__model.predict(df)[0])

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise


if __name__ == '__main__':
    load_saved_artifacts()
    print("Brand options:", get_category_options('Brand'))
    print("Material options:", get_category_options('Material'))
    print("Style options:", get_category_options('Style'))
    print("Color options:", get_category_options('Color'))

    test_input = {
        'Brand': 'Nike',
        'Material': 'Polyester',
        'Size': 'Medium',
        'Compartments': 3,
        'Laptop Compartment': 'Yes',
        'Waterproof': 'No',
        'Style': 'Backpack',
        'Color': 'Black',
        'Weight Capacity (kg)': 12.5
    }
    print("Test prediction:", predict_price(test_input))