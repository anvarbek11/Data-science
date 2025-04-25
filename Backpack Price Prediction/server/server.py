from flask import Flask, request, jsonify
import util

app = Flask(__name__)


@app.route('/api/get_category_options', methods=['GET'])
def get_category_options():
    try:
        category = request.args.get('category')
        if not category:
            return jsonify({'error': 'Category parameter is required'}), 400

        options = util.get_category_options(category)
        response = jsonify({
            'options': options,
            'status': 'success'
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/predict_price', methods=['POST'])
def predict_price():
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        required_fields = [
            'brand', 'material', 'size', 'compartments',
            'laptop_compartment', 'waterproof', 'style',
            'color', 'weight_capacity'
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        input_data = {
            'Brand': data['brand'],
            'Material': data['material'],
            'Size': data['size'],
            'Compartments': int(data['compartments']),
            'Laptop Compartment': data['laptop_compartment'],
            'Waterproof': data['waterproof'],
            'Style': data['style'],
            'Color': data['color'],
            'Weight Capacity (kg)': float(data['weight_capacity'])
        }

        prediction = util.predict_price(input_data)

        return jsonify({
            'estimated_price': prediction,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


if __name__ == "__main__":
    print("Starting Python Flask Server for Backpack Price Prediction...")
    util.load_saved_artifacts()
    app.run(host='0.0.0.0', port=5000, debug=True)