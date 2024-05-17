from flask import Flask, request, jsonify
from flask_cors import CORS
from Controller_FF_X_ANB import predict_ff_x_anb, data_ff_x_anb, X_data_ff_x_anb, predict_forcasting
import json
app = Flask(__name__)
CORS(app)

@app.route('/ff-x-anb', methods=['GET'])
def ff_x_anb():
    if request.method=='GET':
        # data = request.get_json()
        predicted = predict_ff_x_anb(X_data_ff_x_anb)
        return jsonify({'predicted' : predicted })
    
@app.route('/ff-x-anb-original', methods=['GET'])
def ff_x_anb_original():
    if request.method=='GET':
        
        return (data_ff_x_anb)

@app.route('/ff-x-anb-forcasting', methods=['GET'])
def ff_x_anb_forcasting():
    if request.method=='GET':
        
        predicted = predict_forcasting()
        # print (predicted)
        return jsonify({'predicted' : predicted })
@app.route('/ff-x-anb-performance', methods=['GET'])
def ff_x_anb_performance():
    if request.method == 'GET':
        # Path ke file JSON
        file_path = 'Bidirectional_GRU_FF_X_ANAMBAS.json'
        
        # Membaca isi file JSON
        with open(file_path, 'r') as file:
            performance_data = json.load(file)
        
        # Mengirimkan file JSON sebagai respons
        return jsonify(performance_data)


if __name__ == '__main__':
    app.run(debug=True)