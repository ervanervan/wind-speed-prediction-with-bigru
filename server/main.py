from flask import Flask, request, jsonify
from flask_cors import CORS
from Controller_FF_X_ANB import predict_ff_x_anb, data_ff_x_anb, X_data_ff_x_anb, predict_forcasting_ff_x_anb, predict_forcasting_ff_x_anb_by_input
from Controller_FF_AVG_ANB import predict_ff_avg_anb, data_ff_avg_anb, X_data_ff_avg_anb, predict_forcasting_ff_avg_anb,predict_forcasting_ff_avg_anb_by_input
from Controller_FF_AVG_TPI import predict_ff_avg_tpi, data_ff_avg_tpi, X_data_ff_avg_tpi, predict_forcasting_ff_avg_tpi
from Controller_FF_X_TPI import predict_ff_x_tpi, data_ff_x_tpi, X_data_ff_x_tpi, predict_forcasting_ff_x_tpi
import json
app = Flask(__name__)
CORS(app)


# ff-x-anb
@app.route('/ff-x-anb', methods=['GET'])
def ff_x_anb():
    if request.method=='GET':
        # data = request.get_json()
        predicted = predict_ff_x_anb(X_data_ff_x_anb)
        return jsonify({'predicted' : predicted })
    
@app.route('/ff-x-anb-input-90', methods=['POST'])
def ff_x_anb_by_input():
    if request.method == 'POST':
        data = request.get_json()
        input_kecepatan_sebelumnya = data["input"]
        # Memisahkan string berdasarkan koma
        string_list = input_kecepatan_sebelumnya.split(',')
        input_kecepatan_sebelumnya = [int(i) for i in string_list]
        predicted = predict_forcasting_ff_x_anb_by_input(input_kecepatan_sebelumnya)
        return jsonify({'predicted': predicted})
    
@app.route('/ff-x-anb-original', methods=['GET'])
def ff_x_anb_original():
    if request.method=='GET':
        
        return (data_ff_x_anb)

@app.route('/ff-x-anb-forcasting', methods=['GET'])
def ff_x_anb_forcasting():
    if request.method=='GET':
        
        predicted = predict_forcasting_ff_x_anb()
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


# ff-avg-anb
@app.route('/ff-avg-anb', methods=['GET'])
def ff_avg_anb():
    if request.method=='GET':
        # data = request.get_json()
        predicted = predict_ff_avg_anb(X_data_ff_avg_anb)
        return jsonify({'predicted' : predicted })
    
@app.route('/ff-avg-anb-input-90', methods=['POST'])
def ff_avg_anb_by_input():
    if request.method == 'POST':
        data = request.get_json()
        input_kecepatan_sebelumnya = data["input"]
        # Memisahkan string berdasarkan koma
        string_list = input_kecepatan_sebelumnya.split(',')
        input_kecepatan_sebelumnya = [int(i) for i in string_list]
        predicted = predict_forcasting_ff_avg_anb_by_input(input_kecepatan_sebelumnya)
        return jsonify({'predicted': predicted})

@app.route('/ff-avg-anb-original', methods=['GET'])
def ff_avg_anb_original():
    if request.method=='GET':
        
        return (data_ff_avg_anb)

@app.route('/ff-avg-anb-forcasting', methods=['GET'])
def ff_avg_anb_forcasting():
    if request.method=='GET':
        
        predicted = predict_forcasting_ff_avg_anb()
        # print (predicted)
        return jsonify({'predicted' : predicted })
    
@app.route('/ff-avg-anb-performance', methods=['GET'])
def ff_avg_anb_performance():
    if request.method == 'GET':
        # Path ke file JSON
        file_path = 'Bidirectional_GRU_FF_AVG_ANAMBAS.json'
        
        # Membaca isi file JSON
        with open(file_path, 'r') as file:
            performance_data = json.load(file)
        
        # Mengirimkan file JSON sebagai respons
        return jsonify(performance_data)
    

# ff-avg-tpi
@app.route('/ff-avg-tpi', methods=['GET'])
def ff_avg_tpi():
    if request.method=='GET':
        # data = request.get_json()
        predicted = predict_ff_avg_tpi(X_data_ff_avg_tpi)
        return jsonify({'predicted' : predicted })
    
@app.route('/ff-avg-tpi-original', methods=['GET'])
def ff_avg_tpi_original():
    if request.method=='GET':
        
        return (data_ff_avg_tpi)

@app.route('/ff-avg-tpi-forcasting', methods=['GET'])
def ff_avg_tpi_forcasting():
    if request.method=='GET':
        
        predicted = predict_forcasting_ff_avg_tpi()
        # print (predicted)
        return jsonify({'predicted' : predicted })
    
@app.route('/ff-avg-tpi-performance', methods=['GET'])
def ff_avg_tpi_performance():
    if request.method == 'GET':
        # Path ke file JSON
        file_path = 'Bidirectional_GRU_FF_AVG_TANJUNGPINANG.json'
        
        # Membaca isi file JSON
        with open(file_path, 'r') as file:
            performance_data = json.load(file)
        
        # Mengirimkan file JSON sebagai respons
        return jsonify(performance_data)


# ff-x-tpi
@app.route('/ff-x-tpi', methods=['GET'])
def ff_x_tpi():
    if request.method=='GET':
        # data = request.get_json()
        predicted = predict_ff_x_tpi(X_data_ff_x_tpi)
        return jsonify({'predicted' : predicted })
    
@app.route('/ff-x-tpi-original', methods=['GET'])
def ff_x_tpi_original():
    if request.method=='GET':
        
        return (data_ff_x_tpi)

@app.route('/ff-x-tpi-forcasting', methods=['GET'])
def ff_x_tpi_forcasting():
    if request.method=='GET':
        
        predicted = predict_forcasting_ff_x_tpi()
        # print (predicted)
        return jsonify({'predicted' : predicted })
    
@app.route('/ff-x-tpi-performance', methods=['GET'])
def ff_x_tpi_performance():
    if request.method == 'GET':
        # Path ke file JSON
        file_path = 'Bidirectional_GRU_FF_X_TANJUNGPINANG.json'
        
        # Membaca isi file JSON
        with open(file_path, 'r') as file:
            performance_data = json.load(file)
        
        # Mengirimkan file JSON sebagai respons
        return jsonify(performance_data)


if __name__ == '__main__':
    app.run(debug=True)