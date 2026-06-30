from flask import Flask
import random

app = Flask(__name__)
memory_cache = []

@app.route('/matchmake')
def matchmake():
    payload = [random.random() for _ in range(5000)]
    
    memory_cache.append(payload) 
    
    return "Calculated", 200

@app.route('/matchmake-healthy')
def matchmake_healthy():
    payload = [random.random() for _ in range(50000)]
    return "Calculated Healthy", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

