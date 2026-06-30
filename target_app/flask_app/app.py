from flask import Flask
import random

app = Flask(__name__)
# A global list to simulate an in-memory cache or database leak
memory_cache = []

@app.route('/matchmake')
def matchmake():
    # Simulate pulling thousands of user profiles and calculating scores
    payload = [random.random() for _ in range(50000)]
    
    # Intentionally caching the result to cause the memory leak over time
    memory_cache.append(payload) 
    
    return "Calculated", 200

@app.route('/matchmake-healthy')
def matchmake_healthy():
    # Simulate pulling thousands of user profiles and calculating scores
    payload = [random.random() for _ in range(50000)]
    return "Calculated Healthy", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

