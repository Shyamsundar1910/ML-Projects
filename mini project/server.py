import json
from http.server import SimpleHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt
import io

# Load and train the model
data = pd.read_csv(r'C:\Users\shyam\Desktop\New folder\New Text Document.csv')
last_10_days = data.tail(10)  # Keep "Date" column for frontend
numeric_data = data.drop(columns=["Date"])

x = numeric_data[["Open", "High", "Low"]].to_numpy()
y = numeric_data["Close"].to_numpy()

model = DecisionTreeRegressor()
model.fit(x, y)

class PredictHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            input_data = json.loads(post_data)
            features = np.array([[float(input_data["Open"]), 
                                  float(input_data["High"]), 
                                  float(input_data["Low"])]])
            prediction = model.predict(features)[0]
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"Predicted Rate": prediction}).encode())
        
        except (KeyError, ValueError) as e:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Invalid input: {str(e)}"}).encode())

    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/graph':
            # (Optional) Serve image if the graph is needed in another way
            pass
        
        elif parsed_path.path == '/last10days':
            # Return last 10 days of data including Date and Close
            response_data = last_10_days[["Date", "Close"]].to_dict(orient='records')
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode())
        
        else:
            super().do_GET()

# Start the server
PORT = 8080
server = HTTPServer(('localhost', PORT), PredictHandler)
print(f"Server running on http://localhost:{PORT}")
server.serve_forever()
