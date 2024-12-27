import http.server
import ssl
import os

folder_to_serve = "images"  # Replace "my_folder" with the path to your folder
os.chdir(folder_to_serve)

# Define the request handler
Handler = http.server.SimpleHTTPRequestHandler

# Start the HTTP server on a specified address and port
server_address = ('0.0.0.0', 8086)  # Change port if needed
httpd = http.server.HTTPServer(server_address, Handler)

# Wrap the server socket with SSL
httpd.socket = ssl.wrap_socket(httpd.socket, certfile="server.pem", server_side=True)

print("Serving on https://nlp-in-477-l.soe.ucsc.edu:8086")
httpd.serve_forever()