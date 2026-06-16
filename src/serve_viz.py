import http.server
import socketserver
import webbrowser
import os

PORT = 8000
# Ensure we serve from the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

Handler = http.server.SimpleHTTPRequestHandler

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Keep the terminal clean by silencing access logs
        pass

print(f"\n{'='*60}")
print(f"  GAIA GEOSPATIAL EXPLORER & DIAGNOSTICS")
print(f"{'='*60}")
print(f"[*] Server initialized at port {PORT}")
print(f"[*] CLICKABLE LINKS:")
print(f"    - Interactive Difficulty Map: http://localhost:{PORT}/viz/difficulty_map.html")
print(f"    - Tile Wavelength Explorer:   http://localhost:{PORT}/viz/index.html")
print(f"{'='*60}\n")

# Try to open browser automatically
try:
    webbrowser.open(f"http://localhost:{PORT}/viz/index.html")
except:
    pass

try:
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("[!] Press Ctrl+C to stop the server.\n")
        httpd.serve_forever()
except OSError:
    print(f"[!] Port {PORT} is already in use. Is another server running?")
except KeyboardInterrupt:
    print("\n[!] Server stopped.")
