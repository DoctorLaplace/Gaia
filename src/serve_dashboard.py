import http.server
import socketserver
import json
import optuna
import os
import webbrowser

PORT = 8080
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

def get_runs_data():
    db_path = "sqlite:///reports/optuna_study.db"
    try:
        study = optuna.load_study(study_name="gaia_hyperparameter_optimization", storage=db_path)
    except Exception as e:
        return {"error": str(e), "trials": []}

    trials_data = []
    for t in study.get_trials(deepcopy=False):
        trials_data.append({
            "number": t.number,
            "state": t.state.name,
            "value": t.value,
            "params": t.params,
            "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
            "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else None,
        })
    return {"trials": trials_data, "best_value": study.best_value if study.best_trial else None}

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/runs":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            data = get_runs_data()
            self.wfile.write(json.dumps(data).encode('utf-8'))
        else:
            # If requesting just '/', redirect to our new dashboard
            if self.path == "/":
                self.path = "/viz_dashboard/index.html"
            super().do_GET()
            
    def log_message(self, format, *args):
        pass

print(f"\n{'='*60}")
print(f"  GAIA TUNING DASHBOARD SERVER")
print(f"{'='*60}")
print(f"[*] API enabled at port {PORT}")
print(f"[*] CLICKABLE LINK: http://localhost:{PORT}/viz_dashboard/index.html")
print(f"{'='*60}\n")

try:
    webbrowser.open(f"http://localhost:{PORT}/viz_dashboard/index.html")
except:
    pass

try:
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print("[!] Press Ctrl+C to stop the server.\n")
        httpd.serve_forever()
except OSError:
    print(f"[!] Port {PORT} is already in use. Is another server running?")
except KeyboardInterrupt:
    print("\n[!] Server stopped.")
