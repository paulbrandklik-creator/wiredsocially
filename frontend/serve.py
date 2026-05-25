import http.server, os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
port = int(sys.argv[1]) if len(sys.argv) > 1 else 3456
http.server.test(HandlerClass=http.server.SimpleHTTPRequestHandler, port=port)
