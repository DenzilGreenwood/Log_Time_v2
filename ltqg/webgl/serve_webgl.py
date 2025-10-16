#!/usr/bin/env python3
"""
Simple HTTP Server for LTQG WebGL Visualizations

Serves the WebGL demonstrations locally with proper MIME types
and cross-origin headers for interactive visualization.
"""

import http.server
import socketserver
import os
import webbrowser
import sys
from pathlib import Path

class LTQGWebGLHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler with proper MIME types for WebGL content."""
    
    def end_headers(self):
        """Add CORS headers for WebGL compatibility."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

def serve_webgl(port=8000, auto_open=True):
    """
    Start local server for LTQG WebGL visualizations.
    
    Args:
        port: Port number (default 8000)
        auto_open: Whether to automatically open browser
    """
    # Change to webgl directory
    webgl_dir = Path(__file__).parent
    os.chdir(webgl_dir)
    
    print(f"="*60)
    print(f"LTQG WebGL Visualization Server")
    print(f"="*60)
    print(f"Starting server on port {port}...")
    print(f"Directory: {webgl_dir}")
    
    # List available visualizations
    html_files = list(webgl_dir.glob("*.html"))
    if html_files:
        print(f"\nAvailable visualizations:")
        for html_file in html_files:
            print(f"  • http://localhost:{port}/{html_file.name}")
    else:
        print("\nNo HTML files found in webgl directory")
    
    try:
        with socketserver.TCPServer(("", port), LTQGWebGLHandler) as httpd:
            print(f"\nServer running at http://localhost:{port}/")
            print(f"Press Ctrl+C to stop the server")
            
            if auto_open and html_files:
                # Open the first visualization automatically
                url = f"http://localhost:{port}/{html_files[0].name}"
                print(f"Opening {url} in default browser...")
                webbrowser.open(url)
            
            print(f"="*60)
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print(f"\nServer stopped by user")
    except OSError as e:
        print(f"Error starting server: {e}")
        print(f"Port {port} may already be in use")
        return False
    
    return True

def check_dependencies():
    """Check if required modules are available."""
    try:
        import webbrowser
        return True
    except ImportError as e:
        print(f"Warning: Missing dependencies: {e}")
        return False

if __name__ == "__main__":
    if not check_dependencies():
        sys.exit(1)
    
    # Parse command line arguments
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}")
            sys.exit(1)
    
    serve_webgl(port=port)