#!/usr/bin/env python3
"""
Start an ngrok tunnel for live demos and print the public URL + QR code.

Install once:
    pip install pyngrok qrcode[pil]

Usage:
    python scripts/demo_tunnel.py          # tunnels localhost:8000
    python scripts/demo_tunnel.py 8080     # custom port

The Raspberry Pi should use the Cloudflare persistent tunnel to POST heatmaps.
This script is only for sharing the app with demo guests.
"""
import sys

try:
    from pyngrok import ngrok
except ImportError:
    print("Missing dependency. Run: pip install pyngrok qrcode[pil]")
    sys.exit(1)

try:
    import qrcode
except ImportError:
    print("Missing dependency. Run: pip install pyngrok qrcode[pil]")
    sys.exit(1)

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

print(f"Opening ngrok tunnel to localhost:{port}...")
tunnel = ngrok.connect(port, "http")
url = tunnel.public_url

print(f"\nPublic URL: {url}")
print("\nScan to open on phone:\n")

qr = qrcode.QRCode(border=2)
qr.add_data(url)
qr.make(fit=True)
qr.print_ascii(invert=True)

print(f"\nShare the link above with guests — they can access the full app.")
print("Press Ctrl+C to close the tunnel when the demo is done.\n")

try:
    input()
except KeyboardInterrupt:
    pass

ngrok.disconnect(url)
print("\nTunnel closed.")
