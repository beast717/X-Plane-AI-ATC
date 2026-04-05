import xpc
import socket

# --- THE FIX ---
# This small patch fixes the 'bytes' error you saw in the terminal
def patch_xpc():
    original_sendUDP = xpc.XPlaneConnect.sendUDP
    def new_sendUDP(self, message):
        if isinstance(message, str):
            message = message.encode('utf-8')
        return original_sendUDP(self, message)
    xpc.XPlaneConnect.sendUDP = new_sendUDP

patch_xpc()
# ---------------

def test_connection():
    print("--- Attempting to read from X-Plane ---")
    try:
        # Use a timeout so it doesn't hang forever if it can't find X-Plane
        with xpc.XPlaneConnect(timeout=3000) as client:
            # Requesting position of player aircraft (0)
            posi = client.getPOSI(0)
            
            if posi:
                altitude_ft = posi[2] * 3.28084
                heading = posi[5]
                print(f"SUCCESS!")
                print(f"Current Altitude: {altitude_ft:.1f} feet")
                print(f"Current Heading: {heading:.1f} degrees")
            else:
                print("FAILED: Connected to X-Plane but received no data.")
                
    except Exception as e:
        print(f"FAILED to connect: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure X-Plane is RUNNING and you are in the cockpit.")
        print("2. Check Plugins > XPlaneConnect > Status is 'Enabled'.")

if __name__ == "__main__":
    test_connection()