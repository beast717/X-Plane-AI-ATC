import sys
import threading
import asyncio
import atc_live

# --- FAKE X-PLANE STATE ---
state = {
    "freq_hz": 12800,       # Start on 128.00 (ATIS)
    "lat": 58.8767,         # ENZV
    "lon": 5.6378,
    "alt_m": 9.0,
    "agl_m": 2.0,
    "hdg": 290.0,
    "spd": 0.0,
    "squawk": 1200,
    "tail": "SAS123"
}

# --- MOCK X-PLANE CONNECTION ---
class MockXPlane:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): pass

    def getDREF(self, dref):
        if "com1_freq_hz" in dref: return [state["freq_hz"]]
        if "y_agl" in dref: return [state["agl_m"]]
        if "mag_psi" in dref: return [state["hdg"]]
        if "indicated_airspeed" in dref: return [state["spd"]]
        if "transponder_code" in dref: return [state["squawk"]]
        if "acf_tailnum" in dref: return [ord(c) for c in state["tail"]] + [0]
        if "wind_direction_degs" in dref: return [150.0]
        if "wind_speed_kt" in dref: return [24.0]
        if "barometer_sealevel_inhg" in dref: return [29.92]
        return [0.0]

    def getPOSI(self, idx):
        return [state["lat"], state["lon"], state["alt_m"]]

atc_live.xpc.XPlaneConnect = MockXPlane

# --- TEXT INJECTION MOCKING (NO MIC NEEDED) ---
text_queue = []
current_text = ""

async def mock_record_audio(squelch=None, fs=44100):
    """Replaces the microphone check with a text-queue check."""
    global current_text
    while True:
        # Check if tuned to ATIS
        com1_mhz = state["freq_hz"] / 100.0
        if abs(com1_mhz - 128.000) < 0.01:
            return "ATIS"
        
        # Check if text was typed in the console
        if text_queue:
            current_text = text_queue.pop(0)
            if squelch: squelch.play()
            
            # Create a dummy file so your main script doesn't crash trying to read the audio file
            with open("mic_input.wav", "wb") as f:
                f.write(b"dummy_audio_data")
            
            return "PTT"
            
        await asyncio.sleep(0.2)

class MockTranscriptions:
    """Replaces Groq's Whisper API so we don't waste API calls on fake audio."""
    def create(self, *args, **kwargs):
        print(f"\n🎙️  [MOCK MIC INJECTED]: {current_text}")
        return current_text

# Apply the patches to your live code
atc_live.record_audio_ptt = mock_record_audio
atc_live.client.audio.transcriptions = MockTranscriptions()

# --- TEST BENCH CONTROL PANEL ---
def control_panel():
    import time
    time.sleep(2)
    print("\n" + "="*60)
    print("🎛️  TEXT-BASED ATC TEST BENCH ACTIVE  🎛️")
    print("Location: ENZV (Stavanger) | Callsign: Scandinavian 123")
    print("="*60)
    print("COMMANDS:")
    print("  say <text> -> E.g., say Sola Delivery, Scandinavian 123...")
    print("  f <freq>   -> Tune COM1 (e.g., f 121.9)")
    print("  a <alt>    -> Set Altitude in feet (e.g., a 5000)")
    print("  sq <code>  -> Set Squawk (e.g., sq 4321)")
    print("  takeoff    -> Simulate takeoff")
    print("="*60 + "\n")

    while True:
        try:
            cmd = input().strip()
            if not cmd: continue
            
            parts = cmd.split()
            command = parts[0].lower()

            if command == 'say':
                # Grab everything after 'say' and put it in the queue
                spoken_text = " ".join(parts[1:])
                text_queue.append(spoken_text)
            elif command == 'f' and len(parts) > 1:
                state['freq_hz'] = int(float(parts[1]) * 100)
                print(f"📻 [SIM] Radio tuned to {parts[1]} MHz")
            elif command == 'a' and len(parts) > 1:
                alt_ft = int(parts[1])
                state['alt_m'] = alt_ft * 0.3048
                state['agl_m'] = state['alt_m'] if alt_ft > 50 else 2.0 
                print(f"✈️ [SIM] Altitude jumped to {alt_ft} ft")
            elif command == 'sq' and len(parts) > 1:
                state['squawk'] = int(parts[1])
                print(f"✈️ [SIM] Squawk changed to {state['squawk']}")
            elif command == 'takeoff':
                state['agl_m'] = 500.0
                state['alt_m'] = 500.0 * 0.3048
                state['spd'] = 160.0
                print("🚀 [SIM] Aircraft is now AIRBORNE (500ft, 160kts)")
            else:
                print("⚠️ Unknown command.")
        except Exception as e:
            pass

if __name__ == "__main__":
    threading.Thread(target=control_panel, daemon=True).start()
    asyncio.run(atc_live.run_atc_loop())