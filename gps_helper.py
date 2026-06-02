import serial
import pynmea2
import time

# open UART once
gps_port = serial.Serial("/dev/serial0", baudrate=9600, timeout=1)

def get_gps(timeout=60):
    """
    Try to get a valid GPS fix within `timeout` seconds.
    Returns (lat, lon) on success, or (None, None) if no fix.
    """
    start = time.time()

    while time.time() - start < timeout:
        line = gps_port.readline().decode(errors="ignore").strip()
        if not line:
            continue

        # GGA has fix quality + lat/lon
        if line.startswith("$GPGGA"):
            try:
                msg = pynmea2.parse(line)
                # fix_quality: 0 = invalid, 1 = GPS fix, 2 = DGPS fix
                if int(msg.gps_qual) > 0:
                    return msg.latitude, msg.longitude
            except Exception:
                continue

        # Can also use RMC:
        # if line.startswith("$GPRMC"):
        #   ...

    # timeout
    return None, None

if __name__ == "__main__":
    print("Waiting for GPS fix...")
    lat, lon = get_gps(timeout=120)
    print("Result:", lat, lon)
