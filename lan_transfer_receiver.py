import socket
import serial
import time

# Create Bluetooth socket
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
arduino_port = 'COM3'  # Update with the correct port for your Arduino
baud_rate = 9600
ser = serial.Serial(arduino_port, baud_rate, timeout=1)
print('Not reaching here')
time.sleep(1)  # Wait for the connection to initialize

host = '10.71.36.55'

port = 8000
def send_angles(angles):
    angle_str = "<" + ", ".join(map(str, angles)) + ">"
    ser.write(f"{angle_str}\n".encode())  # Convert the angle string to bytes and send it
    print(f"Sent angles: {angle_str}")

try:
    print('Before connecting...')
    client.connect((host, port))
    print("Connected")

    while True:

        data = client.recv(1024)
        if not data:
            print("No data received. Exiting loop.")
            break

        x = data.decode('utf-8').strip()
        elements = x.split()  # Split the string into a list of elements
        comma_separated_x = x

        # Check if the received data is in the correct format

        if not (x.startswith('[') and x.endswith(']')):
            print("Invalid data format received:", comma_separated_x)
            continue

        # Print received data
        print("Received:", comma_separated_x)

        # Parse the received data
        try:
            angles = eval(comma_separated_x)  # Convert string representation of list to a list
            print(angles)
            if len(angles) != 6:
                print("Received data does not contain exactly 6 values:", angles)
                continue

            # Send the angles to the Arduino
            send_angles(angles)

            # Wait to avoid overwhelming the Arduino
            time.sleep(0.2)

        except (SyntaxError, ValueError) as e:
            print("Error parsing received data:", e)

except Exception as e:
    print('Connection or communication error:', e)

finally:
    client.close()
    ser.close()  # Close the serial port
    print("Serial connection closed")



