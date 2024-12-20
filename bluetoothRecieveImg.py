import os
import bluetooth
import base64
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext
from PIL import Image, ImageTk
from io import BytesIO
import time  # Importing the time module


# Global variables
server_sock = None
client_sock = None
client_info = None
port = None
uuid = None
flag='True'


def create_socket():
    global flag
    global server_sock, port, uuid
    uuid = uuid_entry.get()
    if not uuid:
        messagebox.showerror("Error", "UUID cannot be empty.")
        return
    server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    server_sock.bind(("", bluetooth.PORT_ANY))
    server_sock.listen(1)
    port = server_sock.getsockname()[1]
    bluetooth.advertise_service(server_sock, "SampleServer", service_id=uuid,
                                service_classes=[uuid, bluetooth.SERIAL_PORT_CLASS],
                                profiles=[bluetooth.SERIAL_PORT_PROFILE])
    output_text.insert(tk.END, f"\n")
    if flag:
        output_text.delete(1.0, tk.END)
        flag = 'False'
    output_text.insert(tk.END, f"Socket created and listening on port {port}\n")
    output_text.insert(tk.END, "Waiting for connection...\n")

def connect():
    global flag
    global client_sock, client_info
    client_sock, client_info = server_sock.accept()
    if flag:
        output_text.delete(1.0, tk.END)
        flag = 'False'
    output_text.insert(tk.END, f"\n")
    output_text.insert(tk.END, f"Connection established with {client_info}\n")

# Function to justify text
def insert_justified_text(widget, text, alignment='left', flush_after_initial=True):
    if flush_after_initial:
        widget.delete(1.0, tk.END)  # Clear existing text
    lines = text.split('\n')
    max_width = 75
    for line in lines:
        if alignment == 'left':
            widget.insert(tk.END, line + '\n')
        elif alignment == 'center':
            widget.insert(tk.END, line.center(max_width) + '\n')
        elif alignment == 'right':
            widget.insert(tk.END, line.rjust(max_width) + '\n')

def receive_images():
    global flag

    global client_sock
    image_count = 1
    try:
        while True:
            stringImg = b""
            try:
                while True:
                    data = client_sock.recv(1024)
                    if b"stop" in data:
                        stringImg += data.split(b"stop")[0]  # Get data before "stop"
                        break
                    if not data:
                        break
                    stringImg += data
            except IOError as e:
                output_text.insert(tk.END, f"IOError while receiving data: {e}\n")
                break
            except bluetooth.btcommon.BluetoothError as e:
                output_text.insert(tk.END, f"Bluetooth error: {e}\n")
                break
            except Exception as e:
                output_text.insert(tk.END, f"Unexpected error: {e}\n")
                break

            # Ensure stringImg is properly padded
            while len(stringImg) % 4 != 0:
                stringImg += b'='

            try:
                decoded_data = base64.b64decode(stringImg)
            except base64.binascii.Error as e:
                output_text.insert(tk.END, f"Base64 decode error: {e}\n")
                continue

            folder_path = r'E:\Indumathi\Multiview-3D-Reconstruction-main\Multiview-3D-Reconstruction-main\Received_images'
            os.makedirs(folder_path, exist_ok=True)

            filename = os.path.join(folder_path, f'img{image_count}.jpg')

            with open(filename, 'wb') as img_result:
                img_result.write(decoded_data)
                output_text.insert(tk.END, f"Image {filename} successfully received and saved.\n")
                output_text.insert(tk.END, f"Image {filename} successfully received and saved.\n")

            image_count += 1
            if image_count > 10:
                image_count = 1
                # time.sleep(60)

    except KeyboardInterrupt:
        output_text.insert(tk.END, "\nProgram terminated by user.\n")

    finally:
        output_text.insert(tk.END, "Disconnected.\n")
        if client_sock:
            client_sock.close()
        if server_sock:
            server_sock.close()
        output_text.insert(tk.END, "All done. Images received and saved.\n")


def start_receiving_images():
    global flag
    if flag:
        output_text.delete(1.0, tk.END)
        flag = 'False'
    threading.Thread(target=receive_images).start()

# Creating the GUI
root = tk.Tk()
root.title("Remote Surgery Using Augmented Reality Supported by Directional Intensified features")
root.geometry("1600x1000")  # Set the window size

# Load the images
image_path = r"E:\Indumathi\Multiview-3D-Reconstruction-main\Multiview-3D-Reconstruction-main\imgs\Model.png"
connect_icon_path = r"E:\Indumathi\Multiview-3D-Reconstruction-main\Multiview-3D-Reconstruction-main\imgs\Connect.png"
receive_icon_path = r"E:\Indumathi\Multiview-3D-Reconstruction-main\Multiview-3D-Reconstruction-main\imgs\Receive.png"
bind_icon_path = r"E:\Indumathi\Multiview-3D-Reconstruction-main\Multiview-3D-Reconstruction-main\imgs\Socket.jpeg"

# Resize and load the main image
img = Image.open(image_path)
img = img.resize((150, 150))  # Resize the image to fit
img = ImageTk.PhotoImage(img)

# Load the icons
connect_icon = Image.open(connect_icon_path).resize((70, 70))
connect_icon = ImageTk.PhotoImage(connect_icon)
receive_icon = Image.open(receive_icon_path).resize((70, 70))
receive_icon = ImageTk.PhotoImage(receive_icon)
bind_icon = Image.open(bind_icon_path).resize((70, 70))
bind_icon = ImageTk.PhotoImage(bind_icon)


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb
# Function to create rounded buttons

# Create frames for layout
img_frame = tk.Frame(root)
img_frame.grid(row=0, column=0, padx=50, pady=20, sticky="n")

control_frame = tk.Frame(root)
control_frame.grid(row=0, column=1, padx=150, pady=80, sticky="n")

output_frame = tk.Frame(root)
output_frame.grid(row=0, column=2, padx=0, pady=20, sticky="n")

# Display the image
img_label = tk.Label(img_frame, image=img)
img_label.pack()

# Create a frame for the UUID entry
uuid_frame = tk.Frame(control_frame)
uuid_frame.pack(pady=10)

label_font = ("times", 12, "bold")

tk.Label(uuid_frame, text="UUID:", font=label_font).pack(side=tk.LEFT, padx=10)
uuid_entry = tk.Entry(uuid_frame, width=33, bg="white", fg=rgb_to_hex((100,75,150)))  # Set the width of the entry field
uuid_entry.insert(0, "94f39d29-7d6d-437d-973b-fba39e49d4ee")  # Default UUID
uuid_entry.pack(side=tk.LEFT, ipadx=10, ipady=5)  # Add padding to the entry field

def create_rounded_button(frame, text, icon, command):
    button = tk.Button(frame, wraplength = 68,text=text, width = 18, height = 4, command=command, font=("times", 12, "bold"), bg= rgb_to_hex((100,75,150))  , fg="white",  compound="left")
    button.config(highlightbackground="purple", highlightthickness=2, bd=0, relief="flat")
    button.pack(pady=(100, 10), ipadx=10, ipady=5, anchor="center")
    return button

# Function to create square buttons
def create_square_button(frame, text, icon, command):
    button = tk.Button(frame, wraplength = 80, text=text, width = 15, height = 5, command=command, font=("times", 12, "bold"), bg=rgb_to_hex((100,75,150)), fg="white",  compound="left")
    button.pack(pady=(100, 10), padx=(100, 100), ipadx=10, ipady=5, anchor="center")
    return button

# Create and Bind Socket button with icon
bind_button = create_rounded_button(control_frame, 'Create      &        Bind Socket', bind_icon, create_socket)

# Connect button with icon
connect_button = create_square_button(control_frame, "Connect", connect_icon, connect)

# Receive Images button with icon
receive_button = create_square_button(control_frame, "Do surveillance of Surgical Field", receive_icon, start_receiving_images)

# Output text box for displaying notifications
txt = '\n\n\n\n\n\n 1. You are withiin an user interface of an omni directional surveillance system. \n\n 2. The interface facilitates a doctor to conduct a surgery from remote. \n\n 3. It uses an effective 3D reconstruction of features obtained from 360 degree around the surgical environment. \n\n 4. The interface senses the environment every once in 2 seconds. \n\n 5. It gather information in terms of 2D spatial images. \n\n 6. The gathered images will be used for the feature extraction and 3D reconstruction. \n\n 7. Enjoy operating!'
output_text = scrolledtext.ScrolledText(output_frame, width=75, height=50, wrap=tk.WORD, foreground='purple')
output_text.pack(padx=10, pady=10)
output_text.pack(side=tk.LEFT, expand=True, pady=10)
flag = 'True'
# Insert the justified text initially
insert_justified_text(output_text, txt, alignment='left')

# Flush the text field after initial message
# root.after(3000, lambda: output_text.delete(1.0, tk.END))  # Clear the text field after 3 seconds

# Start the Tkinter main loop
root.mainloop()
