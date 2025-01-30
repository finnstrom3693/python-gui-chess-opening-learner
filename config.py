import json
import tkinter as tk
from tkinter import filedialog, messagebox

# Default JSON configuration file name
CONFIG_FILE = "config.json"

# Function to load JSON file
def load_json():
    try:
        with open(CONFIG_FILE, "r") as file:
            data = json.load(file)
            tokenizer_var.set(data.get("tokenizer", ""))
            onnx_model_var.set(data.get("onnx_model", ""))
    except FileNotFoundError:
        messagebox.showerror("Error", "Config file not found!")
    except json.JSONDecodeError:
        messagebox.showerror("Error", "Invalid JSON format!")

# Function to save JSON file
def save_json():
    data = {
        "tokenizer": tokenizer_var.get(),
        "onnx_model": onnx_model_var.get()
    }
    try:
        with open(CONFIG_FILE, "w") as file:
            json.dump(data, file, indent=4)
        messagebox.showinfo("Success", "Config saved successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save config: {e}")

# Function to browse and load a JSON file
def browse_file():
    global CONFIG_FILE
    filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
    if filename:
        CONFIG_FILE = filename
        load_json()

# GUI setup
root = tk.Tk()
root.title("JSON Config Editor")
root.geometry("400x250")
root.resizable(False, False)

# Labels and entry fields
tk.Label(root, text="Tokenizer:").pack(pady=5)
tokenizer_var = tk.StringVar()
tokenizer_entry = tk.Entry(root, textvariable=tokenizer_var, width=50)
tokenizer_entry.pack()

tk.Label(root, text="ONNX Model:").pack(pady=5)
onnx_model_var = tk.StringVar()
onnx_model_entry = tk.Entry(root, textvariable=onnx_model_var, width=50)
onnx_model_entry.pack()

# Buttons
tk.Button(root, text="Load Config", command=load_json).pack(pady=5)
tk.Button(root, text="Save Config", command=save_json).pack(pady=5)
tk.Button(root, text="Browse File", command=browse_file).pack(pady=5)

# Load default JSON file on startup
load_json()

# Run the Tkinter event loop
root.mainloop()
