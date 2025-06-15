import qrcode
import os

# Sample student list (you can connect it to DB later)
students = ["Sakshi_001", "Aman_002", "Priya_003", "Raj_004"]

# Create folder if it doesn't exist
if not os.path.exists("qrcodes"):
    os.makedirs("qrcodes")

for student in students:
    img = qrcode.make(student)
    img.save(f"qrcodes/{student}.png")

print("✅ QR codes generated for all students!")
