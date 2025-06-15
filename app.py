from flask import Flask, render_template, request, redirect, session, jsonify, url_for, send_file
import sqlite3, csv, os, qrcode, json
from io import BytesIO
import base64
import dlib
import numpy as np
import cv2
from datetime import datetime, date

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load Dlib models
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

# ---------------------- DB Init & Load CSV -----------------------
def init_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        enrollment_no TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(student_id) REFERENCES students(id)
    )''')
    conn.commit()
    conn.close()

def load_students_from_csv():
    if not os.path.exists('students.csv'):
        return
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    with open('students.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            c.execute("INSERT OR IGNORE INTO students (name, enrollment_no, password) VALUES (?, ?, ?)",
                      (row['name'], row['enrollment_no'], row.get('password', '1234')))
    conn.commit()
    conn.close()

# ------------------------ Routes -----------------------------
@app.route('/')
def home():
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == 'admin123':
            session['admin_logged_in'] = True
            return redirect('/admin')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    return redirect('/login')

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if not session.get('admin_logged_in'):
        return redirect('/login')

    date_filter = request.form.get('date') if request.method == 'POST' else None

    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    if date_filter:
        c.execute('''SELECT students.name, students.enrollment_no, attendance.timestamp FROM attendance
                     JOIN students ON attendance.student_id = students.id
                     WHERE DATE(attendance.timestamp) = ?
                     ORDER BY attendance.timestamp DESC''', (date_filter,))
    else:
        c.execute('''SELECT students.name, students.enrollment_no, attendance.timestamp FROM attendance
                     JOIN students ON attendance.student_id = students.id
                     ORDER BY attendance.timestamp DESC''')
    records = c.fetchall()
    conn.close()
    return render_template('admin.html', data=records)

@app.route('/export_csv')
def export_csv():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('''SELECT students.name, students.enrollment_no, attendance.timestamp
                 FROM attendance
                 JOIN students ON attendance.student_id = students.id''')
    rows = c.fetchall()
    conn.close()

    output = BytesIO()
    writer = csv.writer(output)
    writer.writerow(['Name', 'Enrollment No', 'Timestamp'])
    writer.writerows(rows)
    output.seek(0)
    return send_file(output, mimetype='text/csv', download_name='export.csv', as_attachment=True)

@app.route('/index', methods=['GET', 'POST'])
def index():
    if not session.get('admin_logged_in'):
        return redirect('/login')

    qr_code = None

    if request.method == 'POST':
        name = request.form['name']
        enrollment_no = request.form['enrollment_no']
        password = request.form['password']

        conn = sqlite3.connect('attendance.db')
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO students (name, enrollment_no, password) VALUES (?, ?, ?)",
                  (name, enrollment_no, password))
        conn.commit()
        conn.close()

        qr_data = json.dumps({
            "enrollment_no": enrollment_no,
            "password": password
        })

        img = qrcode.make(qr_data)
        buf = BytesIO()
        img.save(buf, format='PNG')
        qr_code = base64.b64encode(buf.getvalue()).decode('utf-8')

    return render_template('index.html', qr_code=qr_code)

@app.route('/scan')
def scan():
    return render_template('scan.html')

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    data = request.get_json()
    try:
        qr_raw = data.get('qr_data')
        image_data = data.get('face_image')  # base64 webcam image
        parsed = json.loads(qr_raw)
        enrollment_no = parsed['enrollment_no']
        password = parsed['password']

        conn = sqlite3.connect('attendance.db')
        c = conn.cursor()
        c.execute("SELECT id FROM students WHERE enrollment_no = ? AND password = ?", (enrollment_no, password))
        student = c.fetchone()

        if not student:
            return jsonify({"message": "Student not found or wrong password.", "status": "error"})

        # --- Face recognition check ---
        face_path = os.path.join("face_data", f"{enrollment_no}.npy")
        if not os.path.exists(face_path):
            return jsonify({"message": f"No reference face found for {enrollment_no}", "status": "error"})

        # Decode base64 image
        img_bytes = base64.b64decode(image_data.split(',')[1])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        dets = detector(frame)
        if len(dets) == 0:
            return jsonify({"message": "No face detected in image", "status": "error"})

        shape = sp(frame, dets[0])
        face_descriptor = facerec.compute_face_descriptor(frame, shape)
        face_descriptor_np = np.array(face_descriptor)

        ref_descriptor = np.load(face_path)
        distance = np.linalg.norm(ref_descriptor - face_descriptor_np)
        print(f"🔍 Face match distance: {distance}")

        if distance < 0.75:
            # Check if attendance already marked today
            today = datetime.now().date().isoformat()

            c.execute("SELECT * FROM attendance WHERE student_id = ? AND DATE(timestamp) = ?", (student[0], today))
            if c.fetchone():
                conn.close()
                return jsonify({"message": "Attendance already marked today.", "status": "error"})

            # Mark attendance
            c.execute("INSERT INTO attendance (student_id) VALUES (?)", (student[0],))
            conn.commit()
            conn.close()
            print("✅ Face matched. Attendance marked.")
            return jsonify({"message": "Attendance marked with face verification", "status": "success"})
        else:
            print("❌ Face did not match.")
            return jsonify({"message": "Face doesn't match", "status": "error"})

    except Exception as e:
        print("⚠️ Error in mark_attendance:", e)
        return jsonify({"message": "Failed due to error", "status": "error"})

# ---------------------- Main --------------------------
if __name__ == '__main__':
    init_db()
    load_students_from_csv()
    app.run(debug=True)
