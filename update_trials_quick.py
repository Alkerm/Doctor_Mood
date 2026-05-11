#!/usr/bin/env python
import os
import threading
from dotenv import load_dotenv
import psycopg2

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL', '').strip()

if not DATABASE_URL:
    print("ERROR: DATABASE_URL not found in .env")
    exit(1)

# Handle postgres:// vs postgresql://
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

result = {'status': 'pending'}

def db_operation():
    try:
        print("Connecting to database...")
        conn = psycopg2.connect(
            DATABASE_URL,
            connect_timeout=5
        )
        print("Connected!")
        
        cur = conn.cursor()
        
        print("Updating trials to 0 for wedmoody277@gmail.com...")
        cur.execute(
            "UPDATE users SET total_uses = 0 WHERE email = %s",
            ('wedmoody277@gmail.com',)
        )
        
        conn.commit()
        print(f"Success! Rows affected: {cur.rowcount}")
        
        # Verify
        print("Verifying...")
        cur.execute(
            "SELECT id, email, total_uses, used_uses FROM users WHERE email = %s",
            ('wedmoody277@gmail.com',)
        )
        row = cur.fetchone()
        if row:
            print(f"ID: {row[0]}, Email: {row[1]}, Trials: {row[2]}, Used: {row[3]}")
        else:
            print("User not found - may need to be created after restart")
        
        cur.close()
        conn.close()
        print("Done!")
        result['status'] = 'success'
        
    except Exception as e:
        print(f"ERROR: {e}")
        result['status'] = 'error'

# Run with timeout
thread = threading.Thread(target=db_operation)
thread.daemon = True
thread.start()
thread.join(timeout=10)

if thread.is_alive():
    print("ERROR: Operation timed out - database not responding")
    exit(1)
elif result['status'] == 'error':
    exit(1)
else:
    exit(0)
