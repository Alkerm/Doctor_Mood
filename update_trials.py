#!/usr/bin/env python
import os
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

try:
    print("Connecting to database...")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    print("Connected!")
    
    print("\nUpdating trials to 0 for wedmoody277@gmail.com...")
    cur.execute(
        "UPDATE users SET total_uses = 0 WHERE email = %s",
        ('wedmoody277@gmail.com',)
    )
    
    conn.commit()
    print(f"Success! Rows affected: {cur.rowcount}")
    
    # Verify
    print("\nVerifying...")
    cur.execute(
        "SELECT id, email, total_uses, used_uses FROM users WHERE email = %s",
        ('wedmoody277@gmail.com',)
    )
    result = cur.fetchone()
    if result:
        print(f"ID: {result[0]}")
        print(f"Email: {result[1]}")
        print(f"Total Trials: {result[2]}")
        print(f"Used: {result[3]}")
    else:
        print("User not found!")
    
    cur.close()
    conn.close()
    print("\nDone!")
    
except Exception as e:
    print(f"ERROR: {e}")
    exit(1)
