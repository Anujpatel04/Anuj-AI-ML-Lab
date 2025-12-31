#!/usr/bin/env python3
"""
Simple script to test MySQL connection
Run this to verify your MySQL credentials work
"""
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text

# Connection details - UPDATE THESE
HOST = "localhost"
PORT = "3306"
DATABASE = "analytics_chatdb"
USERNAME = "root"
PASSWORD = "Test@123"  # Update this if different

# Build connection string
encoded_username = quote_plus(USERNAME)
encoded_password = quote_plus(PASSWORD)
connection_string = f"mysql+pymysql://{encoded_username}:{encoded_password}@{HOST}:{PORT}/{DATABASE}"

print(f"Testing connection to: mysql://{USERNAME}:***@{HOST}:{PORT}/{DATABASE}")
print("=" * 60)

try:
    engine = create_engine(
        connection_string,
        echo=False,
        connect_args={
            "charset": "utf8mb4",
            "connect_timeout": 10
        }
    )
    
    with engine.connect() as conn:
        result = conn.execute(text("SELECT VERSION()"))
        version = result.fetchone()[0]
        print(f"✅ Connection successful!")
        print(f"MySQL Version: {version}")
        
        # Test database access
        result = conn.execute(text(f"SELECT DATABASE()"))
        current_db = result.fetchone()[0]
        print(f"Current Database: {current_db}")
        
        # List tables
        result = conn.execute(text("SHOW TABLES"))
        tables = result.fetchall()
        if tables:
            print(f"\nTables in database ({len(tables)}):")
            for table in tables:
                print(f"  - {table[0]}")
        else:
            print("\nNo tables found in database")
            
except Exception as e:
    print(f"❌ Connection failed: {str(e)}")
    print("\nTroubleshooting tips:")
    print("1. Verify MySQL server is running")
    print("2. Check username and password are correct")
    print("3. Ensure user has permissions to access the database")
    print("4. Try connecting without database name first:")
    print(f"   mysql+pymysql://{encoded_username}:{encoded_password}@{HOST}:{PORT}/")

