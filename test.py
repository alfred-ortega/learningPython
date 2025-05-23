import sqlite3
from datetime import datetime

# 1. Create/Connect to SQLite database
def create_database():
    try:
        conn = sqlite3.connect('tenable_scans.db')
        print("Database created/connected successfully")
        return conn
    except sqlite3.Error as e:
        print(f"Error creating database: {e}")
        return None

# 2. Create table for scan results
def create_scan_results_table(conn):
    try:
        cursor = conn.cursor()
        
        # Create table with relevant fields for Tenable scan results
        create_table_sql = '''
        CREATE TABLE IF NOT EXISTS scan_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id TEXT NOT NULL,
            hostname TEXT,
            ip_address TEXT,
            vulnerability_name TEXT,
            severity INTEGER,
            plugin_id INTEGER,
            description TEXT,
            solution TEXT,
            cvss_score REAL,
            first_discovered DATETIME,
            last_seen DATETIME,
            status TEXT,
            remediated BOOLEAN DEFAULT FALSE,
            remediation_date DATETIME,
            notes TEXT
        )
        '''
        
        cursor.execute(create_table_sql)
        conn.commit()
        print("Scan results table created successfully")
        
    except sqlite3.Error as e:
        print(f"Error creating table: {e}")

# 3. Function to insert sample data (for testing)
def insert_sample_data(conn):
    try:
        cursor = conn.cursor()
        
        sample_data = (
            'SCAN-001',
            'server1.example.com',
            '127.0.0.1',
            'SSL Certificate Expired',
            3,
            15901,
            'The SSL certificate has expired',
            'Renew SSL certificate',
            7.5,
            datetime.now(),
            datetime.now(),
            'Open',
            False,
            None,
            'Ticket created for certificate renewal'
        )
        
        insert_sql = '''
        INSERT INTO scan_results (
            scan_id, hostname, ip_address, vulnerability_name,
            severity, plugin_id, description, solution,
            cvss_score, first_discovered, last_seen,
            status, remediated, remediation_date, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        
        cursor.execute(insert_sql, sample_data)
        conn.commit()
        print("Sample data inserted successfully")
        
    except sqlite3.Error as e:
        print(f"Error inserting sample data: {e}")

# Main execution
def main():
    # Create database connection
    conn = create_database()
    
    if conn is not None:
        # Create table
        create_scan_results_table(conn)
        
        # Insert sample data
        insert_sample_data(conn)
        
        # Close connection
        conn.close()
    else:
        print("Error! Cannot create database connection.")

if __name__ == '__main__':
    main()