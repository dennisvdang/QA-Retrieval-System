import sqlite3
import pandas as pd
import os

# Paths and constants
db_path = '../data/SQL/Probes.db'
df = pd.read_csv('../data/Catalog - Probes.csv')

# Check if the database exists and remove it
if os.path.exists(db_path):
    os.remove(db_path)

# Using a context manager to handle the database connection
with sqlite3.connect(db_path) as conn:
    c = conn.cursor()

    # Create the 'probes' table with Probe_ID column
    c.execute('''
    CREATE TABLE probes (
        Probe_ID INTEGER PRIMARY KEY AUTOINCREMENT,
        Manufacturer TEXT,
        Probe_Model TEXT,
        Connection_Type TEXT,
        Array_Type TEXT,
        Frequency_Range TEXT,
        Applications TEXT,
        Stock INTEGER,
        Description TEXT
    )
    ''')

    # Insert data into the 'probes' table
    for row in df.itertuples(index=False):
        c.execute('''
        INSERT INTO probes (Manufacturer, Probe_Model, Connection_Type, Array_Type, Frequency_Range, Applications, Stock, Description) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (row.Manufacturer, row.Probe_Model, row.Connection_Type, row.Array_Type, row.Frequency_Range, row.Applications, row.Stock, row.Description))

    # Create the 'systems' table
    c.execute('''
    CREATE TABLE IF NOT EXISTS systems (
        System_ID INTEGER PRIMARY KEY AUTOINCREMENT,
        System_Name TEXT UNIQUE,
        Manufacturer TEXT
    )
    ''')

    # Modify the 'compatibility' table schema
    c.execute('''
    CREATE TABLE IF NOT EXISTS compatibility (
        Compatibility_ID INTEGER PRIMARY KEY AUTOINCREMENT,
        Probe_ID INTEGER,
        System_ID INTEGER,
        FOREIGN KEY(Probe_ID) REFERENCES probes(Probe_ID),
        FOREIGN KEY(System_ID) REFERENCES systems(System_ID)
    )
    ''')

    # Extract unique systems with manufacturers
    systems_data = df[['Manufacturer', 'Compatible_Systems']].drop_duplicates()
    systems_data['Systems'] = systems_data['Compatible_Systems'].str.split(', ')

    # Explode and drop duplicates for unique system-manufacturer pairs
    systems_expanded = systems_data.explode('Systems')
    unique_systems = systems_expanded.drop_duplicates(subset=['Systems', 'Manufacturer'])

    # Insert unique systems into the 'systems' table
    for _, row in unique_systems.iterrows():
        c.execute("INSERT OR IGNORE INTO systems (System_Name, Manufacturer) VALUES (?, ?)", (row['Systems'], row['Manufacturer']))

    # Commit to save the systems
    conn.commit()

    # Map Probe_ID and System_ID and insert additional details into the compatibility table
    for index, row in df.iterrows():
        probe_model = row['Probe_Model']
        systems = row['Compatible_Systems'].split(', ')
        for system in systems:
            # Get the Probe_ID for the current Probe_Model
            c.execute("SELECT Probe_ID FROM probes WHERE Probe_Model = ?", (probe_model,))
            probe_id = c.fetchone()[0]
            
            # Get the System_ID for the current system
            c.execute("SELECT System_ID FROM systems WHERE System_Name = ?", (system,))
            system_id = c.fetchone()[0]
            
            # Insert into compatibility table with additional details
            c.execute("INSERT INTO compatibility (Probe_ID, System_ID) VALUES (?, ?)", (probe_id, system_id))

    # Commit the changes
    conn.commit()