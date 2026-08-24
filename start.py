"""
BlueByte AI — Quick Start Script
Initializes the database with sample data and launches the API server.
Run: python start.py
"""
import asyncio
import subprocess
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))


async def init_database():
    """Initialize database and seed with sample data."""
    print("[DB] Initializing database...")
    try:
        from db.connection import init_db
        await init_db()
        print("[OK] Database tables created")
    except Exception as e:
        print(f"[WARN] Database table creation: {e}")

    print("[DB] Seeding sample data...")
    try:
        from db.synthetic_data.generate_synthetic_dataset import main as generate_dataset
        from db.load_synthetic_dataset import load_all

        generate_dataset()  # writes CSV/JSON files to db/synthetic_data/data/
        await load_all()     # loads them into Postgres (buoy/river through
                              # the real outlier-preserving ETL, not a shortcut)
        print("[OK] Sample data generated and seeded successfully")
    except Exception as e:
        print(f"[WARN] Seeding: {e}")


def main():
    print("""
    ======================================================
    |                                                    |
    |         BLUEBYTE AI - Ocean Intelligence           |
    |                                                    |
    |   AI-Driven Unified Data Platform for              |
    |   Oceanographic, Fisheries & Molecular             |
    |   Biodiversity Insight                             |
    |                                                    |
    ======================================================
    """)

    # Step 1: Initialize database
    print("-" * 55)
    print("STEP 1: Database Initialization")
    print("-" * 55)
    asyncio.run(init_database())

    # Step 2: Launch API server
    print()
    print("-" * 55)
    print("STEP 2: Launching API Server")
    print("-" * 55)
    print("[*] API Server starting on http://localhost:8000")
    print("[*] API Docs available at http://localhost:8000/docs")
    print("[*] Dashboard at http://localhost:8000/ (serves client/index.html)")
    print()
    print("[TIP] To enable live streaming, run in separate terminals:")
    print("   Terminal 2: python server/broker/stream_broker.py")
    print("   Terminal 3: python server/broker/telemetry_publisher.py")
    print()
    print("Press Ctrl+C to stop the server.")
    print("-" * 55)

    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "server.api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
    ])


if __name__ == "__main__":
    main()