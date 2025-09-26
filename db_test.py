from astrapy import DataAPIClient
from dotenv import load_dotenv
import os
load_dotenv()
    


def test_astra_connection(): 
    try: 
        client = DataAPIClient(os.getenv("ASTRA_DB_APPLICATION_TOKEN"))
        db = client.get_database_by_api_endpoint(
        os.getenv("ASTRA_DB_API_ENDPOINT")
        )

        print(f"Connected to Astra DB: {db.list_collection_names()}")
        return True

    except Exception as e: 
        raise ValueError(f"Error occurred with exception : {e}")
