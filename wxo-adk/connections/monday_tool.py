# monday_request_tool.py
import requests
from typing import List, Dict, Any
from ibm_watsonx_orchestrate.agent_builder.tools import tool, ToolPermission
from ibm_watsonx_orchestrate.run import connections
from ibm_watsonx_orchestrate.agent_builder.connections import ConnectionType, ExpectedCredentials

test_BOARD_ID = <ENTER YOUR MONDAY BOARD ID HERE>

@tool(
    name="monday_list",
    description="Retrieves all data for requests from a Monday.com board that match a specific title",
    permission=ToolPermission.READ_ONLY,
    expected_credentials=[ExpectedCredentials(
        app_id="monday.com",
        type=ConnectionType.API_KEY_AUTH
    )]
)
def monday_list(title: str) -> Dict[str, Any]:
    """
    Connects to Monday.com and retrieves all data for requests that match the provided title.
    
    :param title: The title of the Monday.com request to search for
    :param board_id: The ID of the Monday.com board to search in
    :returns: Dictionary containing matching request data or error message
    """

    board_id = test_BOARD_ID
    
    try:
        # Get the connection credentials
        conn = connections.api_key_auth("monday.com")
        
        # Monday.com GraphQL API endpoint
        url = "https://api.monday.com/v2"
        
        # Headers for the API request
        headers = {
            "Authorization": conn.api_key,
            "Content-Type": "application/json"
        }
        
        # GraphQL query to get items from the specified board
        query = """
        query ($board_id: ID!) {
            boards (ids: [$board_id]) {
                items_page {
                    items {
                        id
                        name
                    }
                }
            }
        }
        """
        
        # Variables for the GraphQL query
        variables = {
            "board_id": board_id
        }
        
        # Make the API request
        response = requests.post(
            url,
            json={"query": query, "variables": variables},
            headers=headers
        )
        
        # Check if the request was successful
        if response.status_code != 200:
            return {
                "error": f"API request failed with status code {response.status_code}",
                "message": response.text
            }
        
        # Parse the response
        data = response.json()
        
        # Check for GraphQL errors
        if "errors" in data:
            return {
                "error": "GraphQL errors occurred",
                "errors": data["errors"]
            }
        
        # Extract items from the response
        boards = data.get("data", {}).get("boards", [])
        if not boards:
            return {
                "error": f"Board with ID {board_id} not found"
            }
        
        items = boards[0].get("items_page", {}).get("items", [])
        
        # Filter items by title (case-insensitive partial match)
        matching_items = []
        for item in items:
            if title.lower() in item.get("name", "").lower():
                matching_items.append(item)
        
        # Return the results
        if matching_items:
            return {
                "success": True,
                "board_id": board_id,
                "search_title": title,
                "total_matches": len(matching_items),
                "matching_requests": matching_items
            }
        else:
            return {
                "success": True,
                "board_id": board_id,
                "search_title": title,
                "total_matches": 0,
                "message": f"No requests found with title containing '{title}'"
            }
            
    except Exception as e:
        return {
            "error": "An unexpected error occurred",
            "message": str(e)
        }