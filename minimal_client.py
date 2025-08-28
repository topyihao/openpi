#!/usr/bin/env python3
"""
Minimal example of connecting to OpenPI policy server and getting a prediction.

This is the simplest possible client that demonstrates the core functionality.
"""

import numpy as np
from openpi_client import websocket_client_policy


def main():
    """Connect to policy server and get one prediction."""
    
    # Connect to the server (make sure serve_policy.py is running)
    policy = websocket_client_policy.WebsocketClientPolicy(
        host="localhost", 
        port=8000
    )
    
    print(f"Connected to server. Metadata: {policy.get_server_metadata()}")
    
    # Create a sample observation (ALOHA format)
    observation = {
        "state": np.ones((14,), dtype=np.float32),
        "images": {
            # ALOHA: Uses channels-first format (C, H, W)
            "cam_high": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "pick up the red block"
    }
    
    # Get prediction
    result = policy.infer(observation)
    
    # Print results
    actions = result["actions"]
    print(f"Got actions with shape: {actions.shape}")
    print(f"First action: {actions[0]}")
    print(f"Inference took: {result.get('server_timing', {}).get('infer_ms', 'N/A')} ms")


if __name__ == "__main__":
    main()
