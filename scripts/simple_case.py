from src.clients import bedrock_client


def main():
    # Initialize the Bedrock client
    client = bedrock_client.Bedrock_Client(model_id='anthropic.claude-3-5-sonnet-20241022-v2:0')
    
    # Test with a simple text prompt
    print("Testing Bedrock client with text prompt...")
    response = client.invoke_model(prompt='Hello, how are you?')
    
    # Display the response
    print("\n=== Response ===")
    print(f"Text: {response['text']}")
    print(f"Input tokens: {response['input_tokens']}")
    print(f"Output tokens: {response['output_tokens']}")
    print(f"Total tokens: {response['total_tokens']}")
    print(f"Input cost: ${response['input_cost']:.4f}")
    print(f"Output cost: ${response['output_cost']:.4f}")
    print(f"Total cost: ${response['total_cost']:.4f}")


if __name__ == '__main__':
    main()
