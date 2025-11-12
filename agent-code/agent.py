"""
ADK Agent Application with vLLM Integration
This agent uses OpenAI-compatible API to interact with vLLM
and implements custom tools for weather and calculations.
"""

import os
import json
import requests
import uuid
from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
from openai import OpenAI
from datetime import timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

# Store conversation histories per session
conversation_histories = {}

# Environment configuration
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://vllm-llama3-service:8000/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
GCP_HELPER_URL = os.environ.get("GCP_HELPER_URL", "http://gcp-helper-service:8080")

# Custom tool: Get GCP instances
def get_gcp_instances() -> str:
    """
    Get list of Google Cloud compute instances in the project.

    Returns:
        A formatted string with instance details including name, zone, status, and IPs
    """
    try:
        response = requests.get(f"{GCP_HELPER_URL}/api/compute/instances", timeout=10)
        response.raise_for_status()
        data = response.json()

        instances = data.get('instances', [])
        count = data.get('count', 0)
        project = data.get('project_id', 'unknown')

        if count == 0:
            return f"No compute instances found in project {project}."

        result = f"Found {count} compute instance(s) in project {project}:\n\n"
        for instance in instances:
            result += f"- Name: {instance['name']}\n"
            result += f"  Zone: {instance['zone']}\n"
            result += f"  Status: {instance['status']}\n"
            result += f"  Machine Type: {instance['machine_type']}\n"
            if instance.get('internal_ip'):
                result += f"  Internal IP: {instance['internal_ip']}\n"
            if instance.get('external_ip'):
                result += f"  External IP: {instance['external_ip']}\n"
            result += "\n"

        return result
    except Exception as e:
        logger.error(f"Error fetching GCP instances: {e}", exc_info=True)
        return f"Error fetching GCP instances: {str(e)}"

# Define tools for function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_gcp_instances",
            "description": "Get list of Google Cloud compute instances in the current project. Use this when the user asks about GCP instances, VMs, or compute resources.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]

def execute_tool(tool_name: str, arguments: dict) -> str:
    """Execute the requested tool with given arguments."""
    if tool_name == "get_gcp_instances":
        return get_gcp_instances()
    else:
        return f"Unknown tool: {tool_name}"

@app.route('/')
def index():
    """Render the main chat interface."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests from the UI."""
    try:
        data = request.json
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Get or create session ID
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
            session.permanent = True

        session_id = session['session_id']

        # Initialize conversation history for this session if it doesn't exist
        if session_id not in conversation_histories:
            conversation_histories[session_id] = []

        logger.info(f"Session {session_id[:8]}: Received message: {user_message}")

        # Initialize OpenAI client for vLLM
        client = OpenAI(
            base_url=LLM_BASE_URL,
            api_key="EMPTY"  # vLLM doesn't require API key
        )

        # Build messages from conversation history
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant powered by Llama 3.1 8B Instruct with access to tools for querying GCP resources. When asked about yourself or what model you are, respond that you are Llama 3.1 8B Instruct. Only use tools when the user explicitly asks for specific GCP resource information. For greetings, casual conversation, or general questions about yourself, respond naturally without calling any tools. When you receive tool results, read them CAREFULLY and report the EXACT information provided - do not make assumptions or infer information that is not explicitly stated. Machine types like 'e2-standard-4', 'n1-standard-2', 'n2-standard-4', etc. are different and should be reported exactly as shown. If asked about a specific machine type (like N1, N2, E2), check the actual machine_type field and only confirm if it matches exactly."}
        ]

        # Add conversation history (limit to last 10 messages to avoid context overflow)
        messages.extend(conversation_histories[session_id][-10:])

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # Call LLM with tool support
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,  # Zero temperature for maximum accuracy and no hallucinations
            max_tokens=1024
        )

        response_message = response.choices[0].message

        # Check if the model wants to call a tool
        if response_message.tool_calls:
            # Execute tool calls
            # Convert response_message to dict format for messages
            messages.append({
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in response_message.tool_calls
                ]
            })

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                logger.info(f"Calling tool: {function_name} with args: {function_args}")

                # Execute the tool
                tool_result = execute_tool(function_name, function_args)

                logger.info(f"Tool result: {tool_result[:200]}...")  # Log first 200 chars

                # Add tool response to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })

            # Get final response from the model with streaming
            final_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,  # Zero temperature for maximum accuracy and no hallucinations
                max_tokens=1024,
                stream=True
            )

            # Stream the response
            def generate():
                full_response = ""
                for chunk in final_response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield f"data: {json.dumps({'content': content})}\n\n"

                # Save conversation to history after streaming completes
                conversation_histories[session_id].append({"role": "user", "content": user_message})
                conversation_histories[session_id].append({"role": "assistant", "content": full_response})

                # Limit history size
                if len(conversation_histories[session_id]) > 20:
                    conversation_histories[session_id] = conversation_histories[session_id][-20:]

                logger.info(f"Response: {full_response}")
                yield f"data: {json.dumps({'done': True})}\n\n"

            return Response(stream_with_context(generate()), mimetype='text/event-stream')
        else:
            # No tool calls - stream directly
            response_stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.0,
                max_tokens=1024,
                stream=True
            )

            def generate():
                full_response = ""
                for chunk in response_stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield f"data: {json.dumps({'content': content})}\n\n"

                # Save conversation to history
                conversation_histories[session_id].append({"role": "user", "content": user_message})
                conversation_histories[session_id].append({"role": "assistant", "content": full_response})

                # Limit history size
                if len(conversation_histories[session_id]) > 20:
                    conversation_histories[session_id] = conversation_histories[session_id][-20:]

                logger.info(f"Response: {full_response}")
                yield f"data: {json.dumps({'done': True})}\n\n"

            return Response(stream_with_context(generate()), mimetype='text/event-stream')

    except Exception as e:
        logger.error(f"Error processing chat: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/clear', methods=['POST'])
def clear_chat():
    """Clear conversation history for the current session."""
    try:
        if 'session_id' in session:
            session_id = session['session_id']
            if session_id in conversation_histories:
                conversation_histories[session_id] = []
                logger.info(f"Session {session_id[:8]}: Cleared conversation history")

        return jsonify({
            'status': 'success',
            'message': 'Conversation cleared'
        })
    except Exception as e:
        logger.error(f"Error clearing chat: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 8001))
    app.run(host='0.0.0.0', port=port, debug=False)
