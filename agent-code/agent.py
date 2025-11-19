"""
GCP Agent Application with vLLM Integration
This agent uses OpenAI-compatible API to interact with vLLM
and implements custom tools
"""

import os
import json
import requests
import uuid
import time
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
    """Get list of Google Cloud compute instances with detailed information."""
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
            result += f"  CPU Platform: {instance.get('cpu_platform', 'N/A')}\n"
            if instance.get('internal_ip'):
                result += f"  Internal IP: {instance['internal_ip']}\n"
            if instance.get('external_ip'):
                result += f"  External IP: {instance['external_ip']}\n"
            if instance.get('disks'):
                result += f"  Disks: {len(instance['disks'])} attached\n"
                for disk in instance['disks']:
                    result += f"    - {disk['name']} (Boot: {disk['boot']})\n"
            result += "\n"

        return result
    except Exception as e:
        logger.error(f"Error fetching GCP instances: {e}", exc_info=True)
        return f"Error fetching GCP instances: {str(e)}"

def list_gcp_disks() -> str:
    """List all disks in the GCP project."""
    try:
        response = requests.get(f"{GCP_HELPER_URL}/api/compute/disks", timeout=10)
        response.raise_for_status()
        data = response.json()

        disks = data.get('disks', [])
        count = data.get('count', 0)
        total_size = data.get('total_size_gb', 0)
        project = data.get('project_id', 'unknown')

        if count == 0:
            return f"No disks found in project {project}."

        result = f"Found {count} disk(s) in project {project} (Total: {total_size} GB):\n\n"
        for disk in disks:
            result += f"- Name: {disk['name']}\n"
            result += f"  Zone: {disk['zone']}\n"
            result += f"  Size: {disk['size_gb']} GB\n"
            result += f"  Type: {disk['type']}\n"
            result += f"  Status: {disk['status']}\n"
            if disk.get('users'):
                result += f"  Attached to: {', '.join(disk['users'])}\n"
            result += "\n"

        return result
    except Exception as e:
        logger.error(f"Error fetching GCP disks: {e}", exc_info=True)
        return f"Error fetching GCP disks: {str(e)}"

def estimate_gcp_cost() -> str:
    """Estimate monthly cost for GCP compute resources."""
    try:
        response = requests.get(f"{GCP_HELPER_URL}/api/compute/cost-estimate", timeout=10)
        response.raise_for_status()
        data = response.json()

        project = data.get('project_id', 'unknown')
        compute_cost = data.get('total_compute_cost_usd', 0)
        disk_cost = data.get('estimated_disk_cost_usd', 0)
        total_cost = data.get('estimated_total_monthly_cost_usd', 0)
        instances = data.get('compute_instances', [])

        result = f"Cost Estimate for project {project}:\n\n"
        result += f"Running Compute Instances:\n"
        for inst in instances:
            result += f"  - {inst['name']} ({inst['machine_type']}): ${inst['monthly_cost_usd']}/month\n"

        result += f"\nTotal Compute Cost: ${compute_cost}/month\n"
        result += f"Total Disk Cost: ${disk_cost}/month ({data.get('total_disk_gb', 0)} GB)\n"
        result += f"Estimated Total: ${total_cost}/month\n\n"
        result += f"Note: {data.get('note', 'These are estimates')}\n"

        return result
    except Exception as e:
        logger.error(f"Error estimating GCP cost: {e}", exc_info=True)
        return f"Error estimating GCP cost: {str(e)}"

def list_gcp_buckets() -> str:
    """List all storage buckets in the GCP project."""
    try:
        response = requests.get(f"{GCP_HELPER_URL}/api/storage/buckets", timeout=10)
        response.raise_for_status()
        data = response.json()

        buckets = data.get('buckets', [])
        count = data.get('count', 0)
        project = data.get('project_id', 'unknown')

        if count == 0:
            return f"No storage buckets found in project {project}."

        total_size = data.get('total_size_gb', 0)
        total_cost = data.get('total_monthly_cost_usd', 0)

        result = f"Found {count} bucket(s):\n"
        for i, bucket in enumerate(buckets, 1):
            size = bucket.get('size_gb', 0)
            cost = bucket.get('monthly_cost_usd', 0)
            result += f"{i}. {bucket['name']}: {size}GB, ${cost}/mo\n"

        result += f"\nTotal: {total_size}GB, ${total_cost}/month"

        return result
    except Exception as e:
        logger.error(f"Error fetching GCP buckets: {e}", exc_info=True)
        return f"Error fetching GCP buckets: {str(e)}"

def create_gcp_bucket(name: str, location: str = None, storage_class: str = None) -> str:
    """Create a new GCP storage bucket."""
    try:
        if not name:
            return "Error: Bucket name is required."

        if not location or not storage_class:
            available_locations = "US, EU, ASIA, us-central1, europe-west1"
            available_classes = "STANDARD, NEARLINE, COLDLINE, ARCHIVE"
            return f"To create bucket '{name}', please specify:\n- Location (options: {available_locations})\n- Storage class (options: {available_classes})\n\nOr say 'use defaults' for STANDARD class in US."

        payload = {
            "name": name,
            "location": location,
            "storage_class": storage_class
        }
        response = requests.post(f"{GCP_HELPER_URL}/api/storage/buckets/create", json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        result = f"✅ Bucket created successfully:\n"
        result += f"  Name: {data.get('name')}\n"
        result += f"  Location: {data.get('location')}\n"
        result += f"  Storage Class: {data.get('storage_class')}\n"

        return result
    except Exception as e:
        logger.error(f"Error creating GCP bucket: {e}", exc_info=True)
        return f"Error creating GCP bucket: {str(e)}"

def create_gcp_instance(name: str, zone: str = None, machine_type: str = None) -> str:
    """Create a new GCP compute instance."""
    try:
        # If zone or machine_type not provided, return a message asking for them
        if not name:
            return "Error: Instance name is required. Please provide a name for the VM."

        if not zone or not machine_type:
            available_zones = "us-central1-a, us-east1-b, europe-west1-b"
            available_types = "e2-micro (cheapest), e2-small, e2-medium, e2-standard-2, e2-standard-4"
            return f"To create instance '{name}', please specify:\n- Zone (options: {available_zones})\n- Machine type (options: {available_types})\n\nOr say 'use defaults' for e2-micro in us-central1-a."

        payload = {
            "name": name,
            "zone": zone,
            "machine_type": machine_type,
            "boot_disk_size_gb": 10
        }
        response = requests.post(f"{GCP_HELPER_URL}/api/compute/instances/create", json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        result = f"✅ Instance creation initiated:\n"
        result += f"  Name: {name}\n"
        result += f"  Zone: {zone}\n"
        result += f"  Machine Type: {machine_type}\n"
        result += f"  Operation: {data.get('operation', 'N/A')}\n"
        result += f"\nThe instance is being created and will be available shortly."

        return result
    except Exception as e:
        logger.error(f"Error creating GCP instance: {e}", exc_info=True)
        return f"Error creating GCP instance: {str(e)}"

# Define tools for function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_gcp_instances",
            "description": "Get list of Google Cloud compute instances with detailed information including disks, IPs, and machine types. Use when asked about VMs, instances, or compute resources.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_gcp_disks",
            "description": "List all disks in the GCP project with details like size, type, and which instances they're attached to. Use when asked about disks or storage.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_gcp_cost",
            "description": "Estimate monthly costs for running GCP compute resources including instances and disks. Use when asked about costs, pricing, or spending.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_gcp_buckets",
            "description": "List all storage buckets in the GCP project. Use when asked about buckets or cloud storage.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_gcp_bucket",
            "description": "Create a new GCP storage bucket. Use when asked to create a bucket.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the bucket (globally unique, lowercase, hyphens allowed)"
                    },
                    "location": {
                        "type": "string",
                        "description": "Bucket location (e.g., US, EU, ASIA, us-central1)",
                        "default": "US"
                    },
                    "storage_class": {
                        "type": "string",
                        "description": "Storage class (STANDARD, NEARLINE, COLDLINE, ARCHIVE)",
                        "default": "STANDARD"
                    }
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_gcp_instance",
            "description": "Create a new GCP compute instance. Use when asked to create, launch, or start a new VM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the instance to create (lowercase, hyphens allowed)"
                    },
                    "zone": {
                        "type": "string",
                        "description": "GCP zone (e.g., us-central1-a, europe-west1-b)",
                        "default": "us-central1-a"
                    },
                    "machine_type": {
                        "type": "string",
                        "description": "Machine type (e.g., e2-micro, e2-small, e2-medium, e2-standard-2)",
                        "default": "e2-micro"
                    }
                },
                "required": ["name"]
            }
        }
    }
]

def execute_tool(tool_name: str, arguments: dict) -> str:
    """Execute the requested tool with given arguments."""
    if tool_name == "get_gcp_instances":
        return get_gcp_instances()
    elif tool_name == "list_gcp_disks":
        return list_gcp_disks()
    elif tool_name == "list_gcp_buckets":
        return list_gcp_buckets()
    elif tool_name == "estimate_gcp_cost":
        return estimate_gcp_cost()
    elif tool_name == "create_gcp_bucket":
        name = arguments.get('name')
        location = arguments.get('location', 'US')
        storage_class = arguments.get('storage_class', 'STANDARD')
        return create_gcp_bucket(name, location, storage_class)
    elif tool_name == "create_gcp_instance":
        name = arguments.get('name')
        zone = arguments.get('zone', 'us-central1-a')
        machine_type = arguments.get('machine_type', 'e2-micro')
        return create_gcp_instance(name, zone, machine_type)
    else:
        available_tools = "get_gcp_instances, list_gcp_disks, list_gcp_buckets, estimate_gcp_cost, create_gcp_bucket, create_gcp_instance"
        return f"I apologize, but I don't have the capability to perform '{tool_name}'. Available tools: {available_tools}"

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
            {"role": "system", "content": """You are a helpful AI assistant for managing Google Cloud Platform resources.

CRITICAL: Questions about YOUR capabilities or what you CAN do should be answered directly WITHOUT calling any tools!

CAPABILITIES:
- View compute instances (VMs) and their details
- List disks and storage information
- View storage buckets
- Estimate monthly costs for compute resources
- Create new compute instances and storage buckets

TOOL USAGE RULES - READ CAREFULLY:

DO NOT call tools for meta/capability questions:
- "What can you do?" or "What can you do for me?" → Answer directly with capabilities list above
- "What are your features?" → Answer directly with capabilities list
- "How do I...?" or "Can you help me with...?" → Provide guidance, no tools
- "Help" → Answer directly with capabilities list

ONLY call tools for specific data/action requests:
- "Show me my instances" or "List my VMs" → call get_gcp_instances
- "What instances do I have?" → call get_gcp_instances
- "List my disks" → call list_gcp_disks
- "Show buckets" → call list_gcp_buckets
- "What's my cost?" or "Estimate costs" → call estimate_gcp_cost
- "Create instance named X" → call create_gcp_instance (only with name + zone + machine_type OR 'defaults')
- "Create bucket named Y" → call create_gcp_bucket

OTHER RULES:
1. ALWAYS include tool results verbatim in your response - NEVER summarize
2. When tool returns data, show it immediately - don't ask questions about it
3. Follow-up questions about already-retrieved data → Use conversation history, don't call tools again

Available tools: get_gcp_instances, list_gcp_disks, list_gcp_buckets, estimate_gcp_cost, create_gcp_bucket, create_gcp_instance"""}
        ]

        # Add conversation history (limit to last 10 messages to avoid context overflow)
        messages.extend(conversation_histories[session_id][-10:])

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # Call LLM with tool support
        start_time = time.time()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,  # Zero temperature for maximum accuracy and no hallucinations
            max_tokens=1024
        )
        initial_llm_time = time.time() - start_time
        logger.info(f"⏱️ Initial LLM call took {initial_llm_time:.2f}s")

        response_message = response.choices[0].message

        # Check if the model wants to call a tool
        if response_message.tool_calls:
            # Send reasoning indicator to show tool is being called
            def generate_with_tool_call():
                # Show reasoning indicator
                tool_names = [tc.function.name for tc in response_message.tool_calls]
                reasoning_msg = f"🔧 Calling tool: {', '.join(tool_names)}"
                yield f"data: {json.dumps({'reasoning': reasoning_msg})}\n\n"

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

                # Execute all tool calls
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    logger.info(f"Calling tool: {function_name} with args: {function_args}")

                    # Execute the tool
                    tool_start = time.time()
                    tool_result = execute_tool(function_name, function_args)
                    tool_time = time.time() - tool_start
                    logger.info(f"⏱️ Tool execution took {tool_time:.2f}s")

                    logger.info(f"Tool result: {tool_result[:200]}...")  # Log first 200 chars

                    # Add tool response to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })

                # Get final response from the model with streaming (optimized settings)
                final_llm_start = time.time()
                final_response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=512,  # Increased to handle longer lists
                    stream=True
                )
                logger.info(f"⏱️ Final LLM streaming started")

                # Stream the response
                full_response = ""
                first_token = True
                for chunk in final_response:
                    if chunk.choices[0].delta.content:
                        if first_token:
                            ttft = time.time() - final_llm_start
                            logger.info(f"⏱️ Time to first token: {ttft:.2f}s")
                            first_token = False
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield f"data: {json.dumps({'content': content})}\n\n"

                total_stream_time = time.time() - final_llm_start
                logger.info(f"⏱️ Total streaming time: {total_stream_time:.2f}s")

                # Save conversation to history
                conversation_histories[session_id].append({"role": "user", "content": user_message})
                conversation_histories[session_id].append({"role": "assistant", "content": full_response})

                # Limit history size
                if len(conversation_histories[session_id]) > 20:
                    conversation_histories[session_id] = conversation_histories[session_id][-20:]

                logger.info(f"Response: {full_response}")
                yield f"data: {json.dumps({'done': True})}\n\n"

            return Response(stream_with_context(generate_with_tool_call()), mimetype='text/event-stream')
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
                # Show reasoning indicator for direct response (no tools)
                yield f"data: {json.dumps({'reasoning': '💭 Responding directly (no tools needed)'})}\n\n"

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
