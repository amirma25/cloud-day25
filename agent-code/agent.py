import os
import requests
import json
from typing import List, Dict, Any, Optional
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://vllm-llama3-service:8000/v1")
GCLOUD_HELPER_ENDPOINT = os.getenv("GCLOUD_HELPER_ENDPOINT", "http://gcloud-helper-service:8080")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_compute_instances",
            "description": "List all Google Cloud Compute Engine instances in the project",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_zones",
            "description": "List all available Google Cloud zones",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_regions",
            "description": "List all available Google Cloud regions",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_machine_types",
            "description": "List all available Google Cloud machine types",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

TOOL_COMMAND_MAPPING = {
    "list_compute_instances": "list_instances",
    "list_zones": "list_zones",
    "list_regions": "list_regions",
    "list_machine_types": "list_machine_types"
}


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Execute a tool by calling the gcloud-helper service."""
    try:
        command_name = TOOL_COMMAND_MAPPING.get(tool_name)
        if not command_name:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        response = requests.post(
            f"{GCLOUD_HELPER_ENDPOINT}/execute",
            json={"command": command_name},
            timeout=30
        )

        if response.status_code == 200:
            return json.dumps(response.json().get("output", {}))
        else:
            return json.dumps({"error": response.json().get("error", "Unknown error")})

    except Exception as e:
        return json.dumps({"error": str(e)})


def call_llm(messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """Call the vLLM endpoint with messages and optional tools."""
    try:
        payload = {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.7
        }

        if tools:
            payload["tools"] = tools

        response = requests.post(
            f"{VLLM_ENDPOINT}/chat/completions",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"LLM API error: {response.status_code} - {response.text}"}

    except Exception as e:
        return {"error": str(e)}


def run_agent(user_query: str, max_iterations: int = 5) -> Dict[str, Any]:
    """Run the agent loop to process user query."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can query Google Cloud resources. Use the available tools to answer user questions about their GCP infrastructure."
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    conversation_history = []

    for iteration in range(max_iterations):
        llm_response = call_llm(messages, tools=TOOLS)

        if "error" in llm_response:
            return {
                "success": False,
                "error": llm_response["error"],
                "conversation": conversation_history
            }

        choice = llm_response.get("choices", [{}])[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason")

        conversation_history.append({
            "iteration": iteration + 1,
            "message": message
        })

        if finish_reason == "tool_calls" and "tool_calls" in message:
            messages.append(message)

            for tool_call in message["tool_calls"]:
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"]) if tool_call["function"]["arguments"] else {}

                tool_result = execute_tool(function_name, function_args)

                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result
                }
                messages.append(tool_message)

                conversation_history.append({
                    "iteration": iteration + 1,
                    "tool_call": {
                        "name": function_name,
                        "arguments": function_args,
                        "result": tool_result
                    }
                })

        elif finish_reason == "stop":
            final_response = message.get("content", "")
            return {
                "success": True,
                "response": final_response,
                "conversation": conversation_history
            }

        else:
            messages.append(message)

    return {
        "success": False,
        "error": "Max iterations reached",
        "conversation": conversation_history
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    user_query = data.get('query', '')

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    result = run_agent(user_query)
    return jsonify(result)


@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
