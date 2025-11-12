from flask import Flask, request, jsonify
import subprocess
import json

app = Flask(__name__)

ALLOWED_COMMANDS = {
    'list_instances': ['gcloud', 'compute', 'instances', 'list', '--format=json'],
    'list_zones': ['gcloud', 'compute', 'zones', 'list', '--format=json'],
    'list_regions': ['gcloud', 'compute', 'regions', 'list', '--format=json'],
    'list_machine_types': ['gcloud', 'compute', 'machine-types', 'list', '--format=json'],
    'describe_project': ['gcloud', 'compute', 'project-info', 'describe', '--format=json']
}


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route('/execute', methods=['POST'])
def execute_command():
    try:
        data = request.get_json()
        command_name = data.get('command')

        if command_name not in ALLOWED_COMMANDS:
            return jsonify({"error": "Command not allowed"}), 403

        command = ALLOWED_COMMANDS[command_name]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            try:
                output = json.loads(result.stdout)
            except json.JSONDecodeError:
                output = result.stdout
            return jsonify({"output": output}), 200
        else:
            return jsonify({"error": result.stderr}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
