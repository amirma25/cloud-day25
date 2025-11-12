"""
GCP Helper Service
Provides REST API endpoints to query GCP resources using Workload Identity
"""

import os
import logging
from flask import Flask, jsonify, request
from google.cloud import compute_v1
from google.auth import default

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Get project ID from environment or metadata
PROJECT_ID = os.environ.get('GCP_PROJECT_ID', os.environ.get('GOOGLE_CLOUD_PROJECT'))

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200

@app.route('/api/compute/instances', methods=['GET'])
def list_instances():
    """List all compute instances in the project."""
    try:
        # Get project ID from query param or use default
        project_id = request.args.get('project_id', PROJECT_ID)

        if not project_id:
            return jsonify({'error': 'Project ID not configured'}), 500

        logger.info(f"Listing instances for project: {project_id}")

        # Initialize compute client
        instance_client = compute_v1.InstancesClient()

        # Aggregate request to list all instances across all zones
        aggregated_list = instance_client.aggregated_list(project=project_id)

        instances = []
        for zone, response in aggregated_list:
            if response.instances:
                for instance in response.instances:
                    # Extract zone name from zone URL
                    zone_name = zone.split('/')[-1] if '/' in zone else zone

                    instances.append({
                        'name': instance.name,
                        'zone': zone_name,
                        'machine_type': instance.machine_type.split('/')[-1] if instance.machine_type else 'unknown',
                        'status': instance.status,
                        'internal_ip': instance.network_interfaces[0].network_i_p if instance.network_interfaces else None,
                        'external_ip': instance.network_interfaces[0].access_configs[0].nat_i_p if instance.network_interfaces and instance.network_interfaces[0].access_configs else None,
                        'creation_timestamp': instance.creation_timestamp
                    })

        logger.info(f"Found {len(instances)} instances")

        return jsonify({
            'project_id': project_id,
            'instances': instances,
            'count': len(instances)
        }), 200

    except Exception as e:
        logger.error(f"Error listing instances: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/compute/instance/<zone>/<instance_name>', methods=['GET'])
def get_instance(zone, instance_name):
    """Get details of a specific compute instance."""
    try:
        project_id = request.args.get('project_id', PROJECT_ID)

        if not project_id:
            return jsonify({'error': 'Project ID not configured'}), 500

        logger.info(f"Getting instance {instance_name} in zone {zone}")

        # Initialize compute client
        instance_client = compute_v1.InstancesClient()

        # Get instance details
        instance = instance_client.get(
            project=project_id,
            zone=zone,
            instance=instance_name
        )

        instance_data = {
            'name': instance.name,
            'zone': zone,
            'machine_type': instance.machine_type.split('/')[-1],
            'status': instance.status,
            'internal_ip': instance.network_interfaces[0].network_i_p if instance.network_interfaces else None,
            'external_ip': instance.network_interfaces[0].access_configs[0].nat_i_p if instance.network_interfaces and instance.network_interfaces[0].access_configs else None,
            'creation_timestamp': instance.creation_timestamp,
            'cpu_platform': instance.cpu_platform,
            'disks': [{'name': disk.source.split('/')[-1], 'boot': disk.boot} for disk in instance.disks],
            'labels': instance.labels
        }

        return jsonify(instance_data), 200

    except Exception as e:
        logger.error(f"Error getting instance: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/gke/clusters', methods=['GET'])
def list_clusters():
    """List all GKE clusters in the project."""
    try:
        project_id = request.args.get('project_id', PROJECT_ID)

        if not project_id:
            return jsonify({'error': 'Project ID not configured'}), 500

        logger.info(f"Listing GKE clusters for project: {project_id}")

        # Use gcloud command as GKE API requires additional setup
        import subprocess
        result = subprocess.run(
            ['gcloud', 'container', 'clusters', 'list', '--project', project_id, '--format=json'],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return jsonify({'error': result.stderr}), 500

        import json
        clusters = json.loads(result.stdout)

        return jsonify({
            'project_id': project_id,
            'clusters': clusters,
            'count': len(clusters)
        }), 200

    except Exception as e:
        logger.error(f"Error listing clusters: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/info', methods=['GET'])
def get_project_info():
    """Get basic project information."""
    try:
        project_id = request.args.get('project_id', PROJECT_ID)

        if not project_id:
            return jsonify({'error': 'Project ID not configured'}), 500

        # Get credentials
        credentials, project = default()

        return jsonify({
            'project_id': project_id,
            'detected_project': project,
            'service_account': os.environ.get('GOOGLE_SERVICE_ACCOUNT', 'default')
        }), 200

    except Exception as e:
        logger.error(f"Error getting project info: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not PROJECT_ID:
        logger.warning("PROJECT_ID not set, attempting to detect from environment")
        try:
            _, PROJECT_ID = default()
            logger.info(f"Detected project: {PROJECT_ID}")
        except Exception as e:
            logger.error(f"Could not detect project ID: {e}")

    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
