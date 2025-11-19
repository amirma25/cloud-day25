"""
GCP Helper Service
Provides REST API endpoints to query GCP resources using Workload Identity
"""

import os
import logging
from flask import Flask, jsonify, request
from google.cloud import compute_v1
from google.cloud import storage
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

                    # Get disk information
                    disks_info = []
                    for disk in instance.disks:
                        disks_info.append({
                            'name': disk.source.split('/')[-1] if disk.source else 'unknown',
                            'boot': disk.boot,
                            'size_gb': disk.disk_size_gb if hasattr(disk, 'disk_size_gb') else 'N/A'
                        })

                    instances.append({
                        'name': instance.name,
                        'zone': zone_name,
                        'machine_type': instance.machine_type.split('/')[-1] if instance.machine_type else 'unknown',
                        'status': instance.status,
                        'internal_ip': instance.network_interfaces[0].network_i_p if instance.network_interfaces else None,
                        'external_ip': instance.network_interfaces[0].access_configs[0].nat_i_p if instance.network_interfaces and instance.network_interfaces[0].access_configs else None,
                        'creation_timestamp': instance.creation_timestamp,
                        'cpu_platform': instance.cpu_platform if hasattr(instance, 'cpu_platform') else 'N/A',
                        'disks': disks_info,
                        'tags': list(instance.tags.items) if instance.tags and hasattr(instance.tags, 'items') else []
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

@app.route('/api/compute/disks', methods=['GET'])
def list_disks():
    """List all disks in the project."""
    try:
        project_id = request.args.get('project_id', PROJECT_ID)

        if not project_id:
            return jsonify({'error': 'Project ID not configured'}), 500

        logger.info(f"Listing disks for project: {project_id}")

        # Initialize disk client
        disk_client = compute_v1.DisksClient()

        # Aggregate request to list all disks across all zones
        aggregated_list = disk_client.aggregated_list(project=project_id)

        disks = []
        for zone, response in aggregated_list:
            if response.disks:
                for disk in response.disks:
                    zone_name = zone.split('/')[-1] if '/' in zone else zone

                    # Calculate size in GB
                    size_gb = int(disk.size_gb) if hasattr(disk, 'size_gb') else 0

                    disks.append({
                        'name': disk.name,
                        'zone': zone_name,
                        'size_gb': size_gb,
                        'type': disk.type.split('/')[-1] if disk.type else 'unknown',
                        'status': disk.status,
                        'users': [user.split('/')[-1] for user in disk.users] if disk.users else [],
                        'creation_timestamp': disk.creation_timestamp
                    })

        logger.info(f"Found {len(disks)} disks")

        return jsonify({
            'project_id': project_id,
            'disks': disks,
            'count': len(disks),
            'total_size_gb': sum(d['size_gb'] for d in disks)
        }), 200

    except Exception as e:
        logger.error(f"Error listing disks: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/compute/instances/create', methods=['POST'])
def create_instance():
    """Create a new compute instance."""
    try:
        project_id = request.json.get('project_id', PROJECT_ID)
        zone = request.json.get('zone', 'us-central1-a')
        instance_name = request.json.get('name')
        machine_type = request.json.get('machine_type', 'e2-micro')
        boot_disk_size_gb = request.json.get('boot_disk_size_gb', 10)

        if not project_id or not instance_name:
            return jsonify({'error': 'project_id and name are required'}), 400

        logger.info(f"Creating instance {instance_name} in zone {zone}")

        # Initialize compute client
        instance_client = compute_v1.InstancesClient()

        # Create instance configuration
        instance = compute_v1.Instance()
        instance.name = instance_name
        instance.machine_type = f"zones/{zone}/machineTypes/{machine_type}"

        # Boot disk
        disk = compute_v1.AttachedDisk()
        disk.boot = True
        disk.auto_delete = True
        initialize_params = compute_v1.AttachedDiskInitializeParams()
        initialize_params.source_image = "projects/debian-cloud/global/images/family/debian-11"
        initialize_params.disk_size_gb = boot_disk_size_gb
        disk.initialize_params = initialize_params
        instance.disks = [disk]

        # Network interface
        network_interface = compute_v1.NetworkInterface()
        network_interface.name = "global/networks/default"
        access_config = compute_v1.AccessConfig()
        access_config.name = "External NAT"
        access_config.type_ = "ONE_TO_ONE_NAT"
        network_interface.access_configs = [access_config]
        instance.network_interfaces = [network_interface]

        # Create the instance
        operation = instance_client.insert(
            project=project_id,
            zone=zone,
            instance_resource=instance
        )

        logger.info(f"Instance creation initiated: {operation.name}")

        return jsonify({
            'message': f'Instance {instance_name} creation started',
            'operation': operation.name,
            'zone': zone,
            'machine_type': machine_type
        }), 202

    except Exception as e:
        logger.error(f"Error creating instance: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/compute/cost-estimate', methods=['GET'])
def estimate_cost():
    """Estimate monthly cost for compute resources."""
    try:
        project_id = request.args.get('project_id', PROJECT_ID)

        if not project_id:
            return jsonify({'error': 'Project ID not configured'}), 500

        logger.info(f"Estimating costs for project: {project_id}")

        # Get all running instances
        instance_client = compute_v1.InstancesClient()
        aggregated_list = instance_client.aggregated_list(project=project_id)

        # Approximate pricing (USD per month, 730 hours)
        machine_type_pricing = {
            'e2-micro': 6.11,
            'e2-small': 12.23,
            'e2-medium': 24.45,
            'e2-standard-2': 48.91,
            'e2-standard-4': 97.81,
            'e2-standard-8': 195.62,
            'n1-standard-1': 24.27,
            'n1-standard-2': 48.54,
            'n1-standard-4': 97.09,
            'n2-standard-2': 64.54,
            'n2-standard-4': 129.08,
        }

        total_cost = 0
        instance_costs = []

        for zone, response in aggregated_list:
            if response.instances:
                for instance in response.instances:
                    if instance.status == 'RUNNING':
                        machine_type = instance.machine_type.split('/')[-1]
                        monthly_cost = machine_type_pricing.get(machine_type, 50.0)  # Default estimate
                        total_cost += monthly_cost

                        instance_costs.append({
                            'name': instance.name,
                            'machine_type': machine_type,
                            'monthly_cost_usd': round(monthly_cost, 2),
                            'status': instance.status
                        })

        # Get disk costs (approximately $0.04 per GB per month for standard persistent disk)
        disk_client = compute_v1.DisksClient()
        disk_list = disk_client.aggregated_list(project=project_id)

        total_disk_gb = 0
        for zone, response in disk_list:
            if response.disks:
                for disk in response.disks:
                    total_disk_gb += int(disk.size_gb) if hasattr(disk, 'size_gb') else 0

        disk_cost = total_disk_gb * 0.04

        return jsonify({
            'project_id': project_id,
            'compute_instances': instance_costs,
            'total_compute_cost_usd': round(total_cost, 2),
            'total_disk_gb': total_disk_gb,
            'estimated_disk_cost_usd': round(disk_cost, 2),
            'estimated_total_monthly_cost_usd': round(total_cost + disk_cost, 2),
            'note': 'These are approximate estimates. Check GCP billing for exact costs.'
        }), 200

    except Exception as e:
        logger.error(f"Error estimating cost: {e}", exc_info=True)
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

@app.route('/api/storage/buckets', methods=['GET'])
def list_buckets():
    """List all storage buckets in the project."""
    try:
        project_id = request.args.get('project_id', PROJECT_ID)

        if not project_id:
            return jsonify({'error': 'Project ID not configured'}), 500

        logger.info(f"Listing buckets for project: {project_id}")

        # Initialize storage client
        storage_client = storage.Client(project=project_id)

        # List all buckets
        buckets = []
        total_size_gb = 0
        total_monthly_cost = 0

        # Storage pricing per GB per month (approximate)
        storage_pricing = {
            'STANDARD': 0.020,      # $0.020 per GB/month
            'NEARLINE': 0.010,      # $0.010 per GB/month
            'COLDLINE': 0.004,      # $0.004 per GB/month
            'ARCHIVE': 0.0012,      # $0.0012 per GB/month
        }

        for bucket in storage_client.list_buckets():
            # Calculate bucket size
            size_bytes = 0
            try:
                blobs = bucket.list_blobs()
                for blob in blobs:
                    size_bytes += blob.size if blob.size else 0
            except Exception as e:
                logger.warning(f"Could not calculate size for bucket {bucket.name}: {e}")

            size_gb = size_bytes / (1024**3)  # Convert to GB
            total_size_gb += size_gb

            # Calculate monthly cost
            price_per_gb = storage_pricing.get(bucket.storage_class, 0.020)
            monthly_cost = size_gb * price_per_gb
            total_monthly_cost += monthly_cost

            buckets.append({
                'name': bucket.name,
                'location': bucket.location,
                'storage_class': bucket.storage_class,
                'created': bucket.time_created.isoformat() if bucket.time_created else None,
                'versioning_enabled': bucket.versioning_enabled,
                'size_gb': round(size_gb, 2),
                'monthly_cost_usd': round(monthly_cost, 2)
            })

        logger.info(f"Found {len(buckets)} buckets")

        return jsonify({
            'project_id': project_id,
            'buckets': buckets,
            'count': len(buckets),
            'total_size_gb': round(total_size_gb, 2),
            'total_monthly_cost_usd': round(total_monthly_cost, 2)
        }), 200

    except Exception as e:
        logger.error(f"Error listing buckets: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/storage/buckets/create', methods=['POST'])
def create_bucket():
    """Create a new storage bucket."""
    try:
        project_id = request.json.get('project_id', PROJECT_ID)
        bucket_name = request.json.get('name')
        location = request.json.get('location', 'US')
        storage_class = request.json.get('storage_class', 'STANDARD')

        if not project_id or not bucket_name:
            return jsonify({'error': 'project_id and name are required'}), 400

        logger.info(f"Creating bucket {bucket_name} in location {location}")

        # Initialize storage client
        storage_client = storage.Client(project=project_id)

        # Create bucket
        bucket = storage_client.bucket(bucket_name)
        bucket.storage_class = storage_class
        new_bucket = storage_client.create_bucket(bucket, location=location)

        logger.info(f"Bucket {bucket_name} created successfully")

        return jsonify({
            'message': f'Bucket {bucket_name} created successfully',
            'name': new_bucket.name,
            'location': new_bucket.location,
            'storage_class': new_bucket.storage_class
        }), 201

    except Exception as e:
        logger.error(f"Error creating bucket: {e}", exc_info=True)
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
