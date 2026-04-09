import paramiko

VPS_IP = "187.77.19.172"
VPS_USER = "root"
VPS_PASS = "Essalud2026#"

def run_reset():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(VPS_IP, username=VPS_USER, password=VPS_PASS)
    
    # 1. Find the DB container name
    stdin, stdout, stderr = ssh.exec_command('docker ps --format "{{.Names}}"')
    containers = stdout.read().decode().splitlines()
    db_container = next((c for c in containers if "db" in c.lower()), None)
    
    if not db_container:
        print("ERROR: Could not find DB container.")
        ssh.close()
        return

    print(f"Found DB container: {db_container}")

    # List of tables to truncate in nutribot_db
    tables = [
        "usuarios",
        "incoming_messages",
        "outgoing_messages",
        "extraction_jobs",
        "n8n_log",
        "formulario_en_progreso",
        "perfil_nutricional",
        "conversation_state",
        "profile_extractions",
        "memoria_chat",
        "respuestas_formulario",
        "evaluacion_usabilidad"
    ]

    for table in tables:
        # Check if table exists
        check_cmd = f"docker exec -i {db_container} psql -U postgres -d nutribot_db -t -c \"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table}')\""
        stdin, stdout, stderr = ssh.exec_command(check_cmd)
        exists = stdout.read().decode().strip() == 't'
        
        if exists:
            print(f"Truncating {table}...")
            # Use CASCADE only for the main table to be safe, or just run it on all
            truncate_cmd = f"docker exec -i {db_container} psql -U postgres -d nutribot_db -c \"TRUNCATE TABLE {table} CASCADE;\""
            stdin, stdout, stderr = ssh.exec_command(truncate_cmd)
            print(stdout.read().decode().strip())
        else:
            print(f"Table {table} does not exist, skipping.")

    ssh.close()
    print("Reset finished.")

if __name__ == "__main__":
    run_reset()
