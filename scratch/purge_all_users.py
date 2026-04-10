
import paramiko
import sys

VPS_IP = "187.77.19.172"
VPS_USER = "root"
VPS_PASS = "Essalud2026#"

TABLES_TO_PURGE = [
    "usuarios",
    "perfil_nutricional",
    "memoria_chat",
    "conversation_state",
    "formulario_en_progreso",
    "incoming_messages",
    "outgoing_messages",
    "extraction_jobs",
    "profile_extractions",
    "evaluacion_usabilidad",
    "respuestas_formulario"
]

def purge_all():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(VPS_IP, username=VPS_USER, password=VPS_PASS)
    
    stdin, stdout, stderr = ssh.exec_command('docker ps --format "{{.Names}}"')
    containers = stdout.read().decode().splitlines()
    db_container = next((c for c in containers if "db" in c.lower()), None)
    
    if not db_container:
        print("ERROR: Could not find DB container.")
        ssh.close()
        return

    tables_str = ", ".join(TABLES_TO_PURGE)
    query = f"TRUNCATE TABLE {tables_str} RESTART IDENTITY CASCADE;"
    print(f"Executing: {query}")

    cmd = f"docker exec -i {db_container} psql -U postgres -d nutribot_db -c \"{query}\""
    stdin, stdout, stderr = ssh.exec_command(cmd)
    
    print("STDOUT:")
    print(stdout.read().decode())
    print("STDERR:")
    print(stderr.read().decode())
    
    # Verification
    check_query = "SELECT table_name, row_count FROM (SELECT table_name FROM information_schema.tables WHERE table_schema='public') t, LATERAL (SELECT count(*) as row_count FROM p_query_tab_count(t.table_name)) c;"
    # Actually, simpler verification: count users
    verify_cmd = f"docker exec -i {db_container} psql -U postgres -d nutribot_db -c \"SELECT count(*) FROM usuarios;\""
    stdin, stdout, stderr = ssh.exec_command(verify_cmd)
    print("Verification (Users Count):", stdout.read().decode().strip())

    ssh.close()

if __name__ == "__main__":
    purge_all()
