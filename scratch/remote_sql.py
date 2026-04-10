
import paramiko
import sys

VPS_IP = "187.77.19.172"
VPS_USER = "root"
VPS_PASS = "Essalud2026#"

def run_query(query):
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

    cmd = f"docker exec -i {db_container} psql -U postgres -d nutribot_db -c \"{query}\""
    stdin, stdout, stderr = ssh.exec_command(cmd)
    print(stdout.read().decode())
    print(stderr.read().decode())
    ssh.close()

if __name__ == "__main__":
    run_query(sys.argv[1])
