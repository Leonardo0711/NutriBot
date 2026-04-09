import paramiko
import sys

VPS_IP = "187.77.19.172"
VPS_USER = "root"
VPS_PASS = "Essalud2026#"

def run_db_query(sql):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(VPS_IP, username=VPS_USER, password=VPS_PASS)
    
    # Proper escaping for the command
    command = f"docker exec -i nutribot-evolution-api-db psql -U postgres -d nutribot_db -c \"{sql}\""
    stdin, stdout, stderr = ssh.exec_command(command)
    
    out = stdout.read().decode()
    err = stderr.read().decode()
    
    ssh.close()
    return out, err

if __name__ == "__main__":
    # Check profile for the user in the screenshot
    sql = "SELECT p.* FROM perfil_nutricional p JOIN usuarios u ON p.usuario_id = u.id WHERE u.phone = '+51915107251'"
    out, err = run_db_query(sql)
    print("--- OUTPUT ---")
    print(out)
    print("--- ERRORS ---")
    print(err)
