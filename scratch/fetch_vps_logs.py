import paramiko
import os

VPS_IP = "187.77.19.172"
VPS_USER = "root"
VPS_PASS = "Essalud2026#"

def get_vps_logs():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(VPS_IP, username=VPS_USER, password=VPS_PASS)
    
    # Get last 200 lines of logs
    stdin, stdout, stderr = ssh.exec_command("docker logs nutribot-backend --tail 200")
    logs = stdout.read().decode()
    err = stderr.read().decode()
    
    print("--- LOGS ---")
    print(logs)
    print("--- ERRORS ---")
    print(err)
    
    ssh.close()

if __name__ == "__main__":
    import sys
    # Force utf-8 for stdout
    sys.stdout.reconfigure(encoding='utf-8')
    get_vps_logs()

