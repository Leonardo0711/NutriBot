import os
import re

# Known mojibake mappings that commonly occur when UTF-8 gets misinterpreted as Windows-1252
MOJIBAKE_MAPPINGS = {
    'á': 'á', 'é': 'é', 'í': 'í', 'ó': 'ó', 'ú': 'ú',
    'ñ': 'ñ', 'Ñ': 'Ñ',
    'À': 'À', 'È': 'È', 'Ì': 'Ì', 'Ò': 'Ò', 'Ù': 'Ù',
    'ç': 'ç', 'Ç': 'Ç',
    'ü': 'ü', 'Ü': 'Ü',
    '¡': '¡', '¿': '¿',
    '°': '°', 'º': 'º', 'ª': 'ª',
    '“': '“', '—': '—',
    '🤔': '🤔', 'ðŸ™\x8f': '🙏', 'ðŸŽ™ï¸\x8f': '🎙️', '📸': '📸', '😊': '😊', '🥦': '🥦', 'ðŸ\x8d\x8e': '🍎', 'ðŸ\x92ª🏾': '💪🏾', '✨': '✨'
    # Feel free to expand with more exact byte sequences if needed
}

def repair_mojibake(content: str) -> tuple[str, int]:
    """Repairs mojibake based on precise known mappings. Returns (new_content, changes_count)."""
    changes = 0
    # First, let's try a bulk encode/decode trick if it's pure cp1252 mishandled UTF-8
    
    # Try custom replacement
    new_content = content
    for bad, good in MOJIBAKE_MAPPINGS.items():
        if bad in new_content:
            count = new_content.count(bad)
            changes += count
            new_content = new_content.replace(bad, good)
            
    # Sometimes emojis get even more mangled, we'll brute force decode if necessary.
    try:
        # If it was actually double encoded:
        re_encoded = content.encode('windows-1252').decode('utf-8')
        if any(c in re_encoded for c in ['á', 'é', 'í', 'ó', 'ú', 'ñ', '¿', '¡']):
            return re_encoded, 999 
    except Exception:
        pass

    return new_content, changes


def run(directory: str):
    print(f"Buscando mojibake en {directory}...")
    total_files_fixed = 0
    total_replacements = 0

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    print(f"[!] Archivo {path} no puede leerse como UTF-8 puro.")
                    continue

                new_content, changes = repair_mojibake(content)

                if changes > 0:
                    print(f"[+] Archivo reparado: {file} ({changes} cambios).")
                    
                    # Sanity check: Can we read it back as pure utf-8?
                    try:
                        new_content.encode('utf-8').decode('utf-8')
                        
                        # Escribir UTF-8 sin BOM
                        with open(path, 'w', encoding='utf-8', newline='\n') as f:
                            f.write(new_content)
                            
                        total_files_fixed += 1
                        total_replacements += changes
                    except Exception as e:
                        print(f"[-] FATAL: Error verificando reescritura de {file}: {e}")

    print("---")
    print(f"Mojibake solucionado. {total_files_fixed} archivos modificados.")
    print(f"{total_replacements} reemplazos totales.")

if __name__ == "__main__":
    run(".")
