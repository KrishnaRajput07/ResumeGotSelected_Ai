"""
Definitive mojibake fix:
The file has characters that were originally UTF-8 bytes,
written to a Windows CP1252 file, then re-read as UTF-8 (creating double-encoding).

Fix: 
1. Read current file as UTF-8 (get the mojibake chars)
2. Encode each char back to CP1252 byte (reverse the mis-read) 
3. Decode the resulting bytes as UTF-8 (get the real emoji)
"""
import sys
import shutil
sys.stdout.reconfigure(encoding='utf-8')

src = 'app/main.py'
backup = 'app/main.py.bak_orig'

# Step 1: read current (double-encoded) content
with open(src, 'r', encoding='utf-8', errors='replace') as f:
    mojibake_str = f.read()

print(f'Current file chars: {len(mojibake_str)}')

# Step 2: encode back to CP1252 bytes (reverse the Windows mis-read)
# errors='replace' handles chars outside CP1252 range
try:
    original_bytes = mojibake_str.encode('cp1252', errors='replace')
    print(f'Intermediate bytes: {len(original_bytes)}')
except Exception as e:
    print(f'Encode error: {e}')
    sys.exit(1)

# Step 3: decode original bytes as UTF-8
try:
    fixed_str = original_bytes.decode('utf-8', errors='replace')
    print(f'Fixed chars: {len(fixed_str)}')
    bad = fixed_str.count('\ufffd')
    print(f'Replacement chars: {bad}')
except Exception as e:
    print(f'Decode error: {e}')
    sys.exit(1)

# Verify: check the pages list
idx = fixed_str.find('pages = [')
print('\n=== Pages list ===')
print(fixed_str[idx:idx+250])

# Verify syntax
import ast
try:
    ast.parse(fixed_str)
    print('\nSYNTAX OK ✓')
except SyntaxError as e:
    print(f'\nSYNTAX ERROR at line {e.lineno}: {e.msg}')
    sys.exit(1)

# Clean up replacement chars
fixed_str = fixed_str.replace('\ufffd', '')

# Backup and write
shutil.copy2(src, backup)
print(f'Backed up to {backup}')

with open(src, 'w', encoding='utf-8', newline='') as f:
    f.write(fixed_str)

print(f'Fixed file written: {len(fixed_str)} chars')
