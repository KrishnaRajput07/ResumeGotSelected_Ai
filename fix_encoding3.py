"""
Fix broken emoji/special chars by doing direct byte-level search-and-replace.
The file has UTF-8 bytes stored as individual Unicode codepoints (mojibake).
We fix by re-encoding the whole file: latin-1 decode (byte-preserving) -> UTF-8 decode.
"""
import shutil

src = 'app/main.py'
backup = 'app/main.py.bak_clean'

# Read raw bytes
with open(src, 'rb') as f:
    raw = f.read()

print(f'Original file: {len(raw)} bytes')

# KEY INSIGHT: The file was written with UTF-8 bytes but the Python open() call
# that wrote the file used a non-UTF-8 locale. So each UTF-8 byte became a separate
# character in the string, and was then written back as latin-1.
# FIX: treat raw bytes as latin-1 string, then re-decode as UTF-8.
try:
    # This is the canonical way to fix this type of mojibake
    content_fixed = raw.decode('latin-1').encode('latin-1').decode('utf-8', errors='replace')
    print(f'Fixed char count: {len(content_fixed)}')
except Exception as e:
    print(f'Error: {e}')

# Verify - check pages list
idx = content_fixed.find('pages = [')
print('\nPages list:')
print(repr(content_fixed[idx:idx+200]))
