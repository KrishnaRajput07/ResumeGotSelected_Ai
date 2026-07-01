"""
Final, correct mojibake fix for main.py.
Read as latin-1 (byte-preserving), encode back as latin-1 bytes, decode as UTF-8.
Then strip any remaining bad control chars.
"""
import shutil

src = 'app/main.py'
backup = 'app/main.py.bak_clean'

with open(src, 'rb') as f:
    raw = f.read()

print(f'Original: {len(raw)} bytes')

# The canonical mojibake fix: latin-1 round-trip then UTF-8 decode
# errors='replace' will replace the one stray 0x8F byte with the replacement char
content_fixed = raw.decode('latin-1').encode('latin-1').decode('utf-8', errors='replace')
print(f'After decode: {len(content_fixed)} chars')

# Count replacement chars (bad bytes)
bad_count = content_fixed.count('\ufffd')
print(f'Bad bytes replaced: {bad_count}')

# Remove the replacement character (they're just variation selectors / stray control bytes)
# For the gear emoji case: ⚙\ufffd -> ⚙️ (the \ufffd was the VS variation selector 0x8F)
# Actually we want ⚙️ (gear + variation selector U+FE0F), but the VS byte 0x8F is invalid
# The gear ⚙ without VS is fine - just remove the replacement char
content_fixed = content_fixed.replace('\ufffd', '')
print(f'After cleanup: {len(content_fixed)} chars')

# Verify key areas
import sys
sys.stdout.reconfigure(encoding='utf-8')

idx = content_fixed.find('pages = [')
print('\nPages list:')
print(content_fixed[idx:idx+250])

idx2 = content_fixed.find('──')
if idx2 >= 0:
    print('\nDash sample:')
    print(repr(content_fixed[idx2-10:idx2+30]))

# Backup original
shutil.copy2(src, backup)
print(f'\nBacked up to {backup}')

# Write fixed file
with open(src, 'w', encoding='utf-8', newline='') as f:
    f.write(content_fixed)

# Verify syntax
import ast
with open(src, 'r', encoding='utf-8') as f:
    check = f.read()
try:
    ast.parse(check)
    print('SYNTAX OK ✓')
except SyntaxError as e:
    print(f'SYNTAX ERROR at line {e.lineno}: {e.msg}')

print(f'Final file: {len(check)} chars written.')
