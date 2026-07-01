"""
Proper byte-level fix for mojibake in main.py.

The file was written in UTF-8 but some tool encoded emojis as latin-1 byte sequences
stored as Unicode codepoints. This script reads raw bytes, detects mojibake sequences
(where latin-1 chars form valid UTF-8), and corrects them.
"""
import shutil

src = 'app/main.py'
backup = 'app/main.py.bak2'

# Read raw bytes
with open(src, 'rb') as f:
    raw = f.read()

print(f'File size: {len(raw)} bytes')

# Decode with latin-1 (lossless byte->char mapping)
latin1_str = raw.decode('latin-1')

# Re-encode back to bytes treating each char as latin-1
# (This should give us back the exact same bytes as 'raw')
assert latin1_str.encode('latin-1') == raw

# Now fix mojibake: find sequences of latin-1 chars that form valid UTF-8 codepoints
# and replace them with the proper Unicode character
def fix_mojibake(text):
    result = []
    i = 0
    while i < len(text):
        c = text[i]
        b = ord(c)  # Since we decoded as latin-1, ord(c) == byte value
        
        # Check if this byte is a UTF-8 lead byte (0xC2-0xF4)
        if 0xC2 <= b <= 0xF4:
            # Try to consume continuation bytes (0x80-0xBF)
            seq_bytes = [b]
            j = i + 1
            while j < len(text) and 0x80 <= ord(text[j]) <= 0xBF:
                seq_bytes.append(ord(text[j]))
                j += 1
            
            if len(seq_bytes) >= 2:  # Must have at least one continuation byte
                try:
                    decoded = bytes(seq_bytes).decode('utf-8')
                    result.append(decoded)
                    i = j
                    continue
                except (UnicodeDecodeError, ValueError):
                    pass
        
        # Regular char - keep as is (latin-1 to unicode is identity for 0x00-0xFF)
        # For ASCII range, just append directly
        if b < 0x80:
            result.append(c)
        else:
            # Non-ASCII latin-1 char that's NOT a UTF-8 lead byte pattern - keep original
            result.append(c)
        i += 1
    return ''.join(result)

# Process the latin-1 decoded string
fixed_str = fix_mojibake(latin1_str)

print(f'Original chars: {len(latin1_str)}')
print(f'Fixed chars: {len(fixed_str)}')

# Backup first
shutil.copy2(src, backup)
print(f'Backed up to {backup}')

# Write back as UTF-8
with open(src, 'w', encoding='utf-8', newline='') as f:
    f.write(fixed_str)

# Verify
with open(src, 'rb') as f:
    new_raw = f.read()

print(f'New file size: {len(new_raw)} bytes')

# Check a known area
with open(src, 'r', encoding='utf-8') as f:
    content = f.read()

idx = content.find('pages = [')
print('\nPages list check:')
snippet = content[idx:idx+250]
for ch in snippet[:80]:
    if ord(ch) > 127:
        print(f'  char: U+{ord(ch):04X} = {ch}')
print('Done!')
