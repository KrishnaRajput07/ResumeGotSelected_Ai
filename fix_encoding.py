"""Fix mojibake in main.py: latin-1 bytes of UTF-8 multi-byte sequences
were stored as Unicode code points. This script detects and corrects them."""
import shutil
import sys

def fix_mojibake(text):
    result = []
    i = 0
    chars = list(text)
    while i < len(chars):
        c = chars[i]
        cp = ord(c)
        # UTF-8 lead bytes (disguised as latin-1 chars): 0xC2-0xF4
        if 0xC2 <= cp <= 0xF4:
            seq = [cp]
            j = i + 1
            # Continuation bytes: 0x80-0xBF
            while j < len(chars) and 0x80 <= ord(chars[j]) <= 0xBF:
                seq.append(ord(chars[j]))
                j += 1
            try:
                decoded_char = bytes(seq).decode('utf-8')
                result.append(decoded_char)
                i = j
                continue
            except Exception:
                pass
        result.append(c)
        i += 1
    return ''.join(result)


src = 'app/main.py'
backup = 'app/main.py.bak'

shutil.copy2(src, backup)
print(f'Backed up to {backup}')

with open(src, 'r', encoding='utf-8') as f:
    content = f.read()

fixed = fix_mojibake(content)

# Sanity check
original_lines = content.count('\n')
fixed_lines = fixed.count('\n')
print(f'Original lines: {original_lines}, Fixed lines: {fixed_lines}')

changed = sum(1 for a, b in zip(content, fixed) if a != b)
print(f'Characters changed: {changed}, Length diff: {len(fixed) - len(content)}')

with open(src, 'w', encoding='utf-8') as f:
    f.write(fixed)

print('Done! main.py rewritten with correct UTF-8 encoding.')
