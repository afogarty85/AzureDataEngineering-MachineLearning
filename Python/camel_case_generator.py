import re
def to_camel_case(s, user_acronyms):
    '''
    Camel Case Generator;

    Assumes you have a Pascal-cased string (e.g., TheQuickBrownFox)
    or a snake-cased string (e.g., the_quick_brown_fox)

    params::
    s: string
    user_acronyms: list; user-defined acronyms that should be set correctly,
    e.g., SKU, ID, etc

    Sample use:
    to_camel_case('TheQuickBrownFox', None)  # 'theQuickBrownFox'
    to_camel_case('The Quick Brown Fox', None)  # 'theQuickBrownFox'
    to_camel_case('Fru_MemorySPDSize', ['WMI', 'FRU'])  # FRUMemorySPDSize

    '''

    # apply user-specified acrnonym fixes
    if user_acronyms is not None:
        for acronym in user_acronyms:
            s = re.sub(acronym, acronym, s, flags=re.IGNORECASE)

    # handle snake_case
    if '_' in s:
        # split on snake
        s = s.split('_')
        # upper the first
        s = [s[0].upper() + s[1::] for s in s]
        # rejoin
        s = ''.join(s)

    # replace white space/dashes
    s = s.replace('-', '').replace(' ', '')

    # if we cant do anything; return
    if (all(s.lower()) == s) or (all(s.upper()) == s):
        return s

    # container
    word_holder = {}

    # find acronym positions
    acronym_positions = [(m.start(0), m.end(0)) for m in re.finditer(r'[A-Z]{1}[A-Z]*(?![a-z])', s)]

    # find acronyms
    acronyms = [re.findall(r'[A-Z]{1}[A-Z]*(?![a-z])', s)]

    # collapse lists
    acronyms = sum(acronyms, [])

    # find pascal text positions
    pascal_positions = [(m.start(0), m.end(0)) for m in re.finditer(r'[A-Z][a-z]+', s)]

    # find text that starts with capitals but has lowercase of any length after
    pascal_chars = [re.findall(r'[A-Z][a-z]+', s)]

    # collapse list
    pascal_chars = sum(pascal_chars, [])

    # find text that is lowercase and is followed by lowercase, start of str
    starting_chars = [re.findall(r'\b[a-z][a-z]+', s)]

    # collapse list
    starting_chars = sum(starting_chars, [])

    # find starting text positions
    starting_positions = [(m.start(0), m.end(0)) for m in re.finditer(r'\b[a-z][a-z]+', s)]

    # store positions and text
    for a_pos, a_word in zip(acronym_positions, acronyms):
        word_holder[a_pos] = a_word

    # store positions and text
    for p_pos, p_word in zip(pascal_positions, pascal_chars):
        # if the starting word is pascal; lower it
        if 0 in p_pos:
            p_word = p_word.lower()
            word_holder[p_pos] = p_word
        else:
            word_holder[p_pos] = p_word

    # store postions and text
    for s_pos, s_word in zip(starting_positions, starting_chars):
        # lower the starting word
        if 0 in s_pos:
            s_word = s_word.lower()
            word_holder[s_pos] = s_word
        else:
            word_holder[s_pos] = s_word

    # sort the dict
    word_holder = {key: word_holder[key] for key in sorted(word_holder.keys())}

    # assemble
    out = ''.join(word_holder.values())

    return out

# tests
case1 = 'ThisIsATest'
case2 = 'thisIsATest'
case3 = 'ThisIsATEst'
case4 = 'TheQuickBrownFox'
case5 = 'The Quick Brown Fox'
case6 = 'THISISATESTAndThisISATEstB'
case7 = 'thisIsATEstimatedT'
case8 = 'The_Quick_Brown_Fox'
case9 = 'the_quick_brown_fox'

assert to_camel_case(case1, None) == 'thisIsATest', 'failed'
assert to_camel_case(case2, None) == 'thisIsATest', 'failed'
assert to_camel_case(case3, None) == 'thisIsATEst', 'failed'
assert to_camel_case(case4, None) == 'theQuickBrownFox', 'failed'
assert to_camel_case(case5, None) == 'theQuickBrownFox', 'failed'
assert to_camel_case(case6, None) == 'THISISATESTAndThisISATEstB', 'failed'
assert to_camel_case(case7, None) == 'thisIsATEstimatedT', 'failed'
assert to_camel_case(case8, None) == 'theQuickBrownFox', 'failed'
assert to_camel_case(case9, None) == 'theQuickBrownFox', 'failed'

s1 = 'AssetRUHeight'
s2 = 'AssetUCnt'
s3 = 'MemorySpeed_Corrected'
s4 = 'Wmi_SMBIOSMemoryType'
s5 = 'Fru_MemorySPDSize'
s6 = 'Wmi_CS_SystemSKUNumber'
s7 = 'UefiDbx_UefiDbxKeyStatus'

assert to_camel_case(s1, None) == 'assetRUHeight', 'failed'
assert to_camel_case(s2, None) == 'assetUCnt', 'failed'
assert to_camel_case(s3, None) == 'memorySpeedCorrected', 'failed'
assert to_camel_case(s4, ['WMI']) == 'WMISMBIOSMemoryType', 'failed'
assert to_camel_case(s5, ['WMI', 'FRU']) == 'FRUMemorySPDSize', 'failed'
assert to_camel_case(s6, ['WMI', 'FRU', 'SKU']) == 'WMICSSystemSKUNumber', 'failed'
assert to_camel_case(s7, ['WMI', 'FRU', 'SKU', 'UEFI']) == 'UEFIDbxUEFIDbxKeyStatus', 'failed'
