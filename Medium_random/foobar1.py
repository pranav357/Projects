def solution(s):
    # Your code here
    chars = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    braille = ['000000', '100000', '110000', '100100', '100110', '100010',
    '110100', '110110', '110010', '010100', '010110', '101000', '111000',
    '101100', '101110', '101010', '111100', '111110', '111010', '011100',
    '011110', '101001', '111001', '010111', '101101', '101111', '101011']

    s2 = ''
    for i in range(0, len(s)):
        if s[i].isupper():
            s2 = s2 + '000001' + s[i]
        else:
            s2=s2 + s[i]

    str_list=[]
    str_list[:0]=s2

    for j in range(0, len(str_list)):
        if (str_list[j].lower() in chars):
            str_list[j]=braille[chars.index(str_list[j].lower())]

    output=''.join(str_list)

    return output

s="Code"
a=solution(s)

len(output)
