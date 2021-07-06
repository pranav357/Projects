def swap_case(s):
    for i in range(0,len(s)):
        if s[i].isupper():
            s = s[:i] + s[i].lower() + s[i+1:]
        else:
            s = s[:i] + s[i].upper() + s[i+1:]
    return  s


s = 'HackerRank.com presents "Pythonist 2".'

swap_case(s)


