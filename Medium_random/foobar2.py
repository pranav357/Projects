eles = ["1.1.2", "1.0", "1.3.3", 
"1.0.12", "1.0.2"]
eles.sort()

eles


new_versl = []

for ver in eles:
    if ver.count('.') >= 1:
        pos = [i for i, cha in enumerate(ver) if cha == "."]
        pos.sort(reverse=True)
        for ind in pos:
            if ((len(ver[ind:]) == 2) | (ver[ind:ind+3].count('.') == 2)):
                ver = ver[:ind+1] + "0" + ver[ind+1:]
        new_versl.append(ver)
    else:
        new_versl.append(ver)    

new_versl.sort()

sort_versl = []

for vers in new_versl:
    if vers.count('.') >= 1:
        pos = [i for i, cha in enumerate(vers) if cha == "."]
        pos.sort(reverse=True) 
        for ind in pos:
            if ((vers[ind+1] == '0') & (vers[ind+1:ind+3].count('.') == 0)):
                vers = vers[:ind+1] + vers[ind+2:]
        sort_versl.append(vers)
    else:
        sort_versl.append(vers)

