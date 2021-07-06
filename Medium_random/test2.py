from itertools import combinations

def solution(l):

    def find_no(nos_list, rem):
        nos_list.sort()
        for nos in nos_list:
            if nos%3 == rem:
                nos_list.remove(nos)
                break
        if sum(nos_list)%3 == 0:
            nos_list.sort(reverse = True)
            fin_no = int(''.join(str(ints) for ints in nos_list))
        else:    
            nos_list.sort(reverse = True)
            for i in reversed(range(1, len(nos_list) + 1)):
                for tup in combinations(nos_list, i):
                    if sum(tup)%3 == 0:
                        nos_list = list(tup)
                        break
            if ((sum(nos_list)%3 == 0) & (len(nos_list) != 0)):
                nos_list.sort(reverse = True)
                fin_no = int(''.join(str(ints) for ints in nos_list))
            else:
                fin_no = 0    
        return fin_no

    if ((sum(l) == 0) or (len(l) == 1 and l[0]%3 != 0)):
        fin_no = 0  
    elif sum(l)%3 == 0:
        l.sort(reverse=True)
        fin_no = int(''.join(str(ints) for ints in l))
    elif sum(l)%3 != 0:
        rem = sum(l)%3
        fin_no = find_no(l, rem)

    return fin_no

l = [3,1,5,1,4,9]
sum(l)

solution(l)

for i in reversed(range(1, len(l)+1)):
    print(i)    

tup = (3,2,1)
sum(tup)

nos = [9,8,7,6,5,4,3,2,1]

for i in reversed(range(1,9)):
    print(i)
nos_list = l
            nos_list.sort(reverse = True)
            for i in reversed(range(1, len(nos_list) + 1)):
                print("First loop")
                for tup in combinations(nos_list, i):
                    print(tup)
                    if sum(tup)%3 == 0:
                        print("Divisible by 3", tup)
                        nos_list = list(tup)
                        break
                    break        
            print("Done")


def calc_counts(c1, c2):
    if c2 == 0:
        return [3*int(c1/3), 0]
    diff = c1 - c2
    extra = diff % 3
    if extra == 2:
        return [c1, c2-1]
    elif extra == 1:
        return [c1 - 1, c2]
    else:
        return [c1, c2]

def form_array(h, count):
    c1, c2 = count
    ret = h[0][:]
    if c1 == 0 and c2 == 0:
        return ret
    if c1 != 0:
        a1 = h[1][:]
        a1.sort(reverse=True)
        ret.extend(a1[:c1])
    if c2 != 0:
        a2 = h[2][:]
        a2.sort(reverse=True)
        ret.extend(a2[:c2])
    return ret

def solution(l):
    h = {0: [], 1:[], 2: []}
    for i in l:
        h[i%3].append(i)
    count_1 = len(h[1])
    count_2 = len(h[2])
    arr = []
    if count_1 == count_2:
        # all elements
        arr = l
    elif count_1 > count_2:
        count_1, count_2 = calc_counts(count_1, count_2)
        arr = form_array(h, [count_1, count_2])
    else:
        count_2, count_1 = calc_counts(count_2, count_1)
        arr = form_array(h, [count_1, count_2])
    arr.sort(reverse=True)
    ret = ''.join(list(map(str, arr)))
    if len(ret) == 0:
        return 0
    return int(ret)

def solution(l):
    lLen = len(l)
    max = 1 << lLen
    l.sort(reverse=True)
    maxAttempt = 0
    for mask in reversed(range(max)):
        attempt = ""
        for index in range(mask.bit_length()):
            attempt += str(l[index]) if (mask >> index) & 1 == 1 else ''
        if attempt == '':
            attempt = 0
        attempt = int(attempt)
        if attempt % 3 == 0:
            if attempt > maxAttempt:
                maxAttempt = attempt

    return maxAttempt    