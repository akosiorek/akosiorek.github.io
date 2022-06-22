def permutations(seq):
    perms = []

    if len(seq) == 0:
        perms.append([])
    else:
        for i in range(len(seq)):
            letter = seq[i]
            short_seq = seq[:]
            short_seq.pop(i)
            r = permutations(short_seq)
            for x in r:
                perms.append([letter] + x)

    return perms

chars = 'TU*IP'
char_list = list(chars)
res = permutations(char_list)
count = 0
for r in res:
    skip = False
    for i in range(len(r)):
        if r[i] == chars[i] and r[i] != '*':
            skip = True
    if not skip:
        print(''.join(r))
        count += 1
print(count)
