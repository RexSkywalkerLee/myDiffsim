read_file = open('flag.obj', 'r')
write_file = open('folded_flag.obj', 'w')

group1 = [60, 59, 73, 77,
          38, 23, 74, 75,
          21, 22, 34, 9,
          16, 15, 20, 10,
          13, 14, 11, 12,
          69, 37, 39, 41,
          35, 36, 40, 78,
          53, 48, 47, 64,
          54, 52, 46, 79]

group2 = [43, 72, 0, 61, 62, 45, 44, 63, 65]

group3 = [42, 55, 71, 25,
          2, 28, 26, 24,
          1, 29, 4, 3,
          51, 50, 5, 70,
          76, 49, 80, 7,
          57, 27, 8, 6,
          56, 18, 17, 58,
          33, 32, 19, 31,
          67, 68, 66, 30]

cnt = -3

for line in read_file:
    if cnt in group1:
        write_file.write(line)
    elif cnt in group2:
        line = line.split()
        write_file.write('v ')
        write_file.write(line[1])
        write_file.write(' 0.002195')
        write_file.write(' '+str(0.003776+0.005)+'\n')
    elif cnt in group3:
        line = line.split()
        write_file.write('v ')
        write_file.write(line[1])
        write_file.write(' '+str(2*0.002195-float(line[2])))
        write_file.write(' '+str(0.003776+0.01)+'\n')
    else:
        write_file.write(line)

    cnt += 1

read_file.close()
write_file.close()
