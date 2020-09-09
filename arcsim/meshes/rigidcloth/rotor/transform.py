read_file = open('ball.obj', 'r')
write_file = open('rotor.obj', 'w')

for line in read_file:
    if line[:2] != 'v ':
        write_file.write(line)
    else:
        line = line.split()
        write_file.write('v ')
        write_file.write(str(0.99+float(line[1])))
        write_file.write(str(-0.0+float(line[2])))
        write_file.write(' '+str(-0.2+float(line[3]))+'\n')

read_file.close()
write_file.close()
